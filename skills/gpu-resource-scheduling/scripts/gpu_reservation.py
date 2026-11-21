#!/usr/bin/env python3
"""FIFO short-lease coordination for shared physical GPUs.

The script is intended to run on 5090. It coordinates reservations under an
explicit remote root and performs a final nvidia-smi readiness check before a
lease becomes active. It does not replace the project's CUDA execution queue.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid


DEFAULT_ROOT = "/data03/liupl/code-opt/gpu-reservations"
DEFAULT_MAX_ESTIMATE = 3600
DEFAULT_POLL = 10
DEFAULT_STALE_AFTER = 120
PURPOSES = ("correctness", "tuning", "profile", "replay", "integration", "acceptance", "calibration")


def now() -> float:
    return time.time()


def iso(value: float | None) -> str | None:
    if value is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def canonical_uuid(value: str) -> str:
    value = value.strip()
    bare = value[4:] if value[:4].upper() == "GPU-" else value
    if not bare or any(char not in "0123456789abcdefABCDEF-" for char in bare):
        raise ValueError("malformed GPU UUID")
    return "GPU-" + bare.lower()


def gpu_directory(root: Path, gpu_uuid: str) -> Path:
    digest = hashlib.sha256(canonical_uuid(gpu_uuid).encode()).hexdigest()
    return root / digest


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def gpu_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / "reservation.lock"
    with lock_path.open("a+b") as stream:
        if os.name == "nt":
            raise RuntimeError("GPU reservation must run on the remote Linux host")
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def process_start_ticks(pid: int) -> int | None:
    if os.name == "nt":
        return None
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        return int(text.rsplit(")", 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def process_is_same(pid: int, start_ticks: int | None) -> bool:
    if process_start_ticks(pid) is None:
        return False
    return start_ticks is None or process_start_ticks(pid) == start_ticks


def probe_device(gpu_uuid: str) -> dict:
    """Return readiness without creating a CUDA context."""
    uuid_value = canonical_uuid(gpu_uuid)
    try:
        gpu_text = subprocess.check_output(
            ["nvidia-smi", "-i", uuid_value, "--query-gpu=uuid,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT,
        ).strip()
        rows = [item.strip() for item in gpu_text.split(",")]
        if len(rows) != 3 or canonical_uuid(rows[0]) != uuid_value:
            raise RuntimeError("nvidia-smi returned a different GPU")
        utilization, memory = int(rows[1]), int(rows[2])
        app_text = subprocess.check_output(
            ["nvidia-smi", "-i", uuid_value, "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT,
        )
        pids = [int(line.strip()) for line in app_text.splitlines() if line.strip()]
        return {"gpu_uuid": uuid_value, "gpu_utilization_percent": utilization,
                "memory_used_mib": memory, "compute_process_ids": sorted(pids),
                "other_compute_processes": len(pids), "ready": not pids and utilization == 0}
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        return {"gpu_uuid": uuid_value, "ready": False, "probe_error": f"{type(error).__name__}: {error}"}


def event(directory: Path, reservation_id: str, name: str, **details) -> str:
    event_id = f"{time.time_ns()}-{uuid.uuid4().hex}"
    payload = {"schema": "skopa-gpu-reservation-event-v1", "event_id": event_id,
               "reservation_id": reservation_id, "event": name, "at": now(), "at_utc": iso(now()), **details}
    atomic_json(directory / "events" / f"{event_id}.json", payload)
    return event_id


def read_records(directory: Path, state: str) -> list[tuple[Path, dict]]:
    folder = directory / state
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(folder.glob("*.json")):
        try:
            rows.append((path, json.loads(path.read_text(encoding="utf8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def reclaim_stale(directory: Path, stale_after: float) -> int:
    reclaimed = 0
    for path, record in read_records(directory, "pending"):
        requested = float(record.get("requested_at", 0))
        owner_pid = record.get("owner_pid")
        owner_ticks = record.get("owner_start_ticks")
        if now() - requested <= stale_after or (isinstance(owner_pid, int) and process_is_same(owner_pid, owner_ticks)):
            continue
        record.update(state="canceled", canceled_at=now(), canceled_at_utc=iso(now()),
                      recovery_reason="requester_dead_and_pending_request_stale")
        history = directory / "history" / path.name
        atomic_json(history, record)
        path.unlink(missing_ok=True)
        event(directory, record["reservation_id"], "canceled", recovery_reason=record["recovery_reason"])
        reclaimed += 1
    for path, record in read_records(directory, "active"):
        heartbeat = float(record.get("heartbeat_at", record.get("activated_at", 0)))
        owner_pid = record.get("owner_pid")
        owner_ticks = record.get("owner_start_ticks")
        if now() - heartbeat <= stale_after or (isinstance(owner_pid, int) and process_is_same(owner_pid, owner_ticks)):
            if record.get("expected_end_at", 0) < now() and record.get("state") == "active":
                record["state"] = "overdue"
                atomic_json(path, record)
            continue
        record.update(state="abandoned", abandoned_at=now(), abandoned_at_utc=iso(now()),
                      recovery_reason="owner_dead_and_heartbeat_stale")
        history = directory / "history" / path.name
        atomic_json(history, record)
        path.unlink(missing_ok=True)
        event(directory, record["reservation_id"], "abandoned", recovery_reason=record["recovery_reason"])
        reclaimed += 1
    return reclaimed


def validate_estimate(value: float, maximum: float) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("estimated duration must be positive")
    if value > maximum:
        raise ValueError(f"estimated duration exceeds the configured short-lease limit of {maximum:g}s")
    return float(value)


def create_pending(args) -> tuple[Path, Path, dict, str]:
    root = Path(args.root).expanduser().resolve()
    gpu_uuid = canonical_uuid(args.gpu_uuid)
    directory = gpu_directory(root, gpu_uuid)
    reservation_id = uuid.uuid4().hex
    token = secrets.token_urlsafe(32)
    requested = now()
    record = {
        "schema": "skopa-gpu-reservation-v1", "reservation_id": reservation_id,
        "task_id": args.task_id, "gpu_uuid": gpu_uuid, "purpose": args.purpose,
        "owner_name": args.owner_name,
        "state": "pending", "requested_at": requested, "requested_at_utc": iso(requested),
        "estimated_duration_s": args.estimate_seconds, "requested_end_at": requested + args.estimate_seconds,
        "owner_pid": os.getpid(), "owner_start_ticks": process_start_ticks(os.getpid()),
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(), "created_by": "gpu_reservation.py",
        "poll_seconds": args.poll_seconds, "stale_after_seconds": args.stale_after_seconds,
    }
    with gpu_lock(directory):
        atomic_json(directory / "pending" / f"{reservation_id}.json", record)
        event(directory, reservation_id, "registered", task_id=args.task_id, purpose=args.purpose,
              estimated_duration_s=args.estimate_seconds)
    return root, directory, record, token


def activate(args, directory: Path, record: dict, token: str) -> dict | None:
    with gpu_lock(directory):
        reclaim_stale(directory, args.stale_after_seconds)
        pending_path = directory / "pending" / f"{record['reservation_id']}.json"
        if not pending_path.exists():
            raise RuntimeError("reservation is no longer pending")
        active = read_records(directory, "active")
        earlier = [item for path, item in read_records(directory, "pending")
                   if item["reservation_id"] != record["reservation_id"]
                   and item.get("state") == "pending"
                   and (item.get("requested_at", 0), item["reservation_id"]) <
                   (record.get("requested_at", 0), record["reservation_id"])]
        if active or earlier:
            return None
        readiness = probe_device(record["gpu_uuid"])
        record["last_device_preflight"] = readiness
        if not readiness.get("ready"):
            atomic_json(pending_path, record)
            return None
        activated = now()
        record.update(state="active", activated_at=activated, activated_at_utc=iso(activated),
                      expected_end_at=activated + record["estimated_duration_s"], heartbeat_at=activated,
                      heartbeat_at_utc=iso(activated), device_preflight=readiness)
        active_path = directory / "active" / pending_path.name
        atomic_json(active_path, record)
        pending_path.unlink(missing_ok=True)
        event(directory, record["reservation_id"], "activated", device_preflight=readiness)
        return {"reservation_id": record["reservation_id"], "token": token, "gpu_uuid": record["gpu_uuid"],
                "task_id": record["task_id"], "purpose": record["purpose"], "activated_at": activated,
                "expected_end_at": record["expected_end_at"], "device_preflight": readiness}


def acquire(args) -> dict:
    args.estimate_seconds = validate_estimate(args.estimate_seconds, args.max_estimate_seconds)
    _, directory, record, token = create_pending(args)
    try:
        while True:
            result = activate(args, directory, record, token)
            if result is not None:
                return result
            time.sleep(args.poll_seconds)
    except BaseException:
        cancel_args = argparse.Namespace(root=args.root, gpu_uuid=args.gpu_uuid,
                                         reservation_id=record["reservation_id"], token=token)
        try:
            cancel_pending(cancel_args)
        finally:
            raise


def _load_active(args) -> tuple[Path, Path, dict]:
    directory = gpu_directory(Path(args.root).expanduser().resolve(), canonical_uuid(args.gpu_uuid))
    path = directory / "active" / f"{args.reservation_id}.json"
    if not path.exists():
        raise ValueError("active reservation not found")
    record = json.loads(path.read_text(encoding="utf8"))
    return directory, path, record


def _load_pending(args) -> tuple[Path, Path, dict]:
    directory = gpu_directory(Path(args.root).expanduser().resolve(), canonical_uuid(args.gpu_uuid))
    path = directory / "pending" / f"{args.reservation_id}.json"
    if not path.exists():
        raise ValueError("pending reservation not found")
    record = json.loads(path.read_text(encoding="utf8"))
    return directory, path, record


def check_token(record: dict, token: str) -> None:
    if hashlib.sha256(token.encode()).hexdigest() != record.get("token_sha256"):
        raise ValueError("reservation token does not match")


def heartbeat(args) -> dict:
    directory, path, record = _load_active(args)
    with gpu_lock(directory):
        if not path.exists():
            raise ValueError("active reservation was already released")
        record = json.loads(path.read_text(encoding="utf8"))
        check_token(record, args.token)
        current = now()
        state = "overdue" if current > record.get("expected_end_at", current) else record.get("state", "active")
        record.update(state=state, heartbeat_at=current, heartbeat_at_utc=iso(current))
        atomic_json(path, record)
        event(directory, record["reservation_id"], "heartbeat", state=state)
        return {"reservation_id": record["reservation_id"], "state": state, "heartbeat_at": current}


def release(args, exit_code: int | None = None) -> dict:
    directory, path, record = _load_active(args)
    with gpu_lock(directory):
        if not path.exists():
            raise ValueError("active reservation was already released")
        record = json.loads(path.read_text(encoding="utf8"))
        check_token(record, args.token)
        finished = now()
        record.update(state="released", released_at=finished, released_at_utc=iso(finished),
                      actual_duration_s=finished - record["activated_at"], exit_code=exit_code)
        history = directory / "history" / path.name
        atomic_json(history, record)
        path.unlink(missing_ok=True)
        event(directory, record["reservation_id"], "released", exit_code=exit_code,
              actual_duration_s=record["actual_duration_s"])
        return {"reservation_id": record["reservation_id"], "state": "released",
                "actual_duration_s": record["actual_duration_s"], "exit_code": exit_code}


def cancel_pending(args) -> dict:
    directory, path, record = _load_pending(args)
    with gpu_lock(directory):
        if not path.exists():
            raise ValueError("pending reservation was already canceled or activated")
        record = json.loads(path.read_text(encoding="utf8"))
        check_token(record, args.token)
        canceled = now()
        record.update(state="canceled", canceled_at=canceled, canceled_at_utc=iso(canceled),
                      recovery_reason="requester_canceled_pending_request")
        atomic_json(directory / "history" / path.name, record)
        path.unlink(missing_ok=True)
        event(directory, record["reservation_id"], "canceled", recovery_reason=record["recovery_reason"])
        return {"reservation_id": record["reservation_id"], "state": "canceled"}


def status(args) -> dict:
    directory = gpu_directory(Path(args.root).expanduser().resolve(), canonical_uuid(args.gpu_uuid))
    with gpu_lock(directory):
        reclaimed = reclaim_stale(directory, args.stale_after_seconds)
        active = [hide_secret(row) for _, row in read_records(directory, "active")]
        pending = [hide_secret(row) for _, row in read_records(directory, "pending")]
    return {"gpu_uuid": canonical_uuid(args.gpu_uuid), "active": active, "pending": pending,
            "reclaimed": reclaimed, "observed_at": now(), "observed_at_utc": iso(now())}


def hide_secret(record: dict) -> dict:
    result = dict(record)
    result.pop("token_sha256", None)
    return result


def run(args) -> int:
    if not args.command:
        raise ValueError("run requires a command after --")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    lease = acquire(args)
    token = lease.pop("token")
    stop = threading.Event()
    child = None

    def keep_alive():
        while not stop.wait(max(1.0, min(args.poll_seconds, 30.0))):
            try:
                heartbeat_args = argparse.Namespace(root=args.root, gpu_uuid=args.gpu_uuid,
                    reservation_id=lease["reservation_id"], token=token,
                    stale_after_seconds=args.stale_after_seconds)
                heartbeat(heartbeat_args)
            except Exception:
                return

    environment = dict(os.environ)
    environment["SKOPA_GPU_UUID"] = lease["gpu_uuid"]
    environment["CUDA_VISIBLE_DEVICES"] = lease["gpu_uuid"]
    environment["SKOPA_GPU_RESERVATION_ID"] = lease["reservation_id"]
    heartbeat_thread = threading.Thread(target=keep_alive, name="gpu-reservation-heartbeat", daemon=True)
    try:
        child = subprocess.Popen(command, env=environment, start_new_session=(os.name != "nt"))
        heartbeat_thread.start()
        code = child.wait()
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            child.wait()
        code = 130
    finally:
        stop.set()
        heartbeat_thread.join(timeout=2)
        release_args = argparse.Namespace(root=args.root, gpu_uuid=args.gpu_uuid,
            reservation_id=lease["reservation_id"], token=token)
        release(release_args, code if "code" in locals() else 1)
    return code


def add_common(parser: argparse.ArgumentParser, *, estimate: bool = False) -> None:
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--stale-after-seconds", type=float, default=DEFAULT_STALE_AFTER)
    if estimate:
        parser.add_argument("--owner-name", required=True)
        parser.add_argument("--task-id", required=True)
        parser.add_argument("--purpose", choices=PURPOSES, required=True)
        parser.add_argument("--estimate-seconds", type=float, required=True)
        parser.add_argument("--max-estimate-seconds", type=float, default=DEFAULT_MAX_ESTIMATE)
        parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command_name", required=True)
    acquire_parser = commands.add_parser("acquire")
    add_common(acquire_parser, estimate=True)
    release_parser = commands.add_parser("release")
    add_common(release_parser)
    release_parser.add_argument("--reservation-id", required=True)
    release_parser.add_argument("--token", required=True)
    heartbeat_parser = commands.add_parser("heartbeat")
    add_common(heartbeat_parser)
    heartbeat_parser.add_argument("--reservation-id", required=True)
    heartbeat_parser.add_argument("--token", required=True)
    cancel_parser = commands.add_parser("cancel")
    add_common(cancel_parser)
    cancel_parser.add_argument("--reservation-id", required=True)
    cancel_parser.add_argument("--token", required=True)
    status_parser = commands.add_parser("status")
    add_common(status_parser)
    run_parser = commands.add_parser("run")
    add_common(run_parser, estimate=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command_name == "acquire":
        result = acquire(args)
    elif args.command_name == "release":
        result = release(args)
    elif args.command_name == "heartbeat":
        result = heartbeat(args)
    elif args.command_name == "cancel":
        result = cancel_pending(args)
    elif args.command_name == "status":
        result = status(args)
    else:
        return run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"gpu reservation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
