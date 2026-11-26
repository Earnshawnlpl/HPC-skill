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
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid

import gpu_pulse


DEFAULT_ROOT = "/data03/liupl/code-opt/gpu-reservations"
DEFAULT_MAX_ESTIMATE = 3600
DEFAULT_POLL = 10
DEFAULT_STALE_AFTER = 120
PULSE_WAIT_POLL = 30
PURPOSES = ("correctness", "tuning", "profile", "replay", "integration", "acceptance", "calibration")


class ReservationWindowEnding(RuntimeError):
    """The active slot cannot accommodate another GPU callback."""


class ReservationUnavailable(RuntimeError):
    """The same local and, if configured, calendar lease could not be confirmed."""


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
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
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
    pulse_url, pulse_id = getattr(args, "pulse_url", None), getattr(args, "pulse_gpu_id", None)
    if pulse_url is not None or pulse_id is not None:
        if not pulse_url or not pulse_id:
            raise ValueError("GPU Pulse needs both --pulse-url and --pulse-gpu-id")
        record["pulse_config"] = gpu_pulse.configuration(pulse_url, pulse_id, gpu_uuid)
        if record["estimated_duration_s"] > DEFAULT_MAX_ESTIMATE:
            raise ValueError("GPU Pulse reservation must be at most one hour")
        title = getattr(args, "pulse_title", None)
        if title is not None:
            if not title.strip():
                raise ValueError("GPU Pulse title must not be empty")
            record["pulse_title"] = title.strip()
        minimum = getattr(args, "pulse_minimum_seconds", None)
        if minimum is not None:
            record["pulse_minimum_ms"] = round(validate_estimate(minimum, args.estimate_seconds) * 1000)
            if record["pulse_minimum_ms"] <= 0:
                raise ValueError("GPU Pulse minimum duration is below one millisecond")
    with gpu_lock(directory):
        atomic_json(directory / "pending" / f"{reservation_id}.json", record)
        event(directory, reservation_id, "registered", task_id=args.task_id, purpose=args.purpose,
              estimated_duration_s=args.estimate_seconds)
    return root, directory, record, token


def record_wait(pending_path: Path, record: dict, reason: str) -> None:
    observed = now()
    if record.get("last_wait_reason") != reason:
        record["wait_reason_since"] = observed
    record.update(last_wait_reason=reason, last_wait_at=observed)
    atomic_json(pending_path, record)


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
            record_wait(pending_path, record, "prior_lease")
            return None
        readiness = probe_device(record["gpu_uuid"])
        record["last_device_preflight"] = readiness
        if not readiness.get("ready"):
            if getattr(args, "pulse_wait", False) and readiness.get("probe_error"):
                raise RuntimeError("Unable to verify the selected GPU before Pulse reservation")
            record_wait(pending_path, record, "device_unverified" if readiness.get("probe_error") else "device_busy")
            return None
        if "pulse_config" in record:
            wait_reason = "pulse_conflict"
            try:
                receipt = gpu_pulse.reserve(directory, record, atomic_json)
                event(directory, record["reservation_id"], "pulse_reserved", group_id=receipt["group_id"],
                      start_ms=receipt["start_ms"], end_ms=receipt["end_ms"])
                readiness = probe_device(record["gpu_uuid"])
                record["last_device_preflight"] = readiness
                if readiness.get("probe_error"):
                    raise RuntimeError("Unable to verify the selected GPU after Pulse reservation")
                if not readiness.get("ready"):
                    wait_reason = "device_busy"
                    raise gpu_pulse.PulseConflict("Device readiness changed before activation")
                if not receipt["start_ms"] <= now() * 1000 < receipt["end_ms"]:
                    wait_reason = "pulse_window_changed"
                    raise gpu_pulse.PulseConflict("GPU Pulse window changed before activation")
                record["pulse_receipt"] = receipt
            except gpu_pulse.PulseConflict:
                if not getattr(args, "pulse_wait", False):
                    raise
                outcome = finish_pulse(directory, record)
                gpu_pulse.archive_attempt(directory, record, outcome=outcome, write_json=atomic_json)
                record_wait(pending_path, record, wait_reason)
                return None
        activated = now()
        deadline = (record["pulse_receipt"]["end_ms"] / 1000 if "pulse_receipt" in record
                    else activated + record["estimated_duration_s"])
        record.update(state="active", activated_at=activated, activated_at_utc=iso(activated),
                      expected_end_at=deadline, heartbeat_at=activated,
                      heartbeat_at_utc=iso(activated), device_preflight=readiness)
        active_path = directory / "active" / pending_path.name
        atomic_json(active_path, record)
        pending_path.unlink(missing_ok=True)
        event(directory, record["reservation_id"], "activated", device_preflight=readiness)
        return {"reservation_id": record["reservation_id"], "token": token, "gpu_uuid": record["gpu_uuid"],
                "task_id": record["task_id"], "purpose": record["purpose"], "activated_at": activated,
                "expected_end_at": record["expected_end_at"], "device_preflight": readiness,
                **({"expected_end_ms": record["pulse_receipt"]["end_ms"]} if "pulse_receipt" in record else {})}


def acquire(args) -> dict:
    args.estimate_seconds = validate_estimate(args.estimate_seconds, args.max_estimate_seconds)
    pulse_wait = getattr(args, "pulse_wait", False)
    eligible = getattr(args, "eligible_gpu_uuid", None)
    poll_seconds = args.poll_seconds
    if poll_seconds is None:
        poll_seconds = PULSE_WAIT_POLL if pulse_wait or eligible else DEFAULT_POLL
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("Reservation polling interval must be positive and finite")
    args.poll_seconds = poll_seconds
    if eligible:
        if getattr(args, "gpu_uuid", None) or not getattr(args, "pulse_url", None) or getattr(args, "pulse_gpu_id", None):
            raise ValueError("Flexible mode requires --pulse-url and eligible UUIDs instead of fixed GPU flags")
        uuids = [canonical_uuid(value) for value in eligible]
        if len(uuids) < 2 or len(set(uuids)) != len(uuids):
            raise ValueError("Flexible mode requires two or more distinct eligible physical GPU UUIDs")
        for value in uuids:
            gpu_pulse.configuration(args.pulse_url, value, value)
        choices = [argparse.Namespace(**{**vars(args), "gpu_uuid": value, "pulse_gpu_id": value, "pulse_wait": True})
                   for value in uuids]
    else:
        choices = [args]
    if not getattr(args, "pulse_url", None) and (pulse_wait or getattr(args, "pulse_title", None) is not None
                                                or getattr(args, "pulse_minimum_seconds", None) is not None):
        raise ValueError("Pulse title, waiting, and minimum duration require --pulse-url")
    pending = []
    selected = None
    try:
        for choice in choices:
            _, directory, record, token = create_pending(choice)
            pending.append((choice, directory, record, token))
        while True:
            for entry in pending:
                choice, directory, record, token = entry
                result = activate(choice, directory, record, token)
                if result is not None:
                    selected = entry
                    return result
            time.sleep(poll_seconds)
    finally:
        cleanup_error = None
        for entry in pending:
            if entry is selected:
                continue
            choice, _, record, token = entry
            try:
                cancel_pending(argparse.Namespace(root=args.root, gpu_uuid=choice.gpu_uuid,
                               reservation_id=record["reservation_id"], token=token))
            except (RuntimeError, OSError, ValueError) as error:
                cleanup_error = error
        if cleanup_error is not None and selected is not None:
            chosen, _, record, token = selected
            release(argparse.Namespace(root=args.root, gpu_uuid=chosen.gpu_uuid,
                    reservation_id=record["reservation_id"], token=token), 1)
            raise cleanup_error


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
        finish_pulse(directory, record)
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
        finish_pulse(directory, record)
        canceled = now()
        record.update(state="canceled", canceled_at=canceled, canceled_at_utc=iso(canceled),
                      recovery_reason="requester_canceled_pending_request")
        atomic_json(directory / "history" / path.name, record)
        path.unlink(missing_ok=True)
        event(directory, record["reservation_id"], "canceled", recovery_reason=record["recovery_reason"])
        return {"reservation_id": record["reservation_id"], "state": "canceled"}


def finish_pulse(directory, record):
    if "pulse_config" not in record:
        return
    try:
        result = gpu_pulse.cancel(directory, record, atomic_json)
    except (RuntimeError, OSError, ValueError):
        event(directory, record["reservation_id"], "pulse_cleanup_unresolved")
        raise
    if result is not None:
        event(directory, record["reservation_id"], "pulse_cleanup", outcome=result)
    return result


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


def verify_active_reservation(root, gpu_uuid, reservation_id, *, minimum_remaining_seconds=0):
    """Read the active lease and calendar immediately before a trusted queue callback."""
    gpu_uuid = canonical_uuid(gpu_uuid)
    if not re.fullmatch(r"[0-9a-f]{32}", reservation_id or ""):
        raise ValueError("Invalid GPU reservation ID")
    if not math.isfinite(minimum_remaining_seconds) or minimum_remaining_seconds < 0:
        raise ValueError("Minimum remaining time must be nonnegative and finite")
    directory = gpu_directory(Path(root).expanduser().resolve(), gpu_uuid)
    with gpu_lock(directory):
        path = directory / "active" / f"{reservation_id}.json"
        if not path.is_file():
            raise ReservationUnavailable("Matching GPU reservation is not active")
        record = json.loads(path.read_text(encoding="utf8"))
        if (record.get("reservation_id") != reservation_id or record.get("gpu_uuid") != gpu_uuid
                or record.get("state") not in {"active", "overdue"}
                or not process_is_same(record["owner_pid"], record["owner_start_ticks"])):
            raise ReservationUnavailable("Matching GPU reservation is not active")
        receipt = record.get("pulse_receipt")
        end_ms = receipt["end_ms"] if receipt else round(record["expected_end_at"] * 1000)

        def check_window():
            current_ms = now() * 1000
            if (receipt and current_ms < receipt["start_ms"]) or current_ms >= end_ms:
                raise ReservationWindowEnding("GPU reservation window is not current")
            if end_ms - current_ms < minimum_remaining_seconds * 1000:
                raise ReservationWindowEnding("GPU reservation has insufficient remaining time for the next callback")

        check_window()
        if "pulse_config" in record:
            config = record["pulse_config"]
            if not receipt or gpu_pulse.configuration(config["url"], receipt["gpu_id"], gpu_uuid) != config:
                raise ReservationUnavailable("Active lease is missing its matching GPU Pulse receipt")
            try:
                gpu_pulse.verify_reservation(directory, record, receipt)
            except gpu_pulse.PulseReservationUnavailable as error:
                raise ReservationUnavailable(str(error)) from error
            check_window()
        return {"reservation_id": reservation_id, "gpu_uuid": gpu_uuid, "expected_end_ms": end_ms,
                "pulse_verified": receipt is not None}


def acquire_for_launch(args):
    """Reconcile verified prelaunch refusals before acquiring another slot."""
    wait = getattr(args, "pulse_wait", False) or getattr(args, "eligible_gpu_uuid", None)
    while True:
        lease = acquire(args)
        if not getattr(args, "pulse_url", None):
            return lease
        release_args = argparse.Namespace(root=args.root, gpu_uuid=lease["gpu_uuid"],
                                          reservation_id=lease["reservation_id"], token=lease["token"])
        try:
            verify_active_reservation(args.root, lease["gpu_uuid"], lease["reservation_id"])
        except (ReservationWindowEnding, ReservationUnavailable, gpu_pulse.PulseConflict) as error:
            release(release_args)
            directory = gpu_directory(Path(args.root).expanduser().resolve(), lease["gpu_uuid"])
            event(directory, lease["reservation_id"], "pulse_prelaunch_refused", reason=type(error).__name__,
                  child_started=False)
            if not wait:
                raise
            time.sleep(args.poll_seconds)
        except BaseException:
            release(release_args)
            raise
        else:
            return lease


def run(args) -> int:
    if not args.command:
        raise ValueError("run requires a command after --")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("run requires a command after --")
    lease = acquire_for_launch(args)
    token = lease.pop("token")
    stop = threading.Event()
    child = None

    def keep_alive():
        while not stop.wait(max(1.0, min(args.poll_seconds, 30.0))):
            try:
                heartbeat_args = argparse.Namespace(root=args.root, gpu_uuid=lease["gpu_uuid"],
                    reservation_id=lease["reservation_id"], token=token,
                    stale_after_seconds=args.stale_after_seconds)
                heartbeat(heartbeat_args)
            except Exception:
                return

    environment = dict(os.environ)
    environment["SKOPA_GPU_UUID"] = lease["gpu_uuid"]
    environment["CUDA_VISIBLE_DEVICES"] = lease["gpu_uuid"]
    environment["SKOPA_GPU_RESERVATION_ID"] = lease["reservation_id"]
    environment["SKOPA_GPU_RESERVATION_ROOT"] = str(Path(args.root).expanduser().resolve())
    environment["SKOPA_GPU_RESERVATION_END_MS"] = str(lease.get("expected_end_ms", int(lease["expected_end_at"] * 1000)))
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
        if heartbeat_thread.is_alive():
            heartbeat_thread.join(timeout=2)
        release_args = argparse.Namespace(root=args.root, gpu_uuid=lease["gpu_uuid"],
            reservation_id=lease["reservation_id"], token=token)
        release(release_args, code if "code" in locals() else 1)
    return code


def add_common(parser: argparse.ArgumentParser, *, estimate: bool = False) -> None:
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--stale-after-seconds", type=float, default=DEFAULT_STALE_AFTER)
    if estimate:
        gpu_selection = parser.add_mutually_exclusive_group(required=True)
        gpu_selection.add_argument("--gpu-uuid")
        gpu_selection.add_argument("--eligible-gpu-uuid", action="append", help="Physical UUID for a flexible development task. Repeat for each candidate.")
        parser.add_argument("--owner-name", required=True)
        parser.add_argument("--task-id", required=True)
        parser.add_argument("--purpose", choices=PURPOSES, required=True)
        parser.add_argument("--estimate-seconds", type=float, required=True)
        parser.add_argument("--max-estimate-seconds", type=float, default=DEFAULT_MAX_ESTIMATE)
        parser.add_argument("--poll-seconds", type=float, help="Wait interval in seconds. Defaults to 30 in Pulse wait mode.")
        parser.add_argument("--pulse-url", help="Optional GPU Pulse HTTP(S) origin")
        parser.add_argument("--pulse-gpu-id", help="Exact physical GPU ID on GPU Pulse")
        parser.add_argument("--pulse-title", help="Public calendar title (defaults to task ID)")
        parser.add_argument("--pulse-wait", action="store_true", help="Wait and retry verified readiness or calendar conflicts")
        parser.add_argument("--pulse-minimum-seconds", type=float, help="Allow shorter available calendar windows down to this minimum")
    else:
        parser.add_argument("--gpu-uuid", required=True)


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
    check_parser = commands.add_parser("check")
    add_common(check_parser)
    check_parser.add_argument("--reservation-id", required=True)
    check_parser.add_argument("--minimum-remaining-seconds", type=float, default=0)
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
    elif args.command_name == "check":
        result = verify_active_reservation(args.root, args.gpu_uuid, args.reservation_id,
                                           minimum_remaining_seconds=args.minimum_remaining_seconds)
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
