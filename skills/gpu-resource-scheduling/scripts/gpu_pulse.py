"""Optional GPU Pulse calendar reservation for one verified physical UUID."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_SLOT_MS = 3_600_000
UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE)


class PulseConflict(RuntimeError):
    """An overlapping reservation blocks activation."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def configuration(url, gpu_id, gpu_uuid):
    parts = urlsplit(url)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.path not in {"", "/"} or parts.query or parts.fragment):
        raise ValueError("GPU Pulse requires an explicit HTTP(S) origin")
    if not UUID.fullmatch(gpu_id or "") or gpu_id.lower() != gpu_uuid.lower():
        raise ValueError("Pulse GPU ID must match the selected physical GPU UUID")
    return {"url": url.rstrip("/"), "gpu_id": gpu_id}


def request(config, method, endpoint, *, query=None, body=None):
    url = config["url"] + endpoint
    if query:
        url += "?" + urlencode(query)
    headers = {"Cache-Control": "no-store", "Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    try:
        with build_opener(NoRedirect()).open(Request(url, data=data, headers=headers, method=method), timeout=10) as response:
            payload = response.read(1_048_577)
    except HTTPError as error:
        if method == "POST" and error.code == 409:
            raise PulseConflict("GPU Pulse rejected an overlapping reservation") from None
        raise RuntimeError(f"GPU Pulse {method} returned HTTP {error.code}") from None
    except (OSError, URLError):
        raise RuntimeError(f"GPU Pulse {method} transport failed") from None
    if len(payload) > 1_048_576:
        raise RuntimeError("GPU Pulse response exceeds the supported size")
    try:
        result = json.loads(payload)
    except (ValueError, UnicodeError):
        raise RuntimeError("GPU Pulse returned invalid JSON") from None
    if not isinstance(result, dict):
        raise RuntimeError("GPU Pulse returned an invalid object")
    return result


def private_identity(directory: Path, reservation_id, *, create):
    path = directory / "pulse-private" / reservation_id / "client-id"
    if create and not path.exists():
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="ascii") as stream:
            stream.write(secrets.token_hex(32))
            stream.flush()
            os.fsync(stream.fileno())
    if os.name != "nt" and (path.stat().st_mode & 0o077):
        raise RuntimeError("GPU Pulse client identity file permissions are too broad")
    value = path.read_text(encoding="ascii")
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError("Invalid stored GPU Pulse client identity")
    return value


def _window(config, window, client_id):
    data = request(config, "GET", "/api/reservations", query={"from": window["start_ms"], "to": window["end_ms"],
                                                              "client_id": client_id})
    items = data.get("items")
    if not isinstance(items, list):
        raise RuntimeError("Invalid GPU Pulse reservation list")
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("gpu_id"), str)
                or type(item.get("start_ms")) is not int or type(item.get("end_ms")) is not int
                or item["start_ms"] >= item["end_ms"] or type(item.get("mine")) is not bool
                or not isinstance(item.get("group_id"), str) or not item["group_id"]):
            raise RuntimeError("GPU Pulse sent an invalid reservation row")
    return items


def _matching_slot(item, intent):
    fields = ("gpu_id", "start_ms", "end_ms", "user", "title", "note")
    return item.get("mine") is True and all(item.get(field) == intent.get(field) for field in fields)


def _overlap(item, intent):
    return (item["gpu_id"] == intent["gpu_id"] and item["start_ms"] < intent["end_ms"]
            and intent["start_ms"] < item["end_ms"])


def reserve(directory, record, write_json):
    config = record["pulse_config"]
    rid = record["reservation_id"]
    journal = directory / "pulse" / f"{rid}.json"
    client_id = private_identity(directory, rid, create=True)
    if journal.exists():
        intent = json.loads(journal.read_text(encoding="utf8"))
        existing = _window(config, intent, client_id)
        matches = [row for row in existing if _matching_slot(row, intent)]
        if len(matches) != 1 or intent.get("group_id", matches[0]["group_id"]) != matches[0]["group_id"]:
            raise RuntimeError("Previous GPU Pulse creation requires reconciliation")
        group_id = matches[0]["group_id"]
    else:
        snapshot = request(config, "GET", "/api/snapshot")
        devices = snapshot.get("gpus")
        if (not isinstance(devices, list) or sum(isinstance(gpu, dict) and gpu.get("id") == config["gpu_id"] for gpu in devices) != 1
                or type(snapshot.get("ts")) is not int or snapshot["ts"] <= 0):
            raise RuntimeError("GPU Pulse snapshot does not identify the selected GPU")
        start = snapshot["ts"]
        duration_ms = round(record["estimated_duration_s"] * 1000)
        if not 0 < duration_ms <= MAX_SLOT_MS:
            raise ValueError("GPU Pulse reservation must be at most one hour")
        intent = {"gpu_id": config["gpu_id"], "start_ms": start, "end_ms": start + duration_ms,
                  "user": record["owner_name"], "title": record["task_id"], "note": "SKOPA " + rid}
        if any(_overlap(row, intent) for row in _window(config, intent, client_id)):
            raise PulseConflict("GPU Pulse has an overlapping reservation on the selected GPU")
        write_json(journal, intent)
        try:
            answer = request(config, "POST", "/api/reservations", body={"gpu_ids": [intent["gpu_id"]],
                             "start": intent["start_ms"], "end": intent["end_ms"], "user": intent["user"],
                             "title": intent["title"], "note": intent["note"], "client_id": client_id, "force": False})
        except PulseConflict:
            write_json(journal, {**intent, "rejected": True})
            raise
        group_id = answer.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError("GPU Pulse creation did not return a group ID")
    confirmed = {**intent, "group_id": group_id}
    write_json(journal, confirmed)
    verify_reservation(directory, record, confirmed)
    return confirmed


def verify_reservation(directory, record, receipt):
    config = record["pulse_config"]
    items = _window(config, receipt, private_identity(directory, record["reservation_id"], create=False))
    rows = [row for row in items if row["group_id"] == receipt["group_id"]]
    if len(rows) != 1 or not _matching_slot(rows[0], receipt):
        raise RuntimeError("GPU Pulse did not confirm the matching owned reservation")
    if any(_overlap(row, receipt) for row in items if row["group_id"] != receipt["group_id"]):
        raise PulseConflict("GPU Pulse reports a second overlapping reservation")


def cancel(directory, record, write_json):
    rid = record["reservation_id"]
    journal = directory / "pulse" / f"{rid}.json"
    if not journal.exists():
        return None
    intent = json.loads(journal.read_text(encoding="utf8"))
    if intent.get("rejected"):
        return "conflict_rejected"
    config = record["pulse_config"]
    client_id = private_identity(directory, rid, create=False)
    rows = _window(config, intent, client_id)
    group_id = intent.get("group_id")
    if group_id:
        found = [row for row in rows if row["group_id"] == group_id]
    else:
        found = [row for row in rows if _matching_slot(row, intent)]
    if found:
        group_id = found[0]["group_id"]
        group_rows = [row for row in rows if row["group_id"] == group_id]
        if len(found) != 1 or len(group_rows) != 1 or not _matching_slot(found[0], intent):
            raise RuntimeError("GPU Pulse cancellation requires the exact owned group and slot")
        confirmed = {**intent, "group_id": group_id}
        write_json(journal, confirmed)
        request(config, "DELETE", "/api/reservations", query={"group_id": group_id, "client_id": client_id})
        if any(row["group_id"] == group_id for row in _window(config, confirmed, client_id)):
            raise RuntimeError("GPU Pulse group still exists after cancellation")
        return "canceled"
    if group_id:
        return "already_absent"
    raise RuntimeError("GPU Pulse creation outcome is unresolved")
