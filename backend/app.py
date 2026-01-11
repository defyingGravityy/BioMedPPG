"""Flask backend for streaming MAX30102 readings and hemoglobin predictions."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO

try:
    import serial  # type: ignore
    import serial.serialutil  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    serial = None  # type: ignore

# Ensure the parent folder (project root) is on sys.path so we can import predictors
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from csv_multi_predictor_backend import (  # noqa: E402
    format_summary,
    process_readings_dataframe,
)

# Load environment variables from optional .env files
for env_candidate in (BASE_DIR / ".env", BASE_DIR / "backend" / ".env"):
    if env_candidate.exists():
        load_dotenv(env_candidate, override=False)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Runtime configuration values
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUDRATE = _env_int("SERIAL_BAUDRATE", 9600)
SERIAL_TIMEOUT = _env_float("SERIAL_TIMEOUT", 1.0)
SERIAL_COLLECTION_SECONDS = _env_int("SERIAL_COLLECTION_SECONDS", 60)
SERIAL_MIN_VALUE = _env_int("SERIAL_MIN_VALUE", 50000)
SERIAL_MAX_VALUE = _env_int("SERIAL_MAX_VALUE", 200000)
SERIAL_SIMULATION_ENABLED = _env_bool("SERIAL_SIMULATION_ENABLED", False)
SERIAL_SIMULATION_INTERVAL = _env_float("SERIAL_SIMULATION_INTERVAL", 0.5)
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*")
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = _env_int("BACKEND_PORT", 8000)

_model_path_env = os.getenv("MODEL_PATH", "hemoglobin_svr_model.pkl")
MODEL_PATH = Path(_model_path_env)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (BASE_DIR / MODEL_PATH).resolve()

_storage_root_env = os.getenv("STORAGE_ROOT")
if _storage_root_env:
    STORAGE_ROOT = Path(_storage_root_env)
    if not STORAGE_ROOT.is_absolute():
        STORAGE_ROOT = (BASE_DIR / STORAGE_ROOT).resolve()
else:
    STORAGE_ROOT = (BASE_DIR / "backend" / "data").resolve()
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)

if CORS_ALLOWED_ORIGINS.strip() in {"", "*"}:
    cors_origins: str | list[str] = "*"
else:
    cors_origins = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]
    if not cors_origins:
        cors_origins = "*"

CORS(
    app,
    resources={
        r"/api/*": {"origins": cors_origins},
        r"/health": {"origins": cors_origins},
    },
    supports_credentials=False,
)

socketio = SocketIO(app, async_mode="threading", cors_allowed_origins=cors_origins)


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def _sanitize_folder(user_id: str, name: str) -> Tuple[str, Path]:
    folder_token = f"{_sanitize_token(user_id)}_{_sanitize_token(name)}"
    folder_path = STORAGE_ROOT / folder_token
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_token, folder_path


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _storage_relative(path: Path) -> str:
    return str(path.resolve().relative_to(STORAGE_ROOT))


def _resolve_storage_path(subpath: str) -> Optional[Path]:
    candidate = (STORAGE_ROOT / subpath).resolve()
    try:
        candidate.relative_to(STORAGE_ROOT)
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


@dataclass
class SessionConfig:
    session_id: str
    user_id: str
    name: str
    age: int
    gender: str
    duration_seconds: int
    port: str
    baudrate: int
    timeout: float
    value_min: int
    value_max: int
    simulate: bool
    simulation_interval: float


class SerialSession(threading.Thread):
    def __init__(self, config: SessionConfig):
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = threading.Event()
        self.readings: List[Tuple[int, int]] = []

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:  # noqa: D401 - threading.Thread entrypoint
        LOGGER.info("Session %s started", self.config.session_id)
        socketio.emit(
            "session_status",
            {
                "sessionId": self.config.session_id,
                "status": "started",
                "user": {
                    "id": self.config.user_id,
                    "name": self.config.name,
                    "age": self.config.age,
                    "gender": self.config.gender,
                },
                "simulate": self.config.simulate,
            },
        )

        try:
            if self.config.simulate:
                self._simulate_stream()
            else:
                self._stream_from_serial()
        except Exception as exc:  # pragma: no cover - runtime protection
            LOGGER.exception("Session %s failed", self.config.session_id)
            socketio.emit(
                "session_error",
                {
                    "sessionId": self.config.session_id,
                    "error": str(exc),
                },
            )
        finally:
            unregister_session(self.config.session_id)

    def _simulate_stream(self) -> None:
        end_time = time.time() + self.config.duration_seconds
        base_red = 115000
        base_ir = 105000
        while not self.stop_event.is_set() and time.time() < end_time:
            red = int(random.gauss(base_red, 1500))
            ir = int(random.gauss(base_ir, 1200))
            if not self._process_reading(red, ir):
                continue
            time.sleep(max(self.config.simulation_interval, 0.05))
        self._finalize_session()

    def _stream_from_serial(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Set SERIAL_SIMULATION_ENABLED=true for dry runs.")

        try:
            ser = serial.Serial(  # type: ignore[call-arg]
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
        except serial.serialutil.SerialException as exc:  # type: ignore[attr-defined]
            raise RuntimeError(f"Failed to open serial port {self.config.port}: {exc}") from exc

        with ser:
            end_time = time.time() + self.config.duration_seconds
            while not self.stop_event.is_set() and time.time() < end_time:
                try:
                    raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue

                if not raw_line:
                    continue

                try:
                    red_str, ir_str, *_ = [part.strip() for part in raw_line.split(",")]
                    red = int(red_str)
                    ir = int(ir_str)
                except (ValueError, IndexError):
                    LOGGER.debug("Ignoring unparsable line: %s", raw_line)
                    continue

                self._process_reading(red, ir)

        self._finalize_session()

    def _process_reading(self, red: int, ir: int) -> bool:
        if not (self.config.value_min <= red <= self.config.value_max):
            return False
        if not (self.config.value_min <= ir <= self.config.value_max):
            return False

        index = len(self.readings)
        self.readings.append((red, ir))

        socketio.emit(
            "serial_reading",
            {
                "sessionId": self.config.session_id,
                "index": index,
                "red": red,
                "ir": ir,
                "total": len(self.readings),
                "timestamp": time.time(),
            },
        )
        return True

    def _finalize_session(self) -> None:
        if not self.readings:
            socketio.emit(
                "session_error",
                {
                    "sessionId": self.config.session_id,
                    "error": "No valid readings collected.",
                },
            )
            LOGGER.warning("Session %s finished without readings", self.config.session_id)
            return

        df = pd.DataFrame(self.readings, columns=["Red", "IR"])
        results = process_readings_dataframe(
            df,
            self.config.gender,
            self.config.age,
            model_path=str(MODEL_PATH),
            verbose=False,
        )

        if not results:
            socketio.emit(
                "session_error",
                {
                    "sessionId": self.config.session_id,
                    "error": "Prediction model returned no results.",
                },
            )
            LOGGER.warning("Session %s produced no predictions", self.config.session_id)
            return

        summary_text = format_summary(results, self.config.gender, self.config.age)

        folder_token, folder_path = _sanitize_folder(self.config.user_id, self.config.name)
        readings_path = folder_path / f"{folder_token}_reading.csv"
        results_path = folder_path / f"{folder_token}_result.txt"

        df.to_csv(readings_path, index=False)
        readings_token = _storage_relative(readings_path)
        results_token = _storage_relative(results_path)
        results_payload = {
            "stats": results,
            "user": {
                "id": self.config.user_id,
                "name": self.config.name,
                "age": self.config.age,
                "gender": self.config.gender,
            },
            "sessionId": self.config.session_id,
            "downloadTokens": {
                "readings": readings_token,
                "results": results_token,
            },
        }
        results_path.write_text(
            f"{summary_text}\n\n---\nMachine-readable payload:\n"
            f"{json.dumps(results_payload, indent=2)}\n",
            encoding="utf-8",
        )

        socketio.emit(
            "session_complete",
            {
                "sessionId": self.config.session_id,
                "summary": summary_text,
                "stats": results,
                "readingsFile": _relative_path(readings_path),
                "resultsFile": _relative_path(results_path),
                "downloadTokens": {
                    "readings": readings_token,
                    "results": results_token,
                },
            },
        )
        LOGGER.info("Session %s complete (%d readings)", self.config.session_id, len(self.readings))


active_sessions: Dict[str, SerialSession] = {}
session_lock = threading.Lock()


def register_session(session: SerialSession) -> None:
    with session_lock:
        active_sessions[session.config.session_id] = session


def unregister_session(session_id: str) -> None:
    with session_lock:
        active_sessions.pop(session_id, None)


def _get_session(session_id: str) -> Optional[SerialSession]:
    with session_lock:
        return active_sessions.get(session_id)


def _normalize_gender(value: str) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip().lower()
    if candidate in {"male", "m"}:
        return "Male"
    if candidate in {"female", "f"}:
        return "Female"
    return None


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "modelPath": str(MODEL_PATH)}), 200


@app.route("/api/start-session", methods=["POST"])
def start_session():
    payload = request.get_json(silent=True) or {}

    user_id = str(payload.get("userId", "")).strip()
    name = str(payload.get("name", "")).strip()
    age = payload.get("age")
    gender = payload.get("gender")
    duration = payload.get("durationSeconds", SERIAL_COLLECTION_SECONDS)
    simulate_override = payload.get("simulate")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400
    if not name:
        return jsonify({"error": "name is required"}), 400
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return jsonify({"error": "age must be an integer"}), 400
    normalized_gender = _normalize_gender(gender or "")
    if not normalized_gender:
        return jsonify({"error": "gender must be 'Male' or 'Female'"}), 400
    try:
        duration_int = int(duration)
    except (TypeError, ValueError):
        duration_int = SERIAL_COLLECTION_SECONDS
    duration_int = max(duration_int, 1)

    if not MODEL_PATH.exists():
        return (
            jsonify({"error": f"Model file not found at {MODEL_PATH}"}),
            500,
        )

    simulate = SERIAL_SIMULATION_ENABLED
    if simulate_override is not None:
        simulate = bool(simulate_override)

    session_id = str(uuid.uuid4())
    config = SessionConfig(
        session_id=session_id,
        user_id=user_id,
        name=name,
        age=age_int,
        gender=normalized_gender,
        duration_seconds=duration_int,
        port=SERIAL_PORT,
        baudrate=SERIAL_BAUDRATE,
        timeout=SERIAL_TIMEOUT,
        value_min=SERIAL_MIN_VALUE,
        value_max=SERIAL_MAX_VALUE,
        simulate=simulate,
        simulation_interval=SERIAL_SIMULATION_INTERVAL,
    )

    session = SerialSession(config)
    register_session(session)
    session.start()

    LOGGER.info(
        "Session %s queued (simulate=%s, duration=%ss)",
        session_id,
        simulate,
        duration_int,
    )

    return (
        jsonify(
            {
                "sessionId": session_id,
                "status": "started",
                "simulate": simulate,
                "config": {
                    "port": SERIAL_PORT,
                    "baudrate": SERIAL_BAUDRATE,
                    "timeout": SERIAL_TIMEOUT,
                    "durationSeconds": duration_int,
                },
            }
        ),
        202,
    )


@app.route("/api/session/<session_id>/stop", methods=["POST"])
def stop_session(session_id: str):
    session = _get_session(session_id)
    if not session:
        return jsonify({"error": "session not found"}), 404

    session.stop()
    LOGGER.info("Stop requested for session %s", session_id)
    return jsonify({"sessionId": session_id, "status": "stopping"}), 202


@app.route("/api/session", methods=["GET"])
def list_sessions():
    with session_lock:
        items = [
            {
                "sessionId": sid,
                "userId": sess.config.user_id,
                "name": sess.config.name,
                "simulate": sess.config.simulate,
            }
            for sid, sess in active_sessions.items()
        ]
    return jsonify({"active": items}), 200


@app.route("/api/download/<path:subpath>", methods=["GET"])
def download_generated_file(subpath: str):
    target = _resolve_storage_path(subpath)
    if not target:
        abort(404)
    LOGGER.info("Download requested for %s", target)
    return send_file(target, as_attachment=True)


if __name__ == "__main__":
    LOGGER.info(
        "Starting backend on %s:%s (simulate default=%s)",
        BACKEND_HOST,
        BACKEND_PORT,
        SERIAL_SIMULATION_ENABLED,
    )
    socketio.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
