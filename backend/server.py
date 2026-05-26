#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_VERSION = "0.1.0"
DEFAULT_MODEL_ID = "OpenVINO/whisper-large-v3-fp16-ov"
DEFAULT_MODEL_DIR = Path.home() / ".local" / "share" / "dictophone" / "models" / "whisper-large-v3-fp16-ov"
DEFAULT_POLISH_MODEL_ID = "OpenVINO/Qwen2.5-7B-Instruct-int4-ov"
DEFAULT_POLISH_MODEL_DIR = Path.home() / ".local" / "share" / "dictophone" / "models" / "qwen2.5-7b-instruct-int4-ov"
DEFAULT_OPENVINO_CACHE_DIR = Path.home() / ".cache" / "dictophone" / "openvino"
LANGUAGE_TOKENS = {
    "en": "<|en|>",
    "de": "<|de|>",
    "ru": "<|ru|>",
}


class Runtime:
    def __init__(self) -> None:
        self.model_id = os.environ.get("DICTOPHONE_MODEL_ID", DEFAULT_MODEL_ID)
        self.model_dir = Path(os.environ.get("DICTOPHONE_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
        self.device = os.environ.get("DICTOPHONE_DEVICE", "CPU")
        self.polish_model_id = os.environ.get("DICTOPHONE_POLISH_MODEL_ID", DEFAULT_POLISH_MODEL_ID)
        self.polish_model_dir = Path(os.environ.get("DICTOPHONE_POLISH_MODEL_DIR", DEFAULT_POLISH_MODEL_DIR)).expanduser()
        self.polish_device = os.environ.get("DICTOPHONE_POLISH_DEVICE", self.device)
        self.cache_dir = Path(os.environ.get("DICTOPHONE_OPENVINO_CACHE_DIR", DEFAULT_OPENVINO_CACHE_DIR)).expanduser()
        self.npu_compiler_override = os.environ.get("DICTOPHONE_NPU_COMPILER_DIR")
        self.pipeline: Any | None = None
        self.polish_pipeline: Any | None = None
        self.openvino_genai: Any | None = None
        self.openvino: Any | None = None
        self.openvino_error: str | None = None
        self.mock_enabled = os.environ.get("DICTOPHONE_MOCK_STT", "0") != "0"
        self._import_openvino()

    def _import_openvino(self) -> None:
        try:
            import openvino as ov  # type: ignore
            import openvino_genai as ov_genai  # type: ignore

            self.openvino = ov
            self.openvino_genai = ov_genai
        except Exception as exc:  # pragma: no cover - environment dependent
            self.openvino_error = f"{type(exc).__name__}: {exc}"

    @property
    def openvino_available(self) -> bool:
        return self.openvino is not None and self.openvino_genai is not None

    def available_devices(self) -> list[str]:
        if not self.openvino_available:
            return []
        try:
            return list(self.openvino.Core().available_devices)
        except Exception:
            return []

    def load_model(self, device: str | None = None, model_dir: str | None = None) -> dict[str, Any]:
        if device:
            self.device = device
        if model_dir:
            self.model_dir = Path(model_dir).expanduser()

        if not self.openvino_available:
            return {
                "ok": False,
                "code": "OPENVINO_UNAVAILABLE",
                "message": self.openvino_error or "OpenVINO GenAI is not installed.",
            }

        if not self.model_dir.exists():
            return {
                "ok": False,
                "code": "MODEL_NOT_FOUND",
                "message": f"Model directory does not exist: {self.model_dir}",
            }

        selected_device = resolve_device(self.device, self.available_devices())
        cache_kwargs = openvino_cache_kwargs(self.cache_dir, "stt", selected_device)
        started = time.perf_counter()
        try:
            self.pipeline = self.openvino_genai.WhisperPipeline(str(self.model_dir), selected_device, **cache_kwargs)
        except Exception as exc:
            self.pipeline = None
            return {
                "ok": False,
                "code": "MODEL_LOAD_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }

        return {
            "ok": True,
            "device": selected_device,
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "load_ms": round((time.perf_counter() - started) * 1000),
        }

    def load_polish_model(self, device: str | None = None, model_dir: str | None = None) -> dict[str, Any]:
        if device:
            self.polish_device = device
        if model_dir:
            self.polish_model_dir = Path(model_dir).expanduser()

        if not self.openvino_available:
            return {
                "ok": False,
                "code": "OPENVINO_UNAVAILABLE",
                "message": self.openvino_error or "OpenVINO GenAI is not installed.",
            }

        if not self.polish_model_dir.exists():
            return {
                "ok": False,
                "code": "POLISH_MODEL_NOT_FOUND",
                "message": f"Polish model directory does not exist: {self.polish_model_dir}",
            }

        selected_device = resolve_device(self.polish_device, self.available_devices())
        cache_kwargs = openvino_cache_kwargs(self.cache_dir, "polish", selected_device)
        started = time.perf_counter()
        try:
            self.polish_pipeline = self.openvino_genai.LLMPipeline(
                str(self.polish_model_dir),
                selected_device,
                MAX_PROMPT_LEN=1536,
                MIN_RESPONSE_LEN=32,
                **cache_kwargs,
            )
        except Exception as exc:
            self.polish_pipeline = None
            return {
                "ok": False,
                "code": "POLISH_MODEL_LOAD_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }

        return {
            "ok": True,
            "device": selected_device,
            "model_id": self.polish_model_id,
            "model_dir": str(self.polish_model_dir),
            "load_ms": round((time.perf_counter() - started) * 1000),
        }

    def download_model(self, model_id: str | None = None, model_dir: str | None = None) -> dict[str, Any]:
        selected_model = model_id or self.model_id
        selected_dir = Path(model_dir).expanduser() if model_dir else self.model_dir
        started = time.perf_counter()
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(
                repo_id=selected_model,
                local_dir=selected_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            return {
                "ok": False,
                "code": "MODEL_DOWNLOAD_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }

        self.model_id = selected_model
        self.model_dir = selected_dir
        return {
            "ok": True,
            "model_id": selected_model,
            "model_dir": str(selected_dir),
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }

    def download_polish_model(self, model_id: str | None = None, model_dir: str | None = None) -> dict[str, Any]:
        selected_model = model_id or self.polish_model_id
        selected_dir = Path(model_dir).expanduser() if model_dir else self.polish_model_dir
        started = time.perf_counter()
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(
                repo_id=selected_model,
                local_dir=selected_dir,
            )
        except Exception as exc:
            return {
                "ok": False,
                "code": "POLISH_MODEL_DOWNLOAD_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }

        self.polish_model_id = selected_model
        self.polish_model_dir = selected_dir
        return {
            "ok": True,
            "model_id": selected_model,
            "model_dir": str(selected_dir),
            "duration_ms": round((time.perf_counter() - started) * 1000),
        }

    def polish_text(self, text: str, language: str, mode: str) -> tuple[str, str, str | None]:
        if mode == "off":
            return text, "off", None
        if mode == "light":
            return deterministic_polish(text, language, mode), "rules", None

        load_result = {"ok": True}
        if self.polish_pipeline is None:
            load_result = self.load_polish_model()
        if not load_result.get("ok") or self.polish_pipeline is None:
            message = load_result.get("message", "OpenVINO polish model is not loaded.")
            return text, "unavailable", str(message)

        started = time.perf_counter()
        prompt = polish_prompt(text, language)
        try:
            result = self.polish_pipeline.generate(
                prompt,
                max_new_tokens=polish_token_budget(text),
                do_sample=False,
                repetition_penalty=1.05,
                stop_strings={"\n\n", "Dictated text:", "Task:"},
            )
        except Exception as exc:
            return text, "error", f"{type(exc).__name__}: {exc}"

        polished = sanitize_polish_output(result_to_text(result), text)
        if not polished:
            return text, "empty", "Polish model returned empty text."
        return polished, "openvino-npu", f"Polished in {round((time.perf_counter() - started) * 1000)} ms"

    def transcribe(self, audio: bytes, content_type: str, language: str, polish: str) -> dict[str, Any]:
        started = time.perf_counter()
        wav_path: Path | None = None
        warning: str | None = None

        try:
            wav_path = convert_to_wav(audio, content_type)
            duration = wav_duration(wav_path)
            raw_speech = read_wav_as_floats(wav_path)
            audio_level = audio_level_metrics(raw_speech)
            selected_device = resolve_device(self.device, self.available_devices())

            if is_probably_silent(audio_level):
                response = with_polish_response(
                    text="",
                    polish=polish,
                    language=language,
                    started=started,
                    duration=duration,
                    device=selected_device,
                    mode="silence",
                )
                response["empty_audio"] = True
                response["warning"] = "No speech detected."
                response["audio_level"] = audio_level
                return response

            if self.openvino_available:
                if self.pipeline is None:
                    load_result = self.load_model()
                    if not load_result.get("ok"):
                        if not self.mock_enabled:
                            return load_result
                        warning = load_result.get("message", "OpenVINO model is not loaded.")
                if self.pipeline is not None:
                    kwargs: dict[str, str] = {}
                    if language in LANGUAGE_TOKENS:
                        kwargs["language"] = LANGUAGE_TOKENS[language]
                    result = self.pipeline.generate(raw_speech, **kwargs)
                    text = normalize_text(result_to_text(result))
                    return with_polish_response(
                        text=text,
                        polish=polish,
                        language=language,
                        started=started,
                        duration=duration,
                        device=selected_device,
                        mode="openvino",
                        audio_level=audio_level,
                    )

            if not self.mock_enabled:
                return {
                    "ok": False,
                    "code": "STT_UNAVAILABLE",
                    "message": warning or self.openvino_error or "OpenVINO transcription is unavailable.",
                }

            text = mock_transcript(duration, language)
            response = with_polish_response(
                text=text,
                polish=polish,
                language=language,
                started=started,
                duration=duration,
                device="mock",
                mode="mock",
                audio_level=audio_level,
            )
            response["warning"] = warning or self.openvino_error or "OpenVINO is not installed; using development mock transcription."
            return response
        finally:
            if wav_path:
                try:
                    shutil.rmtree(wav_path.parent)
                except OSError:
                    pass


runtime = Runtime()


def resolve_device(requested: str, devices: list[str]) -> str:
    if requested and requested != "AUTO":
        return requested
    for candidate in ("CPU", "GPU", "NPU"):
        if candidate in devices:
            return candidate
    return devices[0] if devices else "CPU"


def openvino_cache_kwargs(cache_root: Path, model_kind: str, device: str) -> dict[str, str]:
    try:
        cache_dir = cache_root / f"{model_kind}-{device.lower()}"
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {}

    properties = {"CACHE_DIR": str(cache_dir)}
    if device == "NPU" and model_kind == "polish":
        properties["CACHE_MODE"] = "OPTIMIZE_SPEED"
    return properties


def json_response(handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(payload)


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def convert_to_wav(audio: bytes, content_type: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="dictophone-audio-"))
    suffix = ".wav" if "wav" in content_type else ".webm"
    input_path = temp_dir / f"input{suffix}"
    output_path = temp_dir / "audio.wav"
    input_path.write_bytes(audio)

    if not shutil.which("ffmpeg"):
        if suffix == ".wav":
            return input_path
        raise RuntimeError("ffmpeg is required to convert browser-recorded audio to 16 kHz WAV.")

    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def read_wav_as_floats(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if channels != 1 or sample_width != 2:
        raise RuntimeError("Expected 16 kHz mono 16-bit WAV after conversion.")

    return [
        int.from_bytes(frames[i : i + 2], "little", signed=True) / 32768.0
        for i in range(0, len(frames), 2)
    ]


def audio_level_metrics(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"rms": 0.0, "peak": 0.0}
    peak = max(abs(sample) for sample in samples)
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return {
        "rms": round(mean_square**0.5, 6),
        "peak": round(peak, 6),
    }


def is_probably_silent(audio_level: dict[str, float]) -> bool:
    return audio_level["peak"] < 0.02 and audio_level["rms"] < 0.003


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def result_to_text(result: Any) -> str:
    for attr in ("text", "texts"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str):
                return value
    return str(result)


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([({\[])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)}\]])", r"\1", cleaned)
    return cleaned


FILLERS = {
    "en": ("um", "uh", "erm", "hmm"),
    "de": ("äh", "ähm", "hm"),
    "ru": ("э", "эм", "ээ", "ну"),
}


def deterministic_polish(text: str, language: str, mode: str) -> str:
    words = FILLERS.get(language, FILLERS["en"])
    pattern = r"(?i)\b(" + "|".join(re.escape(word) for word in words) + r")\b[, ]*"
    polished = re.sub(pattern, "", text)
    polished = normalize_text(polished)
    if mode == "polished" and polished:
        polished = polished[0].upper() + polished[1:]
    return polished


def polish_prompt(text: str, language: str) -> str:
    language_rule = {
        "en": "Preserve English.",
        "de": "Preserve German. Do not translate.",
        "ru": "Preserve Russian. Do not translate.",
        "auto": "Infer the language and preserve it. Do not translate.",
    }.get(language, "Infer the language and preserve it. Do not translate.")

    return (
        "Task: clean up dictated text.\n"
        "Rules:\n"
        "- Preserve the exact original meaning.\n"
        f"- {language_rule}\n"
        "- Add normal punctuation and capitalization.\n"
        "- Remove filler words, repeated words, and minor stutters.\n"
        "- Return only the cleaned text.\n\n"
        "Example:\n"
        "Dictated text: das ist ein test und ich glaube es funktioniert\n"
        "Cleaned text: Das ist ein Test und ich glaube, es funktioniert.\n\n"
        "Example:\n"
        "Dictated text: hello this is a test and i think it should have punctuation\n"
        "Cleaned text: Hello, this is a test, and I think it should have punctuation.\n\n"
        "Example:\n"
        "Dictated text: ну вот это тест я думаю он должен работать\n"
        "Cleaned text: Ну вот это тест, я думаю, он должен работать.\n\n"
        f"Dictated text: {text}\n"
        "Cleaned text:"
    )


def polish_token_budget(text: str) -> int:
    word_count = max(1, len(re.findall(r"\S+", text)))
    return min(768, max(128, word_count * 3 + 64))


def sanitize_polish_output(output: str, original: str) -> str:
    cleaned = normalize_text(output)
    cleaned = cleaned.strip("`")
    cleaned = re.sub(r"^(cleaned text|text|output)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1]
    cleaned = normalize_text(cleaned)
    if not cleaned:
        return ""
    if len(cleaned) > max(4096, len(original) * 4):
        return original
    return cleaned


def with_polish_response(
    text: str,
    polish: str,
    language: str,
    started: float,
    duration: float,
    device: str,
    mode: str,
    audio_level: dict[str, float] | None = None,
) -> dict[str, Any]:
    raw_text = normalize_text(text)
    polished = raw_text
    polish_mode = "off"
    polish_engine = "off"
    warning: str | None = None
    if raw_text and polish in {"light", "polished"}:
        polish_mode = polish
        polished, polish_engine, warning = runtime.polish_text(raw_text, language, polish)

    response = {
        "ok": True,
        "text": polished,
        "raw_text": raw_text,
        "polish": polish_mode,
        "polish_engine": polish_engine,
        "language": language,
        "audio_duration_secs": round(duration, 2),
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "device": device,
        "mode": mode,
    }
    if warning is not None:
        response["warning"] = warning
    if audio_level is not None:
        response["audio_level"] = audio_level
    return response


def mock_transcript(duration: float, language: str) -> str:
    labels = {
        "en": "This is a local Dictophone test.",
        "de": "Dies ist ein lokaler Dictophone-Test.",
        "ru": "Это локальный тест Dictophone.",
        "auto": "This is a local Dictophone test.",
    }
    return f"{labels.get(language, labels['auto'])} Recorded {duration:.1f} seconds."


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        json_response(self, 200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/health":
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "version": APP_VERSION,
                    "openvino_available": runtime.openvino_available,
                    "openvino_error": runtime.openvino_error,
                    "mock_enabled": runtime.mock_enabled,
                    "model_loaded": runtime.pipeline is not None,
                    "model_id": runtime.model_id,
                    "model_dir": str(runtime.model_dir),
                    "device": runtime.device,
                    "resolved_device": resolve_device(runtime.device, runtime.available_devices()),
                    "polish_model_loaded": runtime.polish_pipeline is not None,
                    "polish_model_id": runtime.polish_model_id,
                    "polish_model_dir": str(runtime.polish_model_dir),
                    "polish_device": runtime.polish_device,
                    "resolved_polish_device": resolve_device(runtime.polish_device, runtime.available_devices()),
                    "devices": runtime.available_devices(),
                    "npu_compiler_override": runtime.npu_compiler_override,
                },
            )
            return
        if self.path == "/devices":
            devices = runtime.available_devices()
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "devices": [
                        {"device": name, "name": name, "available": name in devices}
                        for name in ("NPU", "GPU", "CPU")
                    ],
                    "raw_devices": devices,
                },
            )
            return
        json_response(self, 404, {"ok": False, "code": "NOT_FOUND"})

    def do_POST(self) -> None:
        try:
            if self.path == "/load-model":
                payload = read_json(self)
                result = runtime.load_model(
                    device=payload.get("device"),
                    model_dir=payload.get("model_dir"),
                )
                json_response(self, 200 if result.get("ok") else 409, result)
                return

            if self.path == "/load-polish-model":
                payload = read_json(self)
                result = runtime.load_polish_model(
                    device=payload.get("device"),
                    model_dir=payload.get("model_dir"),
                )
                json_response(self, 200 if result.get("ok") else 409, result)
                return

            if self.path == "/download-model":
                payload = read_json(self)
                result = runtime.download_model(
                    model_id=payload.get("model_id"),
                    model_dir=payload.get("model_dir"),
                )
                json_response(self, 200 if result.get("ok") else 409, result)
                return

            if self.path == "/download-polish-model":
                payload = read_json(self)
                result = runtime.download_polish_model(
                    model_id=payload.get("model_id"),
                    model_dir=payload.get("model_dir"),
                )
                json_response(self, 200 if result.get("ok") else 409, result)
                return

            if self.path == "/benchmark":
                devices = runtime.available_devices()
                json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "results": [
                            {
                                "device": device,
                                "available": device in devices,
                                "status": "ready" if device in devices else "unavailable",
                            }
                            for device in ("NPU", "GPU", "CPU")
                        ],
                    },
                )
                return

            if self.path.startswith("/transcribe"):
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
                language = params.get("language", "auto")
                polish = params.get("polish", "off")
                length = int(self.headers.get("Content-Length", "0"))
                audio = self.rfile.read(length)
                result = runtime.transcribe(audio, self.headers.get("Content-Type", ""), language, polish)
                json_response(self, 200 if result.get("ok") else 409, result)
                return

            if self.path == "/polish":
                payload = read_json(self)
                text = normalize_text(str(payload.get("text", "")))
                language = str(payload.get("language", "auto"))
                mode = str(payload.get("polish", "light"))
                polished, engine, warning = runtime.polish_text(text, language, mode)
                body = {"ok": True, "text": polished, "raw_text": text, "polish_engine": engine}
                if warning:
                    body["warning"] = warning
                json_response(self, 200, body)
                return

            json_response(self, 404, {"ok": False, "code": "NOT_FOUND"})
        except Exception as exc:
            json_response(self, 500, {"ok": False, "code": "SERVER_ERROR", "message": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DICTOPHONE_PORT", "8765")))
    parser.add_argument("--smoke-wav", help="Transcribe one WAV file and exit")
    parser.add_argument("--device", default=os.environ.get("DICTOPHONE_DEVICE", "AUTO"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--polish", default="off")
    args = parser.parse_args()

    if args.smoke_wav:
        runtime.device = args.device
        with open(args.smoke_wav, "rb") as handle:
            result = runtime.transcribe(handle.read(), "audio/wav", args.language, args.polish)
        print(json.dumps(result, indent=2), flush=True)
        raise SystemExit(0 if result.get("ok") else 1)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dictophone backend listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
