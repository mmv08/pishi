#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
VENDORED_NPU_COMPILER_DIR="$ROOT/vendor/intel-npu-compiler"
LOCAL_NPU_COMPILER_DIR="$HOME/.local/share/dictophone/npu-compiler"
NPU_COMPILER_DIR="${DICTOPHONE_NPU_COMPILER_DIR:-}"
STT_MODEL_DIR="${DICTOPHONE_MODEL_DIR:-$HOME/.local/share/dictophone/models/whisper-large-v3-fp16-ov}"
POLISH_MODEL_DIR="${DICTOPHONE_POLISH_MODEL_DIR:-$HOME/.local/share/dictophone/models/qwen2.5-7b-instruct-int4-ov}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python venv: $PYTHON" >&2
  echo "Run: python3 -m venv .venv && .venv/bin/python -m pip install -r backend/requirements.txt" >&2
  exit 1
fi

if [[ -z "$NPU_COMPILER_DIR" && -f "$VENDORED_NPU_COMPILER_DIR/libnpu_driver_compiler.so" ]]; then
  NPU_COMPILER_DIR="$VENDORED_NPU_COMPILER_DIR"
fi
if [[ -z "$NPU_COMPILER_DIR" && -f "$LOCAL_NPU_COMPILER_DIR/libnpu_driver_compiler.so" ]]; then
  NPU_COMPILER_DIR="$LOCAL_NPU_COMPILER_DIR"
fi

if [[ -n "$NPU_COMPILER_DIR" && -f "$NPU_COMPILER_DIR/libnpu_driver_compiler.so" ]]; then
  export LD_LIBRARY_PATH="$NPU_COMPILER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export DICTOPHONE_NPU_COMPILER_DIR="$NPU_COMPILER_DIR"
fi

export DICTOPHONE_MOCK_STT="${DICTOPHONE_MOCK_STT:-0}"
export DICTOPHONE_DEVICE="${DICTOPHONE_DEVICE:-NPU}"
export DICTOPHONE_POLISH_DEVICE="${DICTOPHONE_POLISH_DEVICE:-NPU}"

if [[ -d "$STT_MODEL_DIR" ]]; then
  export DICTOPHONE_MODEL_ID="${DICTOPHONE_MODEL_ID:-OpenVINO/whisper-large-v3-fp16-ov}"
  export DICTOPHONE_MODEL_DIR="$STT_MODEL_DIR"
fi

if [[ -d "$POLISH_MODEL_DIR" ]]; then
  export DICTOPHONE_POLISH_MODEL_ID="${DICTOPHONE_POLISH_MODEL_ID:-OpenVINO/Qwen2.5-7B-Instruct-int4-ov}"
  export DICTOPHONE_POLISH_MODEL_DIR="$POLISH_MODEL_DIR"
fi

exec "$PYTHON" -u "$ROOT/backend/server.py" --port "${DICTOPHONE_PORT:-8765}"
