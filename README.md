# Pishi

Local dictation for Fedora/GNOME with a Tauri + React UI and an OpenVINO backend tuned for Intel Core Ultra laptops.

Pishi records from PipeWire, transcribes speech with Whisper on the Intel NPU, and can run an OpenVINO LLM on the NPU to polish punctuation and cleanup.

## Stack

- Tauri 2 desktop shell
- React + Vite UI
- Python backend using `openvino-genai`
- PipeWire recording via `pw-record`
- Wayland paste via `wl-copy` + `wtype`
- STT model: `OpenVINO/whisper-large-v3-fp16-ov`
- Polish model: `OpenVINO/Qwen2.5-7B-Instruct-int4-ov`

## Requirements

- Fedora/GNOME on Wayland
- Node.js + npm
- Rust toolchain
- Python 3.14 venv
- `pw-record`, `wl-copy`, `wtype`, `ffmpeg`
- OpenVINO Python packages from `backend/requirements.txt`
- Intel NPU runtime visible to OpenVINO

Install Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
```

Install Node dependencies:

```bash
npm install
```

## Models

Download the default local models:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path

base = Path.home() / ".local/share/dictophone/models"
snapshot_download("OpenVINO/whisper-large-v3-fp16-ov", local_dir=base / "whisper-large-v3-fp16-ov")
snapshot_download("OpenVINO/Qwen2.5-7B-Instruct-int4-ov", local_dir=base / "qwen2.5-7b-instruct-int4-ov")
PY
```

These defaults optimize for correctness over minimum latency. The first NPU run can spend several minutes compiling model graphs; subsequent runs use the OpenVINO cache at `~/.cache/dictophone/openvino`. On the target laptop, the two default models use about 7 GB and the warmed NPU cache uses about 9 GB.

## NPU Compiler Workaround

On this Fedora setup, the Fedora NPU compiler package did not compile OpenVINO NPU graphs correctly. The app launcher supports a local compiler override:

```text
vendor/intel-npu-compiler/libnpu_driver_compiler.so
```

That binary is tracked with Git LFS. The launcher prefers the vendored copy. It also supports a local override:

```text
~/.local/share/dictophone/npu-compiler/libnpu_driver_compiler.so
```

Alternatively set:

```bash
export DICTOPHONE_NPU_COMPILER_DIR=/path/to/compiler-dir
```

The directory must contain `libnpu_driver_compiler.so`.

## Running

For normal development, let Tauri start the backend:

```bash
npm run tauri:dev
```

If running the backend manually, use the wrapper so it uses the venv and NPU defaults:

```bash
npm run backend
DICTOPHONE_NO_BACKEND_AUTOSTART=1 npm run tauri:dev
```

Avoid running `python3 backend/server.py` directly unless you have installed OpenVINO into system Python.

## Build

```bash
npm run build
npx tauri build --no-bundle
```

The no-bundle build writes the binary under `src-tauri/target/release/`.
