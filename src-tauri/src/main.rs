use serde::Serialize;
use std::{
    env,
    io::Write,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tauri::Manager;

#[derive(Default)]
struct BackendState {
    child: Mutex<Option<Child>>,
    recorder: Mutex<Option<RecorderState>>,
}

struct RecorderState {
    child: Child,
    path: PathBuf,
}

#[derive(Serialize)]
struct InsertResult {
    ok: bool,
    method: String,
    message: Option<String>,
}

#[tauri::command]
fn start_recording(state: tauri::State<'_, BackendState>) -> Result<String, String> {
    let mut guard = state
        .recorder
        .lock()
        .map_err(|_| "Recorder lock is poisoned".to_string())?;
    if guard.is_some() {
        return Err("Recording is already active".to_string());
    }

    let path = std::env::temp_dir().join(format!(
        "dictophone-recording-{}-{}.wav",
        std::process::id(),
        current_millis()
    ));

    let mut command = Command::new("pw-record");
    command.args([
        "--media-category",
        "Capture",
        "--media-role",
        "Communication",
        "--rate",
        "16000",
        "--channels",
        "1",
        "--format",
        "s16",
        "--container",
        "wav",
    ]);
    if let Some(source) = default_audio_source() {
        command.args(["--target", &source]);
    }
    let child = command
        .arg(&path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Failed to start pw-record: {error}"))?;

    *guard = Some(RecorderState {
        child,
        path: path.clone(),
    });
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
fn stop_recording_and_transcribe(
    state: tauri::State<'_, BackendState>,
    language: String,
    polish: String,
) -> Result<serde_json::Value, String> {
    let recording = {
        let mut guard = state
            .recorder
            .lock()
            .map_err(|_| "Recorder lock is poisoned".to_string())?;
        guard
            .take()
            .ok_or_else(|| "No active recording to stop".to_string())?
    };

    stop_recorder(recording.child)?;
    let audio = std::fs::read(&recording.path)
        .map_err(|error| format!("Failed to read recorded audio: {error}"))?;
    let _ = std::fs::copy(&recording.path, std::env::temp_dir().join("dictophone-last.wav"));

    let url = format!(
        "http://127.0.0.1:8765/transcribe?language={}&polish={}",
        language, polish
    );
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "audio/wav")
        .body(audio)
        .send()
        .map_err(|error| format!("Failed to reach transcription backend: {error}"))?;
    let status = response.status();
    let value = response
        .json::<serde_json::Value>()
        .map_err(|error| format!("Backend returned invalid JSON: {error}"))?;

    let _ = std::fs::remove_file(&recording.path);

    if status.is_success() {
        Ok(value)
    } else {
        Err(value
            .get("message")
            .and_then(|message| message.as_str())
            .unwrap_or("Transcription backend failed")
            .to_string())
    }
}

#[tauri::command]
fn insert_text(text: String) -> Result<InsertResult, String> {
    copy_to_clipboard(&text)?;

    match Command::new("wtype")
        .args(["-M", "ctrl", "-k", "v", "-m", "ctrl"])
        .status()
    {
        Ok(status) if status.success() => Ok(InsertResult {
            ok: true,
            method: "clipboard paste".to_string(),
            message: None,
        }),
        Ok(_) => Ok(InsertResult {
            ok: false,
            method: "clipboard".to_string(),
            message: Some("Copied to clipboard. Automatic paste did not complete.".to_string()),
        }),
        Err(_) => Ok(InsertResult {
            ok: false,
            method: "clipboard".to_string(),
            message: Some("Copied to clipboard. Install wtype or eitype for automatic paste.".to_string()),
        }),
    }
}

fn current_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn stop_recorder(mut child: Child) -> Result<(), String> {
    #[cfg(unix)]
    unsafe {
        libc::kill(child.id() as i32, libc::SIGINT);
    }

    #[cfg(not(unix))]
    child
        .kill()
        .map_err(|error| format!("Failed to stop recorder: {error}"))?;

    child
        .wait()
        .map_err(|error| format!("Failed to wait for recorder: {error}"))?;
    Ok(())
}

fn default_audio_source() -> Option<String> {
    let output = Command::new("pactl")
        .args(["get-default-source"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let source = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if source.is_empty() {
        None
    } else {
        Some(source)
    }
}

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    let mut child = Command::new("wl-copy")
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Failed to start wl-copy: {error}"))?;

    let stdin = child
        .stdin
        .as_mut()
        .ok_or_else(|| "Failed to open wl-copy stdin".to_string())?;
    stdin
        .write_all(text.as_bytes())
        .map_err(|error| format!("Failed to write clipboard text: {error}"))?;

    let status = child
        .wait()
        .map_err(|error| format!("Failed to wait for wl-copy: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("wl-copy failed".to_string())
    }
}

fn backend_script_path() -> PathBuf {
    project_root_path()
        .join("backend")
        .join("server.py")
}

fn project_root_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri has a parent directory")
        .to_path_buf()
}

fn backend_python_path() -> PathBuf {
    let project_root = project_root_path();
    let venv_python = project_root.join(".venv").join("bin").join("python");
    if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("python3")
    }
}

fn npu_compiler_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("DICTOPHONE_NPU_COMPILER_DIR") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        project_root_path()
            .join("vendor")
            .join("intel-npu-compiler"),
    );
    if let Some(home) = env::var_os("HOME") {
        candidates.push(
            PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("dictophone")
                .join("npu-compiler"),
        );
    }

    candidates
        .into_iter()
        .find(|path| path.join("libnpu_driver_compiler.so").exists())
}

fn fast_npu_model_dir() -> Option<PathBuf> {
    let home = env::var_os("HOME")?;
    let model_dir = PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("dictophone")
        .join("models")
        .join("whisper-tiny-fp16-ov");
    if model_dir.exists() {
        Some(model_dir)
    } else {
        None
    }
}

fn npu_polish_model_dir() -> Option<PathBuf> {
    let home = env::var_os("HOME")?;
    let model_dir = PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("dictophone")
        .join("models")
        .join("qwen2.5-1.5b-instruct-int4-ov");
    if model_dir.exists() {
        Some(model_dir)
    } else {
        None
    }
}

fn configure_backend_environment(command: &mut Command) {
    if env::var_os("DICTOPHONE_MOCK_STT").is_none() {
        command.env("DICTOPHONE_MOCK_STT", "0");
    }

    if let Some(compiler_dir) = npu_compiler_dir() {
        let mut library_path = compiler_dir.to_string_lossy().to_string();
        if let Some(existing) = env::var_os("LD_LIBRARY_PATH") {
            if !existing.is_empty() {
                library_path.push(':');
                library_path.push_str(&existing.to_string_lossy());
            }
        }
        command.env("LD_LIBRARY_PATH", library_path);
        command.env("DICTOPHONE_NPU_COMPILER_DIR", &compiler_dir);

        if env::var_os("DICTOPHONE_DEVICE").is_none() {
            command.env("DICTOPHONE_DEVICE", "NPU");
        }
        if env::var_os("DICTOPHONE_MODEL_DIR").is_none() {
            if let Some(model_dir) = fast_npu_model_dir() {
                command.env("DICTOPHONE_MODEL_ID", "OpenVINO/whisper-tiny-fp16-ov");
                command.env("DICTOPHONE_MODEL_DIR", model_dir);
            }
        }
        if env::var_os("DICTOPHONE_POLISH_DEVICE").is_none() {
            command.env("DICTOPHONE_POLISH_DEVICE", "NPU");
        }
        if env::var_os("DICTOPHONE_POLISH_MODEL_DIR").is_none() {
            if let Some(model_dir) = npu_polish_model_dir() {
                command.env("DICTOPHONE_POLISH_MODEL_ID", "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov");
                command.env("DICTOPHONE_POLISH_MODEL_DIR", model_dir);
            }
        }
    }
}

fn start_backend(state: &BackendState) {
    if env::var("DICTOPHONE_NO_BACKEND_AUTOSTART").is_ok() {
        return;
    }

    let script = backend_script_path();
    let mut command = Command::new(backend_python_path());
    command
        .arg("-u")
        .arg(script)
        .arg("--port")
        .arg("8765")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    configure_backend_environment(&mut command);

    let child = command.spawn();

    if let Ok(child) = child {
        if let Ok(mut guard) = state.child.lock() {
            *guard = Some(child);
        }
    }
}

fn stop_backend(state: &BackendState) {
    if let Ok(mut guard) = state.child.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn main() {
    tauri::Builder::default()
        .manage(BackendState::default())
        .setup(|app| {
            let state = app.state::<BackendState>();
            start_backend(&state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            insert_text,
            start_recording,
            stop_recording_and_transcribe
        ])
        .build(tauri::generate_context!())
        .expect("failed to build tauri app")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app_handle.state::<BackendState>();
                stop_backend(&state);
            }
        });
}
