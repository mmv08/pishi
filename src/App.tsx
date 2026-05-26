import { invoke } from "@tauri-apps/api/core";
import {
  Activity,
  CheckCircle2,
  Clipboard,
  Copy,
  Cpu,
  Languages,
  Mic,
  Pause,
  Play,
  RefreshCw,
  Sparkles,
  Square,
  Wand2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = "http://127.0.0.1:8765";

type Health = {
  ok: boolean;
  version: string;
  openvino_available: boolean;
  openvino_error?: string;
  mock_enabled: boolean;
  model_loaded: boolean;
  model_id: string;
  model_dir: string;
  device: string;
  resolved_device?: string;
  polish_model_loaded: boolean;
  polish_model_id: string;
  polish_model_dir: string;
  polish_device: string;
  resolved_polish_device?: string;
  devices: string[];
  npu_compiler_override?: string | null;
};

type DeviceResult = {
  device: string;
  available: boolean;
  status: string;
};

type TranscriptResult = {
  ok: boolean;
  text: string;
  raw_text: string;
  polish: string;
  language: string;
  audio_duration_secs: number;
  duration_ms: number;
  device: string;
  mode: string;
  polish_engine?: string;
  warning?: string;
  empty_audio?: boolean;
  code?: string;
  message?: string;
};

type HistoryItem = TranscriptResult & {
  id: string;
  createdAt: string;
};

type InsertResult = {
  ok: boolean;
  method: string;
  message?: string;
};

type RecorderState = "idle" | "recording" | "transcribing" | "error";
type PolishMode = "off" | "light" | "polished";
type LanguageMode = "auto" | "en" | "de" | "ru";

const languageLabels: Record<LanguageMode, string> = {
  auto: "Auto",
  en: "English",
  de: "German",
  ru: "Russian",
};

const polishLabels: Record<PolishMode, string> = {
  off: "Raw",
  light: "Light",
  polished: "Polished",
};

const STORAGE_KEY = "dictophone:v1";

type StoredState = {
  language?: LanguageMode;
  polish?: PolishMode;
  history?: HistoryItem[];
};

function loadStoredState(): StoredState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StoredState) : {};
  } catch {
    return {};
  }
}

function App() {
  const stored = useMemo(loadStoredState, []);
  const [health, setHealth] = useState<Health | null>(null);
  const [devices, setDevices] = useState<DeviceResult[]>([]);
  const [status, setStatus] = useState<RecorderState>("idle");
  const [message, setMessage] = useState("Ready");
  const [language, setLanguage] = useState<LanguageMode>(stored.language ?? "auto");
  const [polish, setPolish] = useState<PolishMode>(stored.polish ?? "light");
  const [lastTranscript, setLastTranscript] = useState<TranscriptResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>(stored.history ?? []);
  const [recordingMs, setRecordingMs] = useState(0);
  const [insertMethod, setInsertMethod] = useState<string>("clipboard paste");
  const [modelBusy, setModelBusy] = useState(false);
  const [polishBusy, setPolishBusy] = useState(false);
  const startedAt = useRef<number>(0);
  const ticker = useRef<number | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const [healthResponse, devicesResponse] = await Promise.all([
        fetch(`${API}/health`),
        fetch(`${API}/devices`),
      ]);
      setHealth(await healthResponse.json());
      setDevices((await devicesResponse.json()).devices ?? []);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Backend is unavailable");
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 5000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  useEffect(() => {
    return () => {
      if (ticker.current) window.clearInterval(ticker.current);
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        language,
        polish,
        history: history.slice(0, 20),
      }),
    );
  }, [language, polish, history]);

  useEffect(() => {
    if (polish !== "polished" || polishBusy || health?.polish_model_loaded || !health?.openvino_available) {
      return;
    }
    void loadPolishModel();
  }, [polish, polishBusy, health?.polish_model_loaded, health?.openvino_available]);

  const activeDevice = useMemo(() => {
    if (health?.device && health.device !== "AUTO") {
      return health.device;
    }
    if (health?.resolved_device) {
      return health.resolved_device;
    }
    const firstAvailable = devices.find((device) => device.available);
    return firstAvailable?.device ?? "CPU";
  }, [devices, health]);

  const backendLabel = health?.mock_enabled
    ? "Dev mock"
    : health?.npu_compiler_override
      ? "NPU tuned"
      : "Local";

  async function startRecording() {
    try {
      startedAt.current = Date.now();
      setRecordingMs(0);
      ticker.current = window.setInterval(() => {
        setRecordingMs(Date.now() - startedAt.current);
      }, 100);
      await invoke<string>("start_recording");
      setStatus("recording");
      setMessage("Recording");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Microphone is unavailable");
    }
  }

  function stopRecording() {
    if (status !== "recording") return;
    setStatus("transcribing");
    setMessage("Transcribing");
    if (ticker.current) window.clearInterval(ticker.current);
    void submitRecording();
  }

  async function submitRecording() {
    try {
      const result = await invoke<TranscriptResult>("stop_recording_and_transcribe", {
        language,
        polish,
      });
      if (!result.ok) {
        throw new Error(result.message ?? result.code ?? "Transcription failed");
      }
      if (result.empty_audio || !result.text.trim()) {
        setLastTranscript(null);
        setStatus("idle");
        setMessage(result.warning ?? "No speech detected");
        return;
      }
      setLastTranscript(result);
      setHistory((items) => [
        {
          ...result,
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
        },
        ...items,
      ].slice(0, 20));
      setStatus("idle");
      setMessage(result.warning ?? "Transcribed");
      await insertText(result.text);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Transcription failed");
    }
  }

  async function insertText(text: string) {
    try {
      const result = await invoke<InsertResult>("insert_text", { text });
      setInsertMethod(result.method);
      if (!result.ok && result.message) {
        setMessage(result.message);
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Text copied failed");
    }
  }

  async function runBenchmark() {
    setMessage("Checking devices");
    try {
      const response = await fetch(`${API}/benchmark`, { method: "POST" });
      const body = await response.json();
      setDevices(body.results ?? []);
      setMessage("Device check complete");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Device check failed");
    }
  }

  async function downloadModel() {
    setModelBusy(true);
    setMessage("Downloading model");
    try {
      const response = await fetch(`${API}/download-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const body = await response.json();
      if (!body.ok) throw new Error(body.message ?? "Model download failed");
      setMessage("Model downloaded");
      await refreshHealth();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Model download failed");
    } finally {
      setModelBusy(false);
    }
  }

  async function loadModel(device = activeDevice) {
    setModelBusy(true);
    setMessage(`Loading model on ${device}`);
    try {
      const response = await fetch(`${API}/load-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device }),
      });
      const body = await response.json();
      if (!body.ok) throw new Error(body.message ?? "Model load failed");
      setMessage(`Model loaded on ${body.device}`);
      await refreshHealth();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Model load failed");
    } finally {
      setModelBusy(false);
    }
  }

  async function loadPolishModel(device = health?.resolved_polish_device ?? health?.polish_device ?? activeDevice) {
    setPolishBusy(true);
    setMessage(`Loading polish on ${device}`);
    try {
      const response = await fetch(`${API}/load-polish-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device }),
      });
      const body = await response.json();
      if (!body.ok) throw new Error(body.message ?? "Polish model load failed");
      setMessage(`Polish loaded on ${body.device}`);
      await refreshHealth();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Polish model load failed");
    } finally {
      setPolishBusy(false);
    }
  }

  const elapsed = (recordingMs / 1000).toFixed(1);
  const isRecording = status === "recording";
  const isBusy = status === "transcribing";
  const polishStatus = health?.polish_model_loaded
    ? "Loaded"
    : polishBusy
      ? "Loading"
      : polish === "polished"
        ? "Needed"
        : "Idle";

  return (
    <main className="app-shell">
      <aside className="side-panel">
        <div className="brand">
          <div className="brand-mark">
            <Mic size={22} />
          </div>
          <div>
            <h1>Dictophone</h1>
            <p>{health?.openvino_available ? "OpenVINO ready" : "OpenVINO pending"}</p>
          </div>
        </div>

        <section className="status-stack">
          <StatusRow icon={<Cpu size={17} />} label="Device" value={activeDevice} />
          <StatusRow icon={<Activity size={17} />} label="Backend" value={backendLabel} />
          <StatusRow icon={<CheckCircle2 size={17} />} label="Model" value={health?.model_loaded ? "Loaded" : "Not loaded"} />
          <StatusRow icon={<Sparkles size={17} />} label="Polish" value={polishStatus} />
          <StatusRow icon={<Clipboard size={17} />} label="Output" value={insertMethod} />
        </section>

        <section className="settings-block">
          <label>
            <span><Languages size={16} /> Language</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value as LanguageMode)}>
              {Object.entries(languageLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>

          <label>
            <span><Sparkles size={16} /> Polish</span>
            <select value={polish} onChange={(event) => setPolish(event.target.value as PolishMode)}>
              {Object.entries(polishLabels).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
        </section>

        <div className="side-actions">
          <button className="secondary-button" onClick={refreshHealth}>
            <RefreshCw size={16} /> Refresh
          </button>
          <button className="secondary-button" onClick={runBenchmark}>
            <Cpu size={16} /> Devices
          </button>
        </div>

        <div className="model-actions">
          <button className="secondary-button" onClick={downloadModel} disabled={modelBusy}>
            <RefreshCw size={16} /> Download
          </button>
          <button className="secondary-button" onClick={() => loadModel()} disabled={modelBusy}>
            <CheckCircle2 size={16} /> Load
          </button>
        </div>

        {polish === "polished" && (
          <button className="secondary-button wide-button" onClick={() => loadPolishModel()} disabled={polishBusy}>
            <Sparkles size={16} /> Load Polish
          </button>
        )}
      </aside>

      <section className="workspace">
        <header className="top-bar">
          <div>
            <p className="eyebrow">Local dictation</p>
            <h2>{message}</h2>
          </div>
          <div className={`state-pill ${status}`}>
            {status === "idle" && <CheckCircle2 size={16} />}
            {status === "recording" && <Pause size={16} />}
            {status === "transcribing" && <Wand2 size={16} />}
            {status === "error" && <Square size={16} />}
            {status}
          </div>
        </header>

        <section className="record-surface">
          <div className={`waveform ${status}`}>
            {Array.from({ length: 38 }).map((_, index) => (
              <span key={index} style={{ "--i": index } as React.CSSProperties} />
            ))}
          </div>
          <button
            className={`record-button ${isRecording ? "recording" : ""}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isBusy}
            aria-label={isRecording ? "Stop recording" : "Start recording"}
          >
            {isRecording ? <Square size={34} fill="currentColor" /> : <Play size={38} fill="currentColor" />}
          </button>
          <div className="record-meta">
            <strong>{isRecording ? `${elapsed}s` : lastTranscript ? `${lastTranscript.duration_ms} ms` : "Ready"}</strong>
            <span>{isRecording ? "Listening" : isBusy ? "Processing" : `${languageLabels[language]} · ${polishLabels[polish]}`}</span>
          </div>
        </section>

        <section className="transcript-area">
          <div className="section-heading">
            <h3>Transcript</h3>
            {lastTranscript && (
              <button className="icon-button" onClick={() => insertText(lastTranscript.text)} aria-label="Insert transcript">
                <Copy size={17} />
              </button>
            )}
          </div>
          <div className="transcript-box">
            {lastTranscript?.text || "Your next dictation will appear here."}
          </div>
          {lastTranscript?.raw_text && lastTranscript.raw_text !== lastTranscript.text && (
            <details className="raw-details">
              <summary>Raw transcript</summary>
              <p>{lastTranscript.raw_text}</p>
            </details>
          )}
        </section>

        <section className="history-area">
          <div className="section-heading">
            <h3>History</h3>
            <span>{history.length}</span>
          </div>
          <div className="history-list">
            {history.length === 0 ? (
              <p className="empty">No dictations yet.</p>
            ) : (
              history.map((item) => (
                <button key={item.id} className="history-item" onClick={() => insertText(item.text)}>
                  <span>{item.text}</span>
                  <small>{new Date(item.createdAt).toLocaleTimeString()} · {item.device}</small>
                </button>
              ))
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

function StatusRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="status-row">
      <div>{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
