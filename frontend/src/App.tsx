import { useState, useRef, useCallback, useEffect } from "react";
import "@google/model-viewer";

const API_URL = "/api/predict";

/** Parse a .npy file (float32/float64, 2D) and return a grayscale data URL. */
async function npyToDataURL(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const view = new DataView(buf);

  // Parse header: \x93NUMPY + version (2 bytes) + header_len (2 or 4 bytes)
  const major = view.getUint8(6);
  let headerLen: number;
  let headerOffset: number;
  if (major >= 2) {
    headerLen = view.getUint32(8, true);
    headerOffset = 12;
  } else {
    headerLen = view.getUint16(8, true);
    headerOffset = 10;
  }
  const headerStr = new TextDecoder().decode(
    buf.slice(headerOffset, headerOffset + headerLen)
  );
  const dataOffset = headerOffset + headerLen;

  // Extract shape and dtype from header like "{'descr': '<f4', 'fortran_order': False, 'shape': (224, 224), }"
  const shapeMatch = headerStr.match(/shape['"]\s*:\s*\((\d+),\s*(\d+)\)/);
  if (!shapeMatch) throw new Error("Cannot parse .npy shape");
  const h = parseInt(shapeMatch[1]);
  const w = parseInt(shapeMatch[2]);

  const isFloat64 = headerStr.includes("<f8") || headerStr.includes("float64");
  const bytesPerEl = isFloat64 ? 8 : 4;

  // Read raw data
  const pixels = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    pixels[i] = isFloat64
      ? view.getFloat64(dataOffset + i * bytesPerEl, true)
      : view.getFloat32(dataOffset + i * bytesPerEl, true);
  }

  // Find min/max for normalization
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i] < min) min = pixels[i];
    if (pixels[i] > max) max = pixels[i];
  }
  const range = max - min || 1;

  // Render to canvas
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const imgData = ctx.createImageData(w, h);
  for (let i = 0; i < pixels.length; i++) {
    const v = Math.round(((pixels[i] - min) / range) * 255);
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL("image/png");
}

type Result = {
  job_id: string;
  glb_url: string;
  ap_url: string;
  ap_preprocessed_url: string;
  lat_url?: string;
  lat_preprocessed_url?: string;
};

function DropZone({
  label,
  file,
  onFile,
  required,
}: {
  label: string;
  file: File | null;
  onFile: (f: File | null) => void;
  required?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    if (file.name.endsWith(".npy")) {
      npyToDataURL(file).then(setPreview).catch(() => setPreview(null));
    } else {
      setPreview(URL.createObjectURL(file));
    }
  }, [file]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  return (
    <div
      className="relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-zinc-600 bg-zinc-800/50 p-4 transition hover:border-blue-500 hover:bg-zinc-800 cursor-pointer min-h-[200px]"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.npy"
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      {preview ? (
        <img
          src={preview}
          alt={label}
          className="max-h-[180px] rounded-lg object-contain"
        />
      ) : (
        <div className="text-center text-zinc-400">
          <p className="text-lg font-medium">{label}</p>
          <p className="text-sm mt-1">
            Drop image or .npy here or click to browse
          </p>
          {!required && (
            <p className="text-xs text-zinc-500 mt-1">Optional</p>
          )}
        </div>
      )}
      {file && (
        <p className="mt-2 text-xs text-zinc-400 truncate max-w-full">
          {file.name}
        </p>
      )}
    </div>
  );
}

function App() {
  const [apFile, setApFile] = useState<File | null>(null);
  const [latFile, setLatFile] = useState<File | null>(null);
  const [preprocess, setPreprocess] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  const handleSubmit = async () => {
    if (!apFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append("ap", apFile);
    if (latFile) form.append("lat", latFile);
    form.append("preprocess", preprocess ? "true" : "false");

    try {
      const res = await fetch(API_URL, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Server error ${res.status}`);
      }
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setApFile(null);
    setLatFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-zinc-900 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4">
        <h1 className="text-2xl font-bold tracking-tight">
          X-ray to 3D Bone Reconstruction
        </h1>
      </header>

      <main className="mx-auto max-w-7xl p-6">
        {/* Upload + Result side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Inputs */}
          <section>
            <h2 className="text-lg font-semibold mb-4">Input X-rays</h2>
            <div className="grid grid-cols-2 gap-4">
              <DropZone
                label="AP View"
                file={apFile}
                onFile={setApFile}
                required
              />
              <DropZone
                label="Lateral View"
                file={latFile}
                onFile={setLatFile}
              />
            </div>

            <label className="mt-4 flex items-center gap-2 text-sm text-zinc-400 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={preprocess}
                onChange={(e) => setPreprocess(e.target.checked)}
                className="accent-blue-500 h-4 w-4"
              />
              Preprocess for real X-rays
              <span className="text-zinc-600">(disable for synthetic DRR images)</span>
            </label>

            <div className="mt-3 flex gap-3">
              <button
                onClick={handleSubmit}
                disabled={!apFile || loading}
                className="flex-1 rounded-lg bg-blue-600 px-4 py-2.5 font-medium text-white transition hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {loading ? "Reconstructing..." : "Reconstruct 3D Bone"}
              </button>
              <button
                onClick={handleReset}
                className="rounded-lg border border-zinc-600 px-4 py-2.5 text-zinc-300 transition hover:bg-zinc-800"
              >
                Reset
              </button>
            </div>

            {loading && (
              <div className="mt-4 flex items-center gap-3 text-zinc-400">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-600 border-t-blue-500" />
                Running inference... this may take a moment.
              </div>
            )}

            {error && (
              <div className="mt-4 rounded-lg bg-red-900/30 border border-red-800 p-3 text-red-300 text-sm">
                {error}
              </div>
            )}

            {/* Preprocessed images */}
            {result && (
              <div className="mt-6">
                <h3 className="text-sm font-medium text-zinc-400 mb-2">
                  Preprocessed (model input)
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="rounded-xl bg-zinc-800 p-2">
                    <img
                      src={result.ap_preprocessed_url}
                      alt="AP preprocessed"
                      className="w-full rounded-lg"
                    />
                    <p className="text-center text-xs text-zinc-500 mt-1">
                      AP
                    </p>
                  </div>
                  {result.lat_preprocessed_url && (
                    <div className="rounded-xl bg-zinc-800 p-2">
                      <img
                        src={result.lat_preprocessed_url}
                        alt="LAT preprocessed"
                        className="w-full rounded-lg"
                      />
                      <p className="text-center text-xs text-zinc-500 mt-1">
                        Lateral
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

          {/* Right: 3D Output */}
          <section>
            <h2 className="text-lg font-semibold mb-4">3D Reconstruction</h2>
            <div className="rounded-xl bg-zinc-800 border border-zinc-700 overflow-hidden aspect-square flex items-center justify-center">
              {result ? (
                <model-viewer
                  src={result.glb_url}
                  alt="3D bone reconstruction"
                  camera-controls=""
                  auto-rotate=""
                  shadow-intensity="0.5"
                  exposure="1.2"
                  style={{ width: "100%", height: "100%" }}
                />
              ) : (
                <div className="text-zinc-500 text-center p-8">
                  <svg
                    className="mx-auto mb-3 h-16 w-16 text-zinc-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1}
                  >
                    <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
                  </svg>
                  <p>Upload X-ray images and click Reconstruct</p>
                  <p className="text-sm text-zinc-600 mt-1">
                    The 3D model will appear here
                  </p>
                </div>
              )}
            </div>
            {result && (
              <a
                href={result.glb_url}
                download="bone_reconstruction.glb"
                className="mt-3 inline-flex items-center gap-2 rounded-lg border border-zinc-600 px-4 py-2 text-sm text-zinc-300 transition hover:bg-zinc-800"
              >
                <svg
                  className="h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                </svg>
                Download GLB
              </a>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
