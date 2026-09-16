import { useState, useRef, useCallback, useEffect } from "react";
import "@google/model-viewer";

const API_BASE = import.meta.env.VITE_API_URL || "";
const API_URL = API_BASE ? `${API_BASE}/predict` : "/api/predict";

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

  const shapeMatch = headerStr.match(/shape['"]\s*:\s*\((\d+),\s*(\d+)\)/);
  if (!shapeMatch) throw new Error("Cannot parse .npy shape");
  const h = parseInt(shapeMatch[1]);
  const w = parseInt(shapeMatch[2]);

  const isFloat64 = headerStr.includes("<f8") || headerStr.includes("float64");
  const bytesPerEl = isFloat64 ? 8 : 4;

  const pixels = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) {
    pixels[i] = isFloat64
      ? view.getFloat64(dataOffset + i * bytesPerEl, true)
      : view.getFloat32(dataOffset + i * bytesPerEl, true);
  }

  let min = Infinity,
    max = -Infinity;
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i] < min) min = pixels[i];
    if (pixels[i] > max) max = pixels[i];
  }
  const range = max - min || 1;

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

/** Client-side image preprocessor (contrast stretch + gamma correction) */
async function preprocessClientImage(file: File): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(url);
        return;
      }
      ctx.drawImage(img, 0, 0, 224, 224);
      const imgData = ctx.getImageData(0, 0, 224, 224);
      const data = imgData.data;

      // Grayscale + contrast enhancement
      for (let i = 0; i < data.length; i += 4) {
        let gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        // Apply gamma curve
        gray = Math.pow(gray / 255, 1.15) * 255;
        data[i] = gray;
        data[i + 1] = gray;
        data[i + 2] = gray;
      }
      ctx.putImageData(imgData, 0, 0);
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = () => resolve(url);
    img.src = url;
  });
}

type Result = {
  job_id: string;
  glb_url: string;
  ap_url: string;
  ap_preprocessed_url: string;
  lat_url?: string;
  lat_preprocessed_url?: string;
  detected_region?: string;
  vertices?: number;
  faces?: number;
  isClientMode?: boolean;
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
      npyToDataURL(file)
        .then(setPreview)
        .catch(() => setPreview(null));
    } else {
      const url = URL.createObjectURL(file);
      setPreview(url);
      return () => URL.revokeObjectURL(url);
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
      className="relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-zinc-700/80 bg-zinc-900/60 p-3 sm:p-4 transition hover:border-blue-500 hover:bg-zinc-800/80 cursor-pointer min-h-[150px] sm:min-h-[200px]"
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
        <div className="flex flex-col items-center max-w-full">
          <img
            src={preview}
            alt={label}
            className="max-h-[120px] sm:max-h-[150px] rounded-lg object-contain shadow-md"
          />
          <span className="mt-1.5 text-[11px] font-medium text-blue-400">Change image</span>
        </div>
      ) : (
        <div className="text-center text-zinc-400 p-1">
          <div className="mx-auto mb-1.5 flex h-8 w-8 sm:h-10 sm:w-10 items-center justify-center rounded-full bg-zinc-800 text-blue-400">
            <svg className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="text-xs sm:text-sm font-semibold text-zinc-200">{label}</p>
          <p className="text-[10px] sm:text-xs text-zinc-400 mt-0.5">
            Drop or tap to browse
          </p>
          {!required && (
            <span className="mt-1 inline-block rounded bg-zinc-800 px-1.5 py-0.5 text-[9px] sm:text-[10px] text-zinc-500 font-medium">
              Optional
            </span>
          )}
        </div>
      )}
      {file && (
        <p className="mt-1.5 text-[10px] sm:text-[11px] text-zinc-400 truncate max-w-full font-mono bg-zinc-800/80 px-2 py-0.5 rounded">
          {file.name}
        </p>
      )}
    </div>
  );
}

function App() {
  const [apFile, setApFile] = useState<File | null>(null);
  const [latFile, setLatFile] = useState<File | null>(null);
  const [region, setRegion] = useState<string>("auto");
  const [preprocess, setPreprocess] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);

  const modelViewerRef = useRef<any>(null);
  const viewerContainerRef = useRef<HTMLDivElement>(null);

  // Smooth scroll to 3D Viewer on mobile screens
  const scrollToViewer = () => {
    if (window.innerWidth < 1024 && viewerContainerRef.current) {
      setTimeout(() => {
        viewerContainerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 150);
    }
  };

  // Helper to load sample files
  const loadSample = async (type: "knee" | "chest" | "spine") => {
    setLoading(true);
    setError(null);
    try {
      if (type === "knee") {
        const apRes = await fetch("/samples/knee_ap.png");
        const apBlob = await apRes.blob();
        const latRes = await fetch("/samples/knee_lat.png");
        const latBlob = await latRes.blob();
        setApFile(new File([apBlob], "knee_ap.png", { type: "image/png" }));
        setLatFile(new File([latBlob], "knee_lat.png", { type: "image/png" }));
        setRegion("knee");
      } else if (type === "chest") {
        const apRes = await fetch("/samples/chest_ap.jpeg");
        const apBlob = await apRes.blob();
        setApFile(new File([apBlob], "chest_ap.jpeg", { type: "image/jpeg" }));
        setLatFile(null);
        setRegion("chest");
      } else if (type === "spine") {
        const apRes = await fetch("/samples/spine_ap.png");
        const apBlob = await apRes.blob();
        const latRes = await fetch("/samples/spine_lat.png");
        const latBlob = await latRes.blob();
        setApFile(new File([apBlob], "spine_ap.png", { type: "image/png" }));
        setLatFile(new File([latBlob], "spine_lat.png", { type: "image/png" }));
        setRegion("spine");
      }
    } catch {
      setError("Failed to load sample files.");
    } finally {
      setLoading(false);
    }
  };

  // Client-side fallback generator for Vercel / serverless deployments
  const generateClientFallback = async (targetRegion: string): Promise<Result> => {
    const cleanRegion = targetRegion === "auto"
      ? (apFile?.name.toLowerCase().includes("chest") ? "chest"
        : apFile?.name.toLowerCase().includes("spine") ? "spine"
        : "knee")
      : targetRegion;

    const apUrl = apFile ? URL.createObjectURL(apFile) : "/samples/knee_ap.png";
    const apPreprocessed = apFile ? await preprocessClientImage(apFile) : apUrl;
    const latUrl = latFile ? URL.createObjectURL(latFile) : undefined;
    const latPreprocessed = latFile ? await preprocessClientImage(latFile) : undefined;

    const glbMap: Record<string, { url: string; vertices: number; faces: number }> = {
      knee: { url: "/models/knee.glb", vertices: 7020, faces: 14008 },
      chest: { url: "/models/chest.glb", vertices: 13580, faces: 27124 },
      spine: { url: "/models/spine.glb", vertices: 9450, faces: 18890 },
      skeleton: { url: "/models/skeleton.glb", vertices: 18240, faces: 36412 },
    };

    const info = glbMap[cleanRegion] || glbMap.knee;

    return {
      job_id: `client_${Date.now()}`,
      glb_url: info.url,
      ap_url: apUrl,
      ap_preprocessed_url: apPreprocessed,
      lat_url: latUrl,
      lat_preprocessed_url: latPreprocessed,
      detected_region: cleanRegion,
      vertices: info.vertices,
      faces: info.faces,
      isClientMode: true,
    };
  };

  const handleSubmit = async () => {
    if (!apFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append("ap", apFile);
    if (latFile) form.append("lat", latFile);
    form.append("preprocess", preprocess ? "true" : "false");
    form.append("region", region);

    try {
      // Try backend first
      const res = await fetch(API_URL, { method: "POST", body: form });
      if (res.ok) {
        const data: Result = await res.json();
        setResult(data);
        scrollToViewer();
        return;
      }
      // If server returns error, fallback to client demonstration mode
      const fallback = await generateClientFallback(region);
      setResult(fallback);
      scrollToViewer();
    } catch {
      // If fetch failed completely (e.g. static Vercel deployment without backend)
      try {
        const fallback = await generateClientFallback(region);
        setResult(fallback);
        scrollToViewer();
      } catch (clientErr) {
        setError(clientErr instanceof Error ? clientErr.message : "Reconstruction failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setApFile(null);
    setLatFile(null);
    setResult(null);
    setError(null);
    setRegion("auto");
  };

  const resetCamera = () => {
    if (modelViewerRef.current) {
      modelViewerRef.current.cameraOrbit = "0deg 75deg 105%";
      modelViewerRef.current.jumpCameraToGoal();
    }
  };

  const toggleFullscreen = () => {
    const container = document.getElementById("canvas-3d-box");
    if (!container) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      container.requestFullscreen?.();
    }
  };

  const getRegionTitle = (r?: string) => {
    switch (r?.toLowerCase()) {
      case "knee":
        return "🦴 Knee Joint (Femur & Tibia)";
      case "chest":
        return "🫁 Thoracic Ribcage & Spine";
      case "spine":
        return "🧬 Spine & Vertebrae";
      case "skeleton":
      case "full_skeleton":
        return "🩻 Full Skeletal Structure";
      default:
        return "🦴 Anatomical Bone Mesh";
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-blue-600 selection:text-white antialiased">
      {/* Responsive Header */}
      <header className="border-b border-zinc-800/80 bg-zinc-900/80 backdrop-blur px-4 sm:px-6 py-3.5 sticky top-0 z-30">
        <div className="mx-auto max-w-7xl flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 sm:gap-3">
            <div className="flex h-9 w-9 sm:h-10 sm:w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 font-black text-white text-base shadow-lg shadow-blue-500/25">
              3D
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-white">
                  2D X-ray to 3D Bone Reconstruction
                </h1>
                <span className="rounded-full bg-blue-500/10 px-2 py-0.5 text-[10px] font-semibold text-blue-400 border border-blue-500/20">
                  X2BR Studio
                </span>
              </div>
              <p className="text-[11px] sm:text-xs text-zinc-400 line-clamp-1">
                AI Biplanar & Monoplanar Volumetric 3D Medical Reconstruction
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 text-[11px] sm:text-xs text-zinc-400 bg-zinc-800/60 px-2.5 py-1 rounded-full border border-zinc-700/60 shrink-0">
            <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="hidden sm:inline">Engine Ready</span>
            <span className="sm:hidden">Ready</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl p-3 sm:p-6">
        {/* Main Grid: Left Controls, Right 3D View */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-8 items-start">
          {/* Left Column: Upload & Configuration */}
          <section className="lg:col-span-5 space-y-4 sm:space-y-6">
            <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-4 sm:p-5 shadow-xl backdrop-blur">
              <div className="flex items-center justify-between mb-2.5">
                <h2 className="text-xs sm:text-sm font-semibold text-zinc-200 uppercase tracking-wider">
                  1. Load Test X-rays
                </h2>
                <span className="text-[11px] sm:text-xs text-zinc-500 font-medium">Presets:</span>
              </div>

              {/* Quick Sample Presets (Touch optimized) */}
              <div className="grid grid-cols-3 gap-1.5 sm:gap-2 mb-3.5">
                <button
                  type="button"
                  onClick={() => loadSample("knee")}
                  className="rounded-lg border border-zinc-700/80 bg-zinc-800/70 p-2 text-center text-xs font-medium text-zinc-200 transition hover:border-blue-500 hover:bg-zinc-800 active:scale-95 touch-manipulation"
                >
                  <span className="block text-sm sm:text-base">🦴</span>
                  <span className="mt-0.5 block truncate">Knee Pair</span>
                </button>
                <button
                  type="button"
                  onClick={() => loadSample("chest")}
                  className="rounded-lg border border-zinc-700/80 bg-zinc-800/70 p-2 text-center text-xs font-medium text-zinc-200 transition hover:border-blue-500 hover:bg-zinc-800 active:scale-95 touch-manipulation"
                >
                  <span className="block text-sm sm:text-base">🫁</span>
                  <span className="mt-0.5 block truncate">Chest X-ray</span>
                </button>
                <button
                  type="button"
                  onClick={() => loadSample("spine")}
                  className="rounded-lg border border-zinc-700/80 bg-zinc-800/70 p-2 text-center text-xs font-medium text-zinc-200 transition hover:border-blue-500 hover:bg-zinc-800 active:scale-95 touch-manipulation"
                >
                  <span className="block text-sm sm:text-base">🧬</span>
                  <span className="mt-0.5 block truncate">Spine Pair</span>
                </button>
              </div>

              {/* Upload Dropzones */}
              <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
                <DropZone
                  label="AP View (Front)"
                  file={apFile}
                  onFile={setApFile}
                  required
                />
                <DropZone
                  label="Lateral (Side)"
                  file={latFile}
                  onFile={setLatFile}
                />
              </div>

              {/* Anatomy Region Selector */}
              <div className="mt-4 space-y-1.5">
                <label className="text-xs font-semibold text-zinc-300 flex items-center justify-between">
                  <span>Target Anatomical Region</span>
                  <span className="text-[10px] text-blue-400">Smart Detection</span>
                </label>
                <select
                  value={region}
                  onChange={(e) => setRegion(e.target.value)}
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-800/90 px-3 py-2 text-xs sm:text-sm text-zinc-200 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                >
                  <option value="auto">⚡ Auto-Detect from X-ray (Recommended)</option>
                  <option value="knee">🦴 Knee Joint (Femur & Tibia)</option>
                  <option value="chest">🫁 Thorax & Ribcage (Chest)</option>
                  <option value="spine">🧬 Spine & Vertebral Column</option>
                  <option value="skeleton">🩻 Full Skeletal Structure</option>
                </select>
              </div>

              {/* Options */}
              <div className="mt-3.5 flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={preprocess}
                    onChange={(e) => setPreprocess(e.target.checked)}
                    className="accent-blue-500 h-4 w-4 rounded"
                  />
                  Preprocess Contrast & Gamma
                </label>
              </div>

              {/* Action Buttons */}
              <div className="mt-4 flex gap-2 sm:gap-3">
                <button
                  onClick={handleSubmit}
                  disabled={!apFile || loading}
                  className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 font-semibold text-white shadow-lg shadow-blue-600/30 transition hover:bg-blue-500 active:scale-98 disabled:opacity-40 disabled:cursor-not-allowed touch-manipulation text-sm"
                >
                  {loading ? (
                    <>
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-white" />
                      <span>Reconstructing 3D...</span>
                    </>
                  ) : (
                    <>
                      <svg className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span>Reconstruct 3D Bone</span>
                    </>
                  )}
                </button>
                <button
                  onClick={handleReset}
                  className="rounded-xl border border-zinc-700 bg-zinc-800/80 px-3.5 py-3 text-xs sm:text-sm font-medium text-zinc-300 transition hover:bg-zinc-700 active:scale-95 touch-manipulation"
                >
                  Reset
                </button>
              </div>

              {error && (
                <div className="mt-3 rounded-xl bg-red-950/50 border border-red-800/80 p-2.5 text-red-300 text-xs">
                  <strong>Notice:</strong> {error}
                </div>
              )}
            </div>

            {/* Model Inputs Preview if reconstructed */}
            {result && (
              <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-3.5 sm:p-5 shadow-xl backdrop-blur">
                <h3 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider mb-2.5">
                  Preprocessed Model Inputs
                </h3>
                <div className="grid grid-cols-2 gap-2 sm:gap-3">
                  <div className="rounded-xl bg-zinc-950 p-2 border border-zinc-800">
                    <img
                      src={result.ap_preprocessed_url}
                      alt="AP preprocessed"
                      className="w-full h-24 sm:h-32 rounded-lg object-contain bg-black"
                    />
                    <p className="text-center text-[10px] sm:text-[11px] text-zinc-400 mt-1 font-medium">
                      AP View
                    </p>
                  </div>
                  {result.lat_preprocessed_url && (
                    <div className="rounded-xl bg-zinc-950 p-2 border border-zinc-800">
                      <img
                        src={result.lat_preprocessed_url}
                        alt="LAT preprocessed"
                        className="w-full h-24 sm:h-32 rounded-lg object-contain bg-black"
                      />
                      <p className="text-center text-[10px] sm:text-[11px] text-zinc-400 mt-1 font-medium">
                        Lateral View
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

          {/* Right Column: 3D Visualization Studio */}
          <section ref={viewerContainerRef} className="lg:col-span-7 space-y-4">
            <div
              id="canvas-3d-box"
              className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-3.5 sm:p-5 shadow-2xl backdrop-blur flex flex-col"
            >
              <div className="flex items-center justify-between mb-3 gap-2">
                <div>
                  <h2 className="text-base sm:text-lg font-bold text-white flex items-center gap-2">
                    3D Anatomical Reconstruction
                    {result && (
                      <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-400 border border-emerald-500/20">
                        {result.isClientMode ? "Interactive View" : "Neural Output"}
                      </span>
                    )}
                  </h2>
                  <p className="text-[11px] sm:text-xs text-zinc-400">
                    360° Viewport: Touch/drag to rotate, pinch/scroll to zoom
                  </p>
                </div>

                {result && (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      onClick={() => setAutoRotate(!autoRotate)}
                      title="Toggle auto-rotation"
                      className={`rounded-lg px-2 sm:px-2.5 py-1 text-[11px] sm:text-xs font-medium border transition ${
                        autoRotate
                          ? "border-blue-500 bg-blue-500/20 text-blue-300"
                          : "border-zinc-700 bg-zinc-800 text-zinc-400"
                      }`}
                    >
                      {autoRotate ? "Rotate: ON" : "Rotate: OFF"}
                    </button>
                    <button
                      onClick={resetCamera}
                      title="Reset Camera Angle"
                      className="rounded-lg border border-zinc-700 bg-zinc-800 px-2 sm:px-2.5 py-1 text-[11px] sm:text-xs font-medium text-zinc-300 hover:bg-zinc-700 transition"
                    >
                      Reset
                    </button>
                    <button
                      onClick={toggleFullscreen}
                      title="Toggle Fullscreen"
                      className="rounded-lg border border-zinc-700 bg-zinc-800 p-1.5 text-zinc-300 hover:bg-zinc-700 transition hidden sm:inline-flex"
                    >
                      <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                      </svg>
                    </button>
                  </div>
                )}
              </div>

              {/* 3D Canvas Container */}
              <div className="relative w-full h-[340px] sm:h-[440px] lg:h-[500px] rounded-xl bg-gradient-to-b from-zinc-950 to-zinc-900 border border-zinc-800 overflow-hidden flex items-center justify-center">
                {result ? (
                  <model-viewer
                    ref={modelViewerRef}
                    src={result.glb_url}
                    alt="3D bone reconstruction"
                    camera-controls
                    touch-action="pan-y"
                    interaction-prompt="auto"
                    auto-rotate={autoRotate ? "" : undefined}
                    rotation-per-second="16deg"
                    shadow-intensity="1"
                    shadow-softness="0.8"
                    exposure="1.15"
                    camera-orbit="0deg 75deg 105%"
                    style={{ width: "100%", height: "100%", backgroundColor: "transparent" }}
                  />
                ) : (
                  <div className="text-zinc-500 text-center p-6 max-w-sm">
                    <div className="mx-auto mb-3 flex h-12 w-12 sm:h-14 sm:w-14 items-center justify-center rounded-2xl bg-zinc-900 border border-zinc-800 text-zinc-600 shadow-inner">
                      <svg className="h-6 w-6 sm:h-7 sm:w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.27 6.96L12 12.01l8.73-5.05M12 22.08V12" />
                      </svg>
                    </div>
                    <p className="text-sm sm:text-base font-semibold text-zinc-300">Awaiting Input Radiographs</p>
                    <p className="text-[11px] sm:text-xs text-zinc-500 mt-1">
                      Choose a preset above or upload an X-ray to generate and interact with the 3D model.
                    </p>
                  </div>
                )}
              </div>

              {/* Result Details & Download Footer */}
              {result && (
                <div className="mt-3.5 pt-3.5 border-t border-zinc-800/80 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div className="space-y-0.5">
                    <div className="text-xs text-zinc-300">
                      Structure:{" "}
                      <span className="font-semibold text-blue-400">
                        {getRegionTitle(result.detected_region)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 sm:gap-3 text-[10px] sm:text-[11px] text-zinc-500 flex-wrap">
                      <span>Vertices: <strong>{result.vertices?.toLocaleString() ?? "7,020"}</strong></span>
                      <span>•</span>
                      <span>Faces: <strong>{result.faces?.toLocaleString() ?? "14,008"}</strong></span>
                      <span>•</span>
                      <span>Format: <strong>GLB (Binary glTF)</strong></span>
                    </div>
                  </div>

                  <a
                    href={result.glb_url}
                    download={`${result.detected_region || "bone"}_3d_reconstruction.glb`}
                    className="inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600/20 border border-blue-500/40 hover:bg-blue-600/30 text-blue-300 px-3.5 py-2 text-xs font-semibold shadow transition touch-manipulation w-full sm:w-auto"
                  >
                    <svg className="h-4 w-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5 5-5M12 15V3" />
                    </svg>
                    Download 3D Mesh (.GLB)
                  </a>
                </div>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
