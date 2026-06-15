"""
FastAPI backend for X2BR bone reconstruction.

Accepts AP (+ optional LAT) X-ray uploads, runs inference, returns GLB mesh.
"""

import sys
import uuid
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root so we can import model.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.inference import get_device, load_model, infer_single

UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR = Path(__file__).parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="X2BR Bone Reconstruction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images and output GLBs as static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Lazy-loaded model globals
_model = None
_device = None


def _get_model():
    global _model, _device
    if _model is None:
        _device = get_device()
        checkpoint = PROJECT_ROOT / "model" / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint}. "
                "Download or train a model first."
            )
        _model = load_model(str(checkpoint), _device)
        print(f"Model loaded on {_device}")
    return _model, _device


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(
    ap: UploadFile = File(...),
    lat: UploadFile | None = File(None),
    preprocess: str = Form("true"),
):
    """Upload AP (+ optional LAT) X-ray images and get a 3D bone mesh back."""
    do_preprocess = preprocess.lower() not in ("false", "0", "no")

    job_id = uuid.uuid4().hex[:12]
    job_upload = UPLOAD_DIR / job_id
    job_output = OUTPUT_DIR / job_id
    job_upload.mkdir(parents=True)
    job_output.mkdir(parents=True)

    # Save uploaded files
    ap_path = job_upload / f"ap{Path(ap.filename).suffix}"
    with open(ap_path, "wb") as f:
        shutil.copyfileobj(ap.file, f)

    # Handle optional lateral — FastAPI may send an empty UploadFile instead of None
    lat_path = None
    if lat is not None and lat.filename:
        lat_path = job_upload / f"lat{Path(lat.filename).suffix}"
        with open(lat_path, "wb") as f:
            shutil.copyfileobj(lat.file, f)

    # Load model
    try:
        model, device = _get_model()
    except FileNotFoundError as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

    # Run inference
    glb_path = job_output / "prediction.glb"
    try:
        infer_single(
            model,
            str(ap_path),
            str(lat_path) if lat_path else None,
            device,
            output=str(glb_path),
            preprocess=do_preprocess,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Build response URLs (relative paths served by StaticFiles)
    result = {
        "job_id": job_id,
        "glb_url": f"/outputs/{job_id}/prediction.glb",
        "ap_url": f"/uploads/{job_id}/{ap_path.name}",
        "ap_preprocessed_url": f"/outputs/{job_id}/prediction_ap_preprocessed.png",
    }
    if lat_path:
        result["lat_url"] = f"/uploads/{job_id}/{lat_path.name}"
        result["lat_preprocessed_url"] = f"/outputs/{job_id}/prediction_lat_preprocessed.png"

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
