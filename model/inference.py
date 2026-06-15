"""
Run inference: X-ray image(s) → 3D bone mesh (GLB).

Usage:
    # Single pair (AP + Lateral)
    python -m model.inference --ap xray_ap.png --lat xray_lat.png

    # Single-view (AP only — duplicates as LAT)
    python -m model.inference --ap xray_ap.png

    # Batch: process all images in test_images/ directory
    python -m model.inference --input-dir test_images/ --output-dir output/

    # Custom checkpoint
    python -m model.inference --ap xray.png --checkpoint model/checkpoints/best.pt

Input directory structure (--input-dir):
    test_images/
        patient1_ap.png          # matched by _ap/_lat suffix
        patient1_lat.png
        patient2_ap.png          # single-view if no _lat pair
        standalone_xray.jpg      # treated as AP, single-view

Output:
    output/
        patient1.glb             # 3D bone mesh
        patient2.glb
        standalone_xray.glb
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from skimage.measure import marching_cubes

from .architecture import X2BRBiplanarModel

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_to_drr(img: np.ndarray) -> np.ndarray:
    """Normalize a real X-ray for model input.

    Applies robust normalization and mild contrast adjustment.
    The model's convolutional layers should handle feature extraction
    regardless of exact intensity distribution.
    """
    # Robust percentile normalization
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)

    # Mild gamma to boost bone contrast (γ < 1 brightens bones)
    img = np.power(img, 0.7)

    return img.astype(np.float32)


def load_xray(path: str, target_size: int = 224, preprocess: bool = True) -> np.ndarray:
    """Load an X-ray image as a normalized float32 array.

    Args:
        preprocess: If True, normalize real X-ray images to match DRR distribution.
                    Set False for .npy files (already in DRR format).
    """
    path = Path(path)

    if path.suffix == ".npy":
        img = np.load(path).astype(np.float32)
        preprocess = False  # .npy files are already DRRs
    else:
        img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0

    # Resize if needed
    if img.shape != (target_size, target_size):
        from scipy.ndimage import zoom
        factors = [target_size / s for s in img.shape]
        img = zoom(img, factors, order=1)

    if preprocess:
        img = normalize_to_drr(img)

    return img


def voxels_to_mesh(voxels: np.ndarray, threshold: float = 0.5, smooth_iterations: int = 3) -> trimesh.Trimesh:
    """Convert voxel grid to a triangle mesh."""
    binary = (voxels > threshold).astype(np.float32)

    if binary.sum() < 10:
        raise ValueError("No bone structure detected in prediction")

    verts, faces, _, _ = marching_cubes(binary, level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)

    return mesh


def load_model(checkpoint: str, device: torch.device) -> X2BRBiplanarModel:
    """Load model from checkpoint."""
    model = X2BRBiplanarModel()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        dice = ckpt.get("best_val_dice", ckpt.get("val_dice", "?"))
        print(f"Loaded checkpoint: epoch {epoch}, dice {dice}")
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def infer_single(model, ap_path: str, lat_path: str | None, device: torch.device,
                  output: str = "prediction.glb", use_mise: bool = False,
                  mise_resolution: int = 256, preprocess: bool = True) -> trimesh.Trimesh:
    """Run inference on a single pair and export as GLB."""
    ap = load_xray(ap_path, preprocess=preprocess)
    lat = load_xray(lat_path, preprocess=preprocess) if lat_path else ap.copy()

    # Save preprocessed X-rays alongside the output
    out_dir = Path(output).parent
    out_stem = Path(output).stem
    ap_out = out_dir / f"{out_stem}_ap_preprocessed.png"
    lat_out = out_dir / f"{out_stem}_lat_preprocessed.png"
    Image.fromarray((ap * 255).clip(0, 255).astype(np.uint8)).save(ap_out)
    Image.fromarray((lat * 255).clip(0, 255).astype(np.uint8)).save(lat_out)
    print(f"  Preprocessed X-rays: {ap_out.name}, {lat_out.name}")

    ap_t = torch.from_numpy(ap).unsqueeze(0).unsqueeze(0).to(device)
    lat_t = torch.from_numpy(lat).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_mise:
            print(f"  Using MISE (target resolution: {mise_resolution}³)...")
            pred = model(ap_t, lat_t, mise=True, mise_resolution=mise_resolution)
        else:
            pred = model(ap_t, lat_t)

    voxels = pred.squeeze().cpu().numpy()
    print(f"  Prediction: {voxels.shape[0]}³ grid, range=[{voxels.min():.3f}, {voxels.max():.3f}], "
          f"bone voxels={int((voxels > 0.5).sum())}/{voxels.size}")

    mesh = voxels_to_mesh(voxels)
    mesh.export(output)
    print(f"  Exported: {output} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    return mesh


def find_image_pairs(input_dir: Path) -> list[tuple[str, Path, Path | None]]:
    """Find AP/LAT image pairs in a directory.

    Matching rules:
    - Files ending with _ap.* and _lat.* with same prefix → paired
    - Other image files → treated as single-view AP
    """
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".npy"}
    all_images = sorted([f for f in input_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in image_exts])

    # Find AP/LAT pairs
    ap_files = {}
    lat_files = {}
    standalone = []

    for img in all_images:
        stem = img.stem.lower()
        if stem.endswith("_ap"):
            key = img.stem[:-3]  # preserve original case for key
            ap_files[key] = img
        elif stem.endswith("_lat") or stem.endswith("_lateral"):
            key = img.stem[:-4] if stem.endswith("_lat") else img.stem[:-8]
            lat_files[key] = img
        else:
            standalone.append(img)

    pairs = []

    # Matched pairs
    for key in ap_files:
        ap = ap_files[key]
        lat = lat_files.get(key)
        pairs.append((key, ap, lat))
        if key in lat_files:
            del lat_files[key]

    # Unmatched LAT files (treat as single-view)
    for key, lat in lat_files.items():
        pairs.append((key, lat, None))

    # Standalone files
    for img in standalone:
        pairs.append((img.stem, img, None))

    return pairs


def infer_batch(model, input_dir: Path, output_dir: Path, device: torch.device,
                use_mise: bool = False, mise_resolution: int = 256,
                preprocess: bool = True):
    """Process all images in input_dir, output GLBs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_image_pairs(input_dir)
    if not pairs:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(pairs)} image(s) to process:")
    for name, ap, lat in pairs:
        lat_str = f" + {lat.name}" if lat else " (single-view)"
        print(f"  {ap.name}{lat_str}")
    if use_mise:
        print(f"  MISE enabled (target: {mise_resolution}³)")
    if preprocess:
        print(f"  DRR preprocessing: enabled")
    print()

    for name, ap_path, lat_path in pairs:
        output_path = output_dir / f"{name}.glb"
        print(f"Processing: {name}")
        try:
            infer_single(model, str(ap_path), str(lat_path) if lat_path else None,
                          device, str(output_path), use_mise=use_mise,
                          mise_resolution=mise_resolution, preprocess=preprocess)
        except Exception as e:
            print(f"  FAILED: {e}")
        print()

    print(f"Done! Results in: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="X-ray to 3D bone inference")
    parser.add_argument("--ap", help="AP X-ray image (.png or .npy)")
    parser.add_argument("--lat", help="Lateral X-ray image (optional)")
    parser.add_argument("--input-dir", help="Directory of X-ray images for batch processing")
    parser.add_argument("--output-dir", default="output", help="Output directory for batch mode (default: output/)")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH), help="Model checkpoint path")
    parser.add_argument("-o", "--output", default="prediction.glb", help="Output GLB path (single mode)")
    parser.add_argument("--mise", action="store_true",
                        help="Use MISE for high-resolution output (default: off)")
    parser.add_argument("--mise-resolution", type=int, default=256,
                        help="Target MISE grid resolution (default: 256)")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip DRR normalization for real X-rays (use raw pixel values)")
    args = parser.parse_args()

    if not args.ap and not args.input_dir:
        parser.error("Provide either --ap (single image) or --input-dir (batch mode)")

    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    preprocess = not args.no_preprocess

    if args.input_dir:
        infer_batch(model, Path(args.input_dir), Path(args.output_dir), device,
                    use_mise=args.mise, mise_resolution=args.mise_resolution,
                    preprocess=preprocess)
    else:
        infer_single(model, args.ap, args.lat, device, args.output,
                     use_mise=args.mise, mise_resolution=args.mise_resolution,
                     preprocess=preprocess)


if __name__ == "__main__":
    main()
