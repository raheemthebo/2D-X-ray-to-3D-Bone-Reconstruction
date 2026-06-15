"""
Load CT volumes (DICOM or NIfTI) and convert to GLB 3D bone meshes.

Supports both DICOM (.dcm slices) and NIfTI (.nii/.nii.gz) volumes.
Extracts the bone surface using marching cubes and exports as GLB.

Usage:
    # Convert a single scan series
    python data_factory/volume_loader.py data_factory/extracted/004/SMIR.Head.089Y.M.CT.31

    # Convert all scans for a subject
    python data_factory/volume_loader.py data_factory/extracted/004

    # Convert everything in extracted/
    python data_factory/volume_loader.py

    # Adjust bone threshold (default 300 HU)
    python data_factory/volume_loader.py --threshold 200

    # Simplify mesh to reduce file size
    python data_factory/volume_loader.py --decimate 0.5
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pydicom
import trimesh
from skimage.measure import marching_cubes

EXTRACTED_DIR = Path(__file__).parent / "extracted"
OUTPUT_DIR = Path(__file__).parent / "glb_output"


def load_dicom_volume(scan_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a directory of DICOM slices into a 3D numpy volume.

    Returns (volume_hu, voxel_spacing) or None if no valid slices found.
    volume_hu is in Hounsfield Units.
    voxel_spacing is (z, y, x) in mm.
    """
    dcm_files = sorted(scan_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    # Read all slices and sort by Z position
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        if hasattr(ds, "pixel_array"):
            slices.append(ds)

    if len(slices) < 2:
        return None

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # Extract pixel spacing
    pixel_spacing = [float(s) for s in slices[0].PixelSpacing]
    slice_spacing = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    if slice_spacing == 0:
        slice_spacing = float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)

    voxel_spacing = np.array([slice_spacing, pixel_spacing[0], pixel_spacing[1]])

    # Stack into 3D volume and convert to HU
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    slope = float(getattr(slices[0], "RescaleSlope", 1))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    volume_hu = volume * slope + intercept

    return volume_hu, voxel_spacing


def load_nifti_volume(nii_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a NIfTI (.nii/.nii.gz) file into a 3D numpy volume.

    Returns (volume_hu, voxel_spacing) or None if loading fails.
    volume_hu is in Hounsfield Units.
    voxel_spacing is (z, y, x) in mm.
    """
    import nibabel as nib

    try:
        img = nib.load(str(nii_path))
    except Exception:
        return None

    data = img.get_fdata()

    # Squeeze singleton dimensions (e.g. 512x512x3097x1 → 512x512x3097)
    data = np.squeeze(data)
    if data.ndim != 3:
        return None

    # NIfTI stores as (X, Y, Z) — transpose to (Z, Y, X) to match DICOM convention
    data = np.transpose(data, (2, 1, 0)).astype(np.float32)

    zooms = img.header.get_zooms()[:3]
    # zooms is (x, y, z) — reorder to (z, y, x) to match DICOM convention
    voxel_spacing = np.array([float(zooms[2]), float(zooms[1]), float(zooms[0])])

    return data, voxel_spacing


def extract_bone_mesh(volume_hu: np.ndarray, voxel_spacing: np.ndarray, threshold: float, min_faces_ratio: float = 0.001) -> trimesh.Trimesh | None:
    """Extract bone surface mesh from a CT volume using marching cubes.

    Removes small disconnected fragments (noise, CT artifacts, scan table, etc.)
    by keeping only connected components with more than min_faces_ratio of total faces.
    """
    from scipy.ndimage import binary_fill_holes, binary_opening

    bone_mask = volume_hu >= threshold

    if bone_mask.sum() < 100:
        return None

    # Light morphological cleanup: remove isolated single-voxel noise
    # Using a small cross-shaped kernel to preserve thin structures (ribs, vertebral processes)
    from scipy.ndimage import generate_binary_structure
    struct = generate_binary_structure(3, 1)  # 6-connectivity (face-adjacent only)
    bone_mask = binary_opening(bone_mask, structure=struct, iterations=1)

    if bone_mask.sum() < 100:
        return None

    try:
        verts, faces, _, _ = marching_cubes(bone_mask.astype(np.float32), level=0.5, spacing=voxel_spacing)
    except Exception:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Split into connected components and remove small fragments
    components = mesh.split(only_watertight=False)
    if not components:
        return mesh

    min_faces = max(100, int(len(mesh.faces) * min_faces_ratio))
    kept = [c for c in components if len(c.faces) >= min_faces]

    if not kept:
        # Fallback: keep the largest component
        kept = [max(components, key=lambda c: len(c.faces))]

    removed = len(components) - len(kept)
    if removed > 0:
        print(f"    Removed {removed} small fragments (kept {len(kept)} components)")

    return trimesh.util.concatenate(kept)


def convert_scan(scan_dir: Path, output_dir: Path, threshold: float, decimate: float | None) -> bool:
    """Convert a single DICOM scan series to GLB."""
    result = load_dicom_volume(scan_dir)
    if result is None:
        print(f"    Skipped (no valid DICOM slices)")
        return False

    volume_hu, voxel_spacing = result
    print(f"    Volume: {volume_hu.shape}, spacing: {voxel_spacing.round(3)} mm, HU range: [{volume_hu.min():.0f}, {volume_hu.max():.0f}]")

    mesh = extract_bone_mesh(volume_hu, voxel_spacing, threshold)
    if mesh is None:
        print(f"    Skipped (no bone found at threshold {threshold} HU)")
        return False

    if decimate and 0 < decimate < 1:
        import fast_simplification
        reduction = 1.0 - decimate  # decimate=0.1 means keep 10%, so reduce by 90%
        verts_out, faces_out = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_reduction=reduction
        )
        mesh = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=True)
        print(f"    Decimated to {len(mesh.faces)} faces")

    # Build output path: subject_scanname.glb
    subject_id = scan_dir.parent.name
    scan_name = scan_dir.name
    out_path = output_dir / f"{subject_id}_{scan_name}.glb"
    mesh.export(str(out_path))
    print(f"    {len(mesh.vertices)} verts, {len(mesh.faces)} faces → {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return True


def find_scan_dirs(input_path: Path) -> list[Path]:
    """Find directories containing .dcm files."""
    # Check if the input itself contains .dcm files
    if list(input_path.glob("*.dcm")):
        return [input_path]

    # Otherwise, look for subdirectories with .dcm files
    scan_dirs = []
    for dirpath, _, filenames in os.walk(input_path):
        if any(f.endswith(".dcm") for f in filenames):
            scan_dirs.append(Path(dirpath))
    return sorted(scan_dirs)


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM CT scans to GLB bone meshes")
    parser.add_argument("input", nargs="?", default=str(EXTRACTED_DIR), help="Scan directory, subject directory, or extracted/ root")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR), help="Output directory (default: data_factory/glb_output/)")
    parser.add_argument("-t", "--threshold", type=float, default=300, help="Bone HU threshold (default: 300)")
    parser.add_argument("--decimate", type=float, help="Reduce mesh faces by this factor (e.g., 0.5 = half)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    scan_dirs = find_scan_dirs(input_path)
    if not scan_dirs:
        print(f"No DICOM scan directories found in {input_path}")
        sys.exit(1)

    print(f"Found {len(scan_dirs)} scan(s), threshold={args.threshold} HU\n")

    converted = 0
    for scan_dir in scan_dirs:
        print(f"  {scan_dir.parent.name}/{scan_dir.name}")
        if convert_scan(scan_dir, output_dir, args.threshold, args.decimate):
            converted += 1

    print(f"\nDone! Converted {converted}/{len(scan_dirs)} scans → {output_dir}/")


if __name__ == "__main__":
    main()
