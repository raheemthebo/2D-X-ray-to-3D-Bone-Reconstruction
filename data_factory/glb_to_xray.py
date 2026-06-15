"""
Generate synthetic X-ray images from 3D bone GLB meshes.

Casts parallel rays through the mesh and measures bone thickness
to simulate X-ray attenuation (thicker bone = brighter).

Usage:
    # Generate X-rays for all GLB files (default: AP + Lateral)
    python data_factory/glb_to_xray.py

    # Single file
    python data_factory/glb_to_xray.py data_factory/glb_output/004_SMIR.Thorax.089Y.M.CT.36.glb

    # Custom angles (degrees around Y axis, 0=front, 90=side)
    python data_factory/glb_to_xray.py --angles 0 45 90 135 180

    # Higher resolution
    python data_factory/glb_to_xray.py --resolution 1024

    # Generate many angles for training data
    python data_factory/glb_to_xray.py --num-angles 36
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

GLB_DIR = Path(__file__).parent / "glb_output"
OUTPUT_DIR = Path(__file__).parent / "xray_output"


def compute_thickness(mesh: trimesh.Trimesh, ray_origins: np.ndarray, ray_directions: np.ndarray, axis: int) -> np.ndarray:
    """Cast rays and compute bone thickness per ray."""
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True
    )

    n_rays = len(ray_origins)
    thickness = np.zeros(n_rays)

    if len(locations) == 0:
        return thickness

    for ray_idx in np.unique(index_ray):
        mask = index_ray == ray_idx
        hits = np.sort(locations[mask, axis])
        if len(hits) >= 2:
            pairs = hits[: len(hits) - len(hits) % 2].reshape(-1, 2)
            thickness[ray_idx] = np.sum(pairs[:, 1] - pairs[:, 0])

    return thickness


def render_xray(mesh: trimesh.Trimesh, angle_deg: float, resolution: int, gamma: float) -> np.ndarray:
    """Render a synthetic X-ray from the given angle.

    angle_deg: rotation around Y axis (0=front/AP, 90=right lateral, etc.)
    Returns a 2D numpy array (resolution x resolution) with pixel values 0-255.
    """
    # Rotate mesh copy around Y axis
    angle_rad = np.radians(angle_deg)
    rot = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
    rotated = mesh.copy()
    rotated.apply_transform(rot)

    ex = rotated.extents
    bounds = rotated.bounds

    # Rays along Z axis (into the screen after rotation)
    # Grid covers X and Y extent of the rotated mesh
    pad = 1.05
    x = np.linspace(bounds[0, 0] * pad, bounds[1, 0] * pad, resolution)
    y = np.linspace(bounds[0, 1] * pad, bounds[1, 1] * pad, resolution)
    xx, yy = np.meshgrid(x, y)

    z_start = bounds[0, 2] - ex[2] * 0.5
    origins = np.column_stack([xx.ravel(), yy.ravel(), np.full(resolution * resolution, z_start)])
    directions = np.tile([0, 0, 1], (resolution * resolution, 1)).astype(np.float64)

    thickness = compute_thickness(rotated, origins, directions, axis=2)
    thickness = np.flipud(thickness.reshape(resolution, resolution))

    # Apply gamma correction and normalize
    if thickness.max() > 0:
        norm = thickness / thickness.max()
        xray = np.power(norm, gamma) * 255
        xray[thickness == 0] = 0
    else:
        xray = np.zeros((resolution, resolution))

    return xray.astype(np.uint8)


def process_glb(glb_path: Path, output_dir: Path, angles: list[float], resolution: int, gamma: float):
    """Generate X-ray images for a single GLB file at multiple angles."""
    mesh = trimesh.load(str(glb_path), force="mesh")
    mesh.vertices -= mesh.centroid  # center at origin

    stem = glb_path.stem
    subject_dir = output_dir / stem
    subject_dir.mkdir(parents=True, exist_ok=True)

    for angle in angles:
        label = f"{int(angle):03d}deg"
        out_path = subject_dir / f"{label}.png"

        print(f"    {label}...", end=" ", flush=True)
        xray = render_xray(mesh, angle, resolution, gamma)

        Image.fromarray(xray).save(str(out_path))
        nonzero = np.count_nonzero(xray)
        print(f"done ({nonzero} bone pixels)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic X-rays from 3D bone meshes")
    parser.add_argument("input", nargs="?", default=str(GLB_DIR), help="GLB file or directory (default: data_factory/glb_output/)")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR), help="Output directory (default: data_factory/xray_output/)")
    parser.add_argument("-r", "--resolution", type=int, default=512, help="Image resolution in pixels (default: 512)")
    parser.add_argument("--angles", type=float, nargs="+", help="Specific angles in degrees (e.g., --angles 0 45 90)")
    parser.add_argument("--num-angles", type=int, help="Generate N evenly spaced angles (e.g., --num-angles 36 for every 10 degrees)")
    parser.add_argument("--gamma", type=float, default=0.4, help="Gamma correction (lower=brighter thin bones, default: 0.4)")
    args = parser.parse_args()

    # Determine angles
    if args.angles:
        angles = args.angles
    elif args.num_angles:
        angles = np.linspace(0, 360, args.num_angles, endpoint=False).tolist()
    else:
        # Default: AP (front) and Lateral (side) — the two standard clinical X-ray views
        angles = [0, 90]

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.suffix == ".glb":
        glb_files = [input_path]
    elif input_path.is_dir():
        glb_files = sorted(input_path.glob("*.glb"))
    else:
        print(f"Error: {input_path} not found or not a GLB file")
        sys.exit(1)

    if not glb_files:
        print(f"No GLB files found in {input_path}")
        sys.exit(1)

    angle_str = ", ".join(f"{a:.0f}" for a in angles)
    print(f"Generating X-rays: {len(glb_files)} mesh(es), {len(angles)} angle(s) [{angle_str} deg], {args.resolution}px\n")

    for glb_file in glb_files:
        print(f"  {glb_file.name}")
        process_glb(glb_file, output_dir, angles, args.resolution, args.gamma)

    total = len(glb_files) * len(angles)
    print(f"\nDone! {total} X-ray images saved to {output_dir}/")


if __name__ == "__main__":
    main()
