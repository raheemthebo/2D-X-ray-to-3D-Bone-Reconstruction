"""Convert .npy X-ray images in downloaded_samples/ to 16-bit PNGs (lossless).

Creates a .png alongside each .npy file.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

samples_dir = Path(__file__).parent / "downloaded_samples"

if not samples_dir.exists():
    print(f"Directory not found: {samples_dir}")
    sys.exit(1)

npy_files = list(samples_dir.rglob("*.npy"))
print(f"Found {len(npy_files)} .npy files")

for npy_path in npy_files:
    arr = np.load(npy_path).astype(np.float32)
    # Scale 0-1 float to 0-65535 uint16 for lossless 16-bit PNG
    img = Image.fromarray((arr * 65535).clip(0, 65535).astype(np.uint16))
    png_path = npy_path.with_suffix(".png")
    img.save(png_path)
    print(f"  {npy_path.parent.name}/{npy_path.name} -> {png_path.name}")

print(f"\nDone. Converted {len(npy_files)} files.")
