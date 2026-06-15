"""
Download real X-ray AP + Lateral pairs for visual testing of the model.

Sources:
  - Open-I (Indiana University chest X-rays, public domain)
    https://huggingface.co/datasets/ykumards/open-i

Usage:
    python fetch_test_xrays.py
    python fetch_test_xrays.py --count 10
"""

import argparse
import io
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download real X-ray AP+LAT pairs for testing")
    parser.add_argument("--output-dir", default="test_images", help="Output directory")
    parser.add_argument("--count", type=int, default=5, help="Number of pairs to download")
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "ykumards/open-i"

    print("Downloading Open-I chest X-ray dataset (AP + Lateral pairs)...")
    local = hf_hub_download(
        repo_id=repo_id, repo_type="dataset",
        filename="data/train-00000-of-00005-c80629f6027d8eb1.parquet",
    )

    df = pd.read_parquet(local)
    # Filter to rows that have both frontal and lateral images
    df = df[df["img_frontal"].notna() & df["img_lateral"].notna()].reset_index(drop=True)
    print(f"Found {len(df)} AP + Lateral pairs")

    count = min(args.count, len(df))
    for i in range(count):
        row = df.iloc[i]
        name = f"xray_{i:03d}"

        ap = Image.open(io.BytesIO(row["img_frontal"])).convert("L")
        lat = Image.open(io.BytesIO(row["img_lateral"])).convert("L")

        ap_path = output_dir / f"{name}_ap.png"
        lat_path = output_dir / f"{name}_lat.png"
        ap.save(ap_path)
        lat.save(lat_path)
        print(f"  {name}: AP {ap.size[0]}x{ap.size[1]}, LAT {lat.size[0]}x{lat.size[1]}")

    print(f"\nDone! {count} AP+LAT pairs saved to {output_dir}/")
    print(f"\nRun inference:")
    print(f"  python -m model.inference --input-dir {output_dir} --output-dir output/")
    print(f"  python -m model.inference --input-dir {output_dir} --output-dir output/ --mise")


if __name__ == "__main__":
    main()
