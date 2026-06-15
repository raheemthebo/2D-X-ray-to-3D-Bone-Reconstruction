"""
Download CADS dataset subsets from HuggingFace.

CADS (huggingface/CADS-dataset) aggregates 43 medical imaging datasets with unified
segmentation labels. We download subsets that have bone data:

  0037_totalsegmentator - 1,203 full-body CTs (all bones)
  0010_verse            - 450 vertebra CTs (spine)
  0013_ribfrac          - 360 rib CTs (ribs + chest)

For each subject we download:
  - images/{subject_id}_0000.nii.gz          - CT volume
  - segmentations/{subject_id}/{subject_id}_part_559.nii.gz - bone mask (label 5)

Usage:
    python data_factory/download_cads.py --output-dir /scratch/cads
    python data_factory/download_cads.py --output-dir ./cads --max-subjects 10
    python data_factory/download_cads.py --dry-run
"""

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ID = "huggingface/CADS-dataset"

# Subsets with bone segmentation data
DEFAULT_SUBSETS = [
    "0037_totalsegmentator",  # 1,203 full-body CTs
    "0010_verse",             # 450 vertebra CTs
    "0013_ribfrac",           # 360 rib CTs
]

# part_559 = tissue types model, label 5 = "Bones" (all bone tissue in body)
BONE_PART = "part_559"


def _hf_token() -> str | None:
    """Get HuggingFace token from env var or cached login."""
    return os.environ.get("HF_TOKEN")


def list_subsets_with_bone() -> list[str]:
    """Discover all CADS subsets that have part_559 (bone) segmentation."""
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token())
    subsets = []

    for item in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=""):
        if hasattr(item, "tree") or not hasattr(item, "rfilename"):
            name = item.path if hasattr(item, "path") else str(item)
            if name.startswith("0"):
                # Check if this subset has part_559 segmentations
                try:
                    seg_items = list(api.list_repo_tree(
                        REPO_ID, repo_type="dataset",
                        path_in_repo=f"{name}/segmentations",
                    ))
                    if seg_items:
                        # Check first subject for part_559
                        first_sub = seg_items[0]
                        sub_path = first_sub.path if hasattr(first_sub, "path") else str(first_sub)
                        sub_files = list(api.list_repo_tree(
                            REPO_ID, repo_type="dataset",
                            path_in_repo=sub_path,
                        ))
                        has_bone = any(BONE_PART in (f.rfilename if hasattr(f, "rfilename") else str(f))
                                       for f in sub_files)
                        if has_bone:
                            subsets.append(name)
                            print(f"  Found bone data: {name}")
                except Exception:
                    pass

    return sorted(subsets)


def list_subjects(subset: str) -> list[str]:
    """List all subject IDs in a CADS subset by scanning the images directory."""
    from huggingface_hub import HfApi

    api = HfApi(token=_hf_token())
    subject_ids = []

    try:
        for item in api.list_repo_tree(
            REPO_ID, repo_type="dataset",
            path_in_repo=f"{subset}/images",
        ):
            if hasattr(item, "rfilename") and item.rfilename.endswith("_0000.nii.gz"):
                filename = Path(item.rfilename).name
                subject_id = filename.replace("_0000.nii.gz", "")
                subject_ids.append(subject_id)
    except Exception as e:
        print(f"  Warning: could not list subjects for {subset}: {e}")
        return []

    return sorted(subject_ids)


def download_subject(subject_id: str, output_dir: Path, subset: str,
                     retries: int = 3, timeout: int = 300) -> bool:
    """Download image + bone segmentation mask for one subject.

    Retries up to `retries` times with a per-attempt timeout (seconds).
    """
    import time
    from huggingface_hub import hf_hub_download

    image_path = f"{subset}/images/{subject_id}_0000.nii.gz"
    seg_path = f"{subset}/segmentations/{subject_id}/{subject_id}_{BONE_PART}.nii.gz"

    token = _hf_token()
    for attempt in range(1, retries + 1):
        try:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
            hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset",
                filename=image_path, local_dir=str(output_dir),
                token=token, force_download=(attempt > 1),
            )
            hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset",
                filename=seg_path, local_dir=str(output_dir),
                token=token, force_download=(attempt > 1),
            )
            return True
        except Exception as e:
            print(f"  FAILED {subject_id} (attempt {attempt}/{retries}): {e}",
                  flush=True)
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def download_batch(subject_ids: list[str], output_dir: Path,
                   subset: str, workers: int = 4,
                   per_subject_timeout: int = 600) -> int:
    """Download a batch of subjects in parallel.

    Returns count of successful downloads. Subjects that take longer than
    per_subject_timeout seconds are skipped (thread left to die).
    """
    done = 0
    failed = 0
    total = len(subject_ids)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit all and track which subject each future belongs to
        future_to_sid = {}
        for sid in subject_ids:
            img = output_dir / subset / "images" / f"{sid}_0000.nii.gz"
            seg = output_dir / subset / "segmentations" / sid / f"{sid}_{BONE_PART}.nii.gz"
            if img.exists() and seg.exists():
                done += 1
                continue
            future = pool.submit(download_subject, sid, output_dir, subset)
            future_to_sid[future] = sid

        # Collect results with per-future timeout
        for future in as_completed(future_to_sid, timeout=per_subject_timeout * total):
            sid = future_to_sid[future]
            try:
                ok = future.result(timeout=per_subject_timeout)
                if ok:
                    done += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  TIMEOUT/ERROR {sid}: {e}", flush=True)
                failed += 1

            completed = done + failed
            if completed % 10 == 0 or completed == total:
                print(f"  [{subset}] Download: {completed}/{total} "
                      f"({failed} failed)", flush=True)

    return done


def main():
    parser = argparse.ArgumentParser(description="Download CADS dataset from HuggingFace")
    parser.add_argument("--output-dir", required=True, help="Directory to download to")
    parser.add_argument("--max-subjects", type=int, default=250,
                        help="Max subjects per subset (default: 250, 0=all)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Subjects per download batch (default: 50)")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS,
                        help="CADS subsets to download")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets:
        print(f"\n{'='*50}")
        print(f"Subset: {subset}")
        print(f"{'='*50}")

        subjects = list_subjects(subset)
        print(f"Found {len(subjects)} subjects")

        if args.max_subjects > 0:
            subjects = subjects[:args.max_subjects]
            print(f"Using first {len(subjects)}")

        if args.dry_run:
            print(f"[Dry run] Would download {len(subjects)} subjects")
            continue

        total_downloaded = 0
        for i in range(0, len(subjects), args.batch_size):
            batch = subjects[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(subjects) + args.batch_size - 1) // args.batch_size
            print(f"\n  Batch {batch_num}/{total_batches}: {len(batch)} subjects")
            done = download_batch(batch, output_path, subset)
            total_downloaded += done

        print(f"Done: {total_downloaded}/{len(subjects)} subjects downloaded")


if __name__ == "__main__":
    main()
