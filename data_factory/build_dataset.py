"""
Data pipeline: download + generate training data from the CADS dataset (HuggingFace).

Uses multiple subsets of CADS (huggingface/CADS-dataset):
  - 0037_totalsegmentator: 1,203 full-body CTs (all bones)
  - 0010_verse: 450 vertebra CTs (spine)
  - 0013_ribfrac: 360 rib CTs (ribs + chest)
  - part_559 label 5 = "Bones" (all bone tissue)

Batch processing: downloads N subjects at a time, generates training samples
(DRR X-rays + voxel grids), deletes raw data, moves to next batch.

DRR generation:
    DRRs (Digitally Reconstructed Radiographs) are generated from the full CT
    volume (not the bone mask), so they include soft tissue attenuation — just
    like real X-rays. The bone segmentation mask is used only for the 64³ voxel
    target. Each subject produces both AP and lateral projections at each angle:
      - AP: beam front-to-back (sum along Y axis) → frontal view
      - LAT: beam left-to-right (sum along X axis) → side view
      - Small angular jitter (±5-15°) around vertical axis for augmentation

Usage:
    # Full pipeline (download + process, 250 subjects per subset)
    python data_factory/build_dataset.py --output-dir /scratch/training_data

    # Smaller run (50 subjects, 4 angle variations)
    python data_factory/build_dataset.py --output-dir ./training_data --max-subjects 50 --num-angles 4

    # Use pre-downloaded data
    python data_factory/build_dataset.py --cads-dir /scratch/cads --skip-download --output-dir ./training_data
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate as ndrotate, zoom

# Add parent directory to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

VOXEL_RES = 64
XRAY_RES = 224
PROGRESS_FILE = "progress.json"

CADS_REPO = "huggingface/CADS-dataset"
CADS_SUBSETS = [
    "0037_totalsegmentator",  # 1,203 full-body CTs
    "0010_verse",             # 450 vertebra CTs
    "0013_ribfrac",           # 360 rib CTs
]
BONE_PART = "part_559"
BONE_LABEL = 5  # label 5 in part_559 = "Bones"


# ─── Core processing functions ─────────────────────────────────────────

def generate_drr(volume: np.ndarray, axis: int) -> np.ndarray:
    """Generate a DRR X-ray by summing attenuation along an axis.

    Works with any input resolution — the projection is resized to 224×224.
    Applies gamma correction (γ=0.5) to enhance bone/soft-tissue contrast,
    mimicking the nonlinear response of real X-ray detectors.
    """
    projection = np.sum(volume.astype(np.float32), axis=axis)
    factors = [XRAY_RES / s for s in projection.shape]
    projection = zoom(projection, factors, order=1)
    if projection.max() > 0:
        projection = projection / projection.max()
    # Gamma correction: γ < 1 brightens highlights (bone) relative to shadows
    projection = np.power(projection, 0.5)
    return np.flipud(projection).astype(np.float32)


def rotate_volume(volume: np.ndarray, angle_deg: float, order: int = 0) -> np.ndarray:
    """Rotate volume around the vertical Z-axis (YX plane).

    After NIfTI transpose, dims are (Z, Y, X) where Z is vertical (head-to-toe).
    Rotation around Z simulates the patient turning left/right, which is how
    oblique X-ray views are taken clinically.

    Args:
        order: interpolation order. 0=nearest for binary masks, 1=linear for
               continuous data like CT Hounsfield units.
    """
    if angle_deg == 0:
        return volume
    rotated = ndrotate(volume.astype(np.float32), angle_deg,
                       axes=(1, 2), reshape=False, order=order,
                       mode="constant", cval=0)
    if order == 0:
        return (rotated > 0.5).astype(volume.dtype)
    return rotated.astype(volume.dtype)


def load_nifti(path: Path) -> np.ndarray | None:
    """Load a NIfTI volume and transpose to (Z, Y, X)."""
    import nibabel as nib

    try:
        img = nib.load(str(path))
    except Exception as e:
        print(f"    Warning: failed to load {path}: {e}")
        return None

    data = np.squeeze(img.get_fdata())
    if data.ndim != 3:
        return None

    # NIfTI is (X, Y, Z) -> transpose to (Z, Y, X)
    return np.transpose(data, (2, 1, 0))


def load_ct_and_mask(ct_path: Path, mask_path: Path,
                     bone_label: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """Load CT volume + bone mask, return (ct_volume, 64³_bone_voxels).

    The CT volume is used for DRR generation (realistic X-rays with soft tissue).
    The bone mask is downsampled to 64³ as the model's training target.
    """
    ct = load_nifti(ct_path)
    mask = load_nifti(mask_path)

    if ct is None or mask is None:
        return None

    if bone_label > 0:
        bone_mask = (mask == bone_label).astype(np.uint8)
    else:
        bone_mask = (mask > 0).astype(np.uint8)

    if bone_mask.sum() < 100:
        return None

    # Downsample bone mask to 64³ for voxel target
    factors = [VOXEL_RES / s for s in bone_mask.shape]
    downsampled = zoom(bone_mask.astype(np.float32), factors, order=0)

    voxels = np.zeros((VOXEL_RES, VOXEL_RES, VOXEL_RES), dtype=np.uint8)
    s = [min(VOXEL_RES, downsampled.shape[i]) for i in range(3)]
    voxels[:s[0], :s[1], :s[2]] = (downsampled[:s[0], :s[1], :s[2]] > 0.5).astype(np.uint8)

    # Apply X-ray attenuation model to CT Hounsfield units.
    # Real X-rays: bone appears bright (high attenuation), soft tissue dim,
    # air/lung nearly transparent. We map HU values to linear attenuation
    # coefficients that approximate real X-ray physics:
    #   Air (-1000 HU) → ~0 attenuation
    #   Soft tissue (0-100 HU) → low attenuation
    #   Bone (300-3000 HU) → high attenuation
    ct = ct.astype(np.float32)
    ct = np.clip(ct, -1000, 3000)

    # Piecewise linear attenuation: bone gets 3x weight vs soft tissue
    # This mimics real X-ray contrast where bone is clearly brighter
    atten = np.zeros_like(ct)
    # Air region: -1000 to -200 HU → very low attenuation (0.0 to 0.05)
    air = ct < -200
    atten[air] = np.clip((ct[air] + 1000) / 800 * 0.05, 0, 0.05)
    # Soft tissue: -200 to 300 HU → low-medium attenuation (0.05 to 0.2)
    soft = (ct >= -200) & (ct < 300)
    atten[soft] = 0.05 + (ct[soft] + 200) / 500 * 0.15
    # Bone: 300+ HU → high attenuation (0.2 to 1.0)
    bone = ct >= 300
    atten[bone] = 0.2 + (ct[bone] - 300) / 2700 * 0.8

    # Use segmentation mask to boost bone attenuation precisely.
    # HU thresholds alone can miss cancellous bone (low-density) or
    # misclassify calcified soft tissue. The segmentation mask identifies
    # exact bone voxels — we ensure they have at least 0.5 attenuation.
    bone_seg = bone_mask.astype(bool)
    atten[bone_seg] = np.maximum(atten[bone_seg], 0.5)

    return atten, voxels


def make_augmentation_angles(num_angles: int, rng: np.random.Generator) -> list[float]:
    """Generate small rotation angles for augmentation.

    Each sample already produces both AP and LAT projections (summing along
    different axes), so we don't need 90° rotations. Instead, we generate
    small angular jitter (±5-15°) around the patient's natural orientation
    to simulate slight positioning variation — as happens in real clinical
    X-rays where the patient isn't perfectly aligned.

    Returns a list of rotation angles in degrees.
    """
    if num_angles <= 1:
        return [0.0]

    # Start with the canonical orientation (0°)
    angles = [0.0]

    # Add small jitter variations
    for _ in range(num_angles - 1):
        jitter = rng.uniform(5, 15) * rng.choice([-1, 1])
        angles.append(jitter)

    return angles


def _downsample_ct_for_drr(ct: np.ndarray, max_dim: int = 256) -> np.ndarray:
    """Downsample CT volume for DRR generation if larger than max_dim.

    Full-res CT volumes (300-500³) are extremely slow to rotate. Since DRRs
    are projected to 224×224 anyway, we don't need full resolution for the
    attenuation volume. Downsampling to ~256³ loses negligible DRR quality
    but makes rotation 4-8x faster.
    """
    largest = max(ct.shape)
    if largest <= max_dim:
        return ct
    factor = max_dim / largest
    return zoom(ct, [factor] * 3, order=1).astype(ct.dtype)


def generate_sample(ct_path: Path, mask_path: Path, subject_id: str,
                    output_dir: Path, force: bool, num_angles: int,
                    bone_label: int = BONE_LABEL) -> list[dict]:
    """Generate training data for one subject at clinically realistic angles.

    DRRs are generated from the full CT volume (with soft tissue) for realistic
    X-ray images. The 64³ bone voxel grid is the model's training target.
    """
    # Per-subject deterministic RNG for reproducible jitter
    seed = int.from_bytes(subject_id.encode()[:4], "little") & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    angles = make_augmentation_angles(num_angles, rng)

    # Check if all angles already generated
    all_exist = True
    for angle in angles:
        sid = f"{subject_id}_a{int(angle):03d}" if angle != 0 else subject_id
        if not (output_dir / sid / "voxels.npy").exists():
            all_exist = False
            break

    if all_exist and not force:
        results = []
        for angle in angles:
            sid = f"{subject_id}_a{int(angle):03d}" if angle != 0 else subject_id
            bone_ratio = float(np.load(output_dir / sid / "voxels.npy").mean())
            results.append({"id": sid, "subject": subject_id, "source": "cads",
                            "body_part": "full_body", "bone_ratio": bone_ratio, "angle": angle})
        return results

    # Load CT volume (for DRRs) + bone mask (for voxel target)
    loaded = load_ct_and_mask(ct_path, mask_path, bone_label=bone_label)
    if loaded is None:
        print(f"    Skipped (no bone): {subject_id}")
        return []

    ct, voxels = loaded

    # Downsample CT for faster rotation — DRRs project to 224² anyway
    ct = _downsample_ct_for_drr(ct, max_dim=256)

    # Generate samples at each angle
    results = []
    for angle in angles:
        sid = f"{subject_id}_a{int(angle):03d}" if angle != 0 else subject_id
        sample_dir = output_dir / sid

        if (sample_dir / "voxels.npy").exists() and not force:
            bone_ratio = float(np.load(sample_dir / "voxels.npy").mean())
            results.append({"id": sid, "subject": subject_id, "source": "cads",
                            "body_part": "full_body", "bone_ratio": bone_ratio, "angle": angle})
            continue

        # Rotate CT (linear interp) for DRRs, bone voxels (nearest) for target
        rotated_ct = rotate_volume(ct, angle, order=1)
        rotated_voxels = rotate_volume(voxels, angle, order=0)

        # DRRs from full CT volume (realistic X-rays with soft tissue)
        # After transpose, dims are (Z, Y, X) where Z=vertical, Y=front-back, X=left-right
        # AP: beam goes front-to-back (sum along Y=axis 1) → frontal view (Z × X)
        # LAT: beam goes left-to-right (sum along X=axis 2) → side view (Z × Y)
        ap = generate_drr(rotated_ct, axis=1)
        lat = generate_drr(rotated_ct, axis=2)

        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "voxels.npy", rotated_voxels)
        np.save(sample_dir / "ap.npy", ap)
        np.save(sample_dir / "lat.npy", lat)

        # Free rotated arrays immediately
        del rotated_ct
        del rotated_voxels

        bone_ratio = float(np.load(sample_dir / "voxels.npy").mean())
        results.append({"id": sid, "subject": subject_id, "source": "cads",
                        "body_part": "full_body", "bone_ratio": bone_ratio, "angle": angle})

    # Free the CT volume after all angles are done
    del ct, voxels

    n_angles = len(results)
    bone_ratio = results[0]["bone_ratio"] if results else 0
    print(f"    Generated: {subject_id} x {n_angles} angles (bone ratio: {bone_ratio:.3f})")
    return results


# ─── CADS pair finding ──────────────────────────────────────────────────

def find_cads_pairs(cads_dir: Path, subset: str,
                    max_volumes: int = 0) -> list[tuple[Path, Path, str]]:
    """Find (image, bone_mask) pairs in downloaded CADS data.

    Expected structure:
        {subset}/images/s####_0000.nii.gz
        {subset}/segmentations/s####/s####_part_559.nii.gz
    """
    images_dir = cads_dir / subset / "images"
    segs_dir = cads_dir / subset / "segmentations"

    if not images_dir.exists():
        print(f"  Warning: {images_dir} not found")
        return []

    pairs = []
    for img_path in sorted(images_dir.glob("*_0000.nii.gz")):
        subject_id = img_path.name.replace("_0000.nii.gz", "")
        mask_path = segs_dir / subject_id / f"{subject_id}_{BONE_PART}.nii.gz"
        if mask_path.exists():
            pairs.append((img_path, mask_path, f"cads_{subject_id}"))

    if max_volumes > 0:
        pairs = pairs[:max_volumes]

    return pairs


# ─── Parallel generation ────────────────────────────────────────────────

def _gen_worker(args_tuple):
    """Worker for parallel generation."""
    ct_path, mask_path, subject_id, output_dir, force, num_angles = args_tuple
    try:
        return generate_sample(ct_path, mask_path, subject_id, output_dir, force, num_angles)
    except Exception as e:
        import traceback
        print(f"    WORKER CRASH {subject_id}: {e}", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        return []


def parallel_generate(work_items, gen_workers):
    """Run data generation in parallel."""
    if gen_workers <= 1:
        results = []
        for item in work_items:
            results.extend(_gen_worker(item))
        return results

    results = []
    total = len(work_items)
    with ProcessPoolExecutor(max_workers=gen_workers, max_tasks_per_child=5) as pool:
        futures = {pool.submit(_gen_worker, item): i for i, item in enumerate(work_items)}
        done = 0
        failed = 0
        for future in as_completed(futures):
            try:
                result = future.result(timeout=600)
                results.extend(result)
            except Exception as e:
                idx = futures[future]
                print(f"  Worker {idx} failed: {e}", flush=True)
                failed += 1
            done += 1
            if done % 5 == 0 or done == total:
                print(f"  Progress: {done}/{total} volumes processed ({failed} failed)", flush=True)
                sys.stdout.flush()
    return results


# ─── Manifest ────────────────────────────────────────────────────────────

def write_manifest(samples: list[dict], output_dir: Path, val_ratio: float):
    """Write train/val manifest, splitting by subject to avoid data leakage."""
    np.random.seed(42)

    subjects = sorted(set(s["subject"] for s in samples))
    n_val = max(1, int(len(subjects) * val_ratio))
    perm = np.random.permutation(len(subjects))
    val_subjects = set(subjects[i] for i in perm[:n_val])

    train_ids = [s["id"] for s in samples if s["subject"] not in val_subjects]
    val_ids = [s["id"] for s in samples if s["subject"] in val_subjects]

    manifest = {
        "voxel_resolution": VOXEL_RES,
        "xray_resolution": XRAY_RES,
        "train": train_ids,
        "val": val_ids,
        "samples": {s["id"]: s for s in samples},
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return len(train_ids), len(val_ids)


# ─── Progress tracking ─────────────────────────────────────────────────

def load_progress(output_dir: Path) -> dict:
    """Load progress file tracking completed subjects per subset."""
    path = output_dir / PROGRESS_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {"completed": {}}


def save_progress(output_dir: Path, progress: dict):
    """Save progress file atomically."""
    path = output_dir / PROGRESS_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.rename(path)


def mark_subjects_done(output_dir: Path, progress: dict,
                       subset: str, subject_ids: list[str]):
    """Mark subjects as completed in the progress file."""
    if subset not in progress["completed"]:
        progress["completed"][subset] = []
    existing = set(progress["completed"][subset])
    for sid in subject_ids:
        existing.add(sid)
    progress["completed"][subset] = sorted(existing)
    save_progress(output_dir, progress)


def get_remaining_subjects(progress: dict, subset: str,
                           all_subjects: list[str]) -> list[str]:
    """Filter out already-completed subjects for a subset."""
    done = set(progress.get("completed", {}).get(subset, []))
    return [s for s in all_subjects if s not in done]


# ─── Batch download + process pipeline ─────────────────────────────────

def download_and_process_batch(subject_ids: list[str], cads_dir: Path,
                               output_dir: Path, subset: str,
                               num_angles: int,
                               gen_workers: int, force: bool,
                               progress: dict | None = None) -> list[dict]:
    """Download a batch of subjects, process them, and clean up raw data."""
    from download_cads import download_batch

    print(f"\n  Downloading {len(subject_ids)} subjects...")
    download_batch(subject_ids, cads_dir, subset, workers=4)

    # Find pairs for this batch
    pairs = find_cads_pairs(cads_dir, subset=subset, max_volumes=0)
    # Filter to only this batch
    batch_set = set(f"cads_{sid}" for sid in subject_ids)
    pairs = [(ct, mask, sid) for ct, mask, sid in pairs if sid in batch_set]

    if not pairs:
        print(f"  No valid pairs found in this batch")
        return []

    print(f"  Processing {len(pairs)} subjects...")
    work_items = [
        (ct_path, mask_path, subject_id, output_dir, force, num_angles)
        for ct_path, mask_path, subject_id in pairs
    ]
    results = parallel_generate(work_items, gen_workers)

    # Mark successfully processed subjects in progress file
    if progress is not None:
        done_ids = [sid for sid in subject_ids
                    if any(r["subject"] == f"cads_{sid}" or r["subject"] == sid
                           for r in results)]
        # Also mark subjects that were skipped (no bone) — they're still "done"
        done_ids = subject_ids  # all attempted subjects are done
        mark_subjects_done(output_dir, progress, subset, done_ids)

    # Clean up raw downloads for this batch
    subset_dir = cads_dir / subset
    for sid in subject_ids:
        img = subset_dir / "images" / f"{sid}_0000.nii.gz"
        seg_dir = subset_dir / "segmentations" / sid
        if img.exists():
            img.unlink()
        if seg_dir.exists():
            shutil.rmtree(seg_dir)

    return results


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and generate training data from CADS dataset")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for training data output")
    parser.add_argument("--cads-dir", type=str, default=None,
                        help="Directory for CADS raw downloads (temp, cleaned up per batch)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, use already-downloaded data in --cads-dir")
    parser.add_argument("--max-subjects", type=int, default=250,
                        help="Max subjects to process (default: 250, 0=all)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Subjects per download batch (default: 50)")
    parser.add_argument("--num-angles", type=int, default=8,
                        help="Angle variations per subject (default: 8). "
                             "Each produces both AP + LAT views. Uses small jitter for augmentation.")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--gen-workers", type=int, default=4,
                        help="Parallel generation workers (default: 4)")
    parser.add_argument("--dl-workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: download ALL subjects first, then process ALL in parallel. "
                             "Needs more disk but much faster on high-CPU machines.")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="CADS subsets to use. 'all' discovers all subsets with bone data. "
                             "Default: 3 known bone subsets.")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if output already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "training_data"
    cads_dir = Path(args.cads_dir) if args.cads_dir else output_dir.parent / "cads_raw"

    output_dir.mkdir(parents=True, exist_ok=True)
    cads_dir.mkdir(parents=True, exist_ok=True)

    num_angles = args.num_angles

    if args.subsets and args.subsets[0] == "all":
        from download_cads import list_subsets_with_bone
        print("Discovering all CADS subsets with bone segmentation...")
        subsets = list_subsets_with_bone()
        np.random.seed(None)  # truly random
        np.random.shuffle(subsets)
        print(f"Found {len(subsets)} subsets with bone data (randomized order)\n")
    elif args.subsets:
        subsets = args.subsets
    else:
        subsets = CADS_SUBSETS

    print(f"CADS Training Data Pipeline")
    print(f"  Dataset: {CADS_REPO}")
    print(f"  Subsets: {', '.join(subsets)}")
    print(f"  Bone mask: {BONE_PART} label {BONE_LABEL}")
    print(f"  DRR source: CT volume (realistic X-rays with soft tissue)")
    print(f"  Max subjects per subset: {args.max_subjects}")
    print(f"  Angle variations: {num_angles} (each → AP + LAT projections)")
    print(f"  Output: {output_dir}")
    print(f"  Expected samples: ~{args.max_subjects * num_angles * len(subsets)}")

    all_samples = []

    if args.skip_download:
        # Use pre-downloaded data
        print(f"\nUsing pre-downloaded data in {cads_dir}")
        for subset in subsets:
            pairs = find_cads_pairs(cads_dir, subset=subset, max_volumes=args.max_subjects)
            if not pairs:
                print(f"  No pairs found for {subset}, skipping")
                continue

            print(f"  [{subset}] Found {len(pairs)} volume/mask pairs")
            work_items = [
                (ct_path, mask_path, subject_id, output_dir, args.force, num_angles)
                for ct_path, mask_path, subject_id in pairs
            ]
            subset_samples = parallel_generate(work_items, args.gen_workers)
            all_samples.extend(subset_samples)
            print(f"  [{subset}] Generated {len(subset_samples)} samples")

        if not all_samples:
            print("No CADS pairs found. Run without --skip-download first.")
            sys.exit(1)

    elif args.fast:
        # Fast mode: download ALL subjects across all subsets, then process ALL
        # Needs more disk space but maximizes parallelism on high-CPU machines
        import time as _time
        from download_cads import list_subjects, download_batch

        dl_workers = args.dl_workers

        # Phase 1: Download everything
        print(f"\n{'='*50}")
        print(f"FAST MODE: Phase 1 — Download all subjects")
        print(f"  Download workers: {dl_workers}")
        print(f"{'='*50}")

        t0 = _time.time()
        all_subject_lists = {}
        for subset in subsets:
            print(f"\n  [{subset}] Listing subjects...")
            subjects = list_subjects(subset)
            if args.max_subjects > 0:
                subjects = subjects[:args.max_subjects]
            all_subject_lists[subset] = subjects
            print(f"  [{subset}] Downloading {len(subjects)} subjects...")
            download_batch(subjects, cads_dir, subset, workers=dl_workers)

        dl_time = _time.time() - t0
        print(f"\n  Download complete in {dl_time:.0f}s")

        # Phase 2: Process everything with max parallelism
        print(f"\n{'='*50}")
        print(f"FAST MODE: Phase 2 — Process all subjects")
        print(f"  Processing workers: {args.gen_workers}")
        print(f"{'='*50}")

        t1 = _time.time()
        for subset in subsets:
            subjects = all_subject_lists[subset]
            pairs = find_cads_pairs(cads_dir, subset=subset, max_volumes=0)
            # Filter to requested subjects
            batch_set = set(f"cads_{sid}" for sid in subjects)
            pairs = [(ct, mask, sid) for ct, mask, sid in pairs if sid in batch_set]

            if not pairs:
                print(f"  [{subset}] No valid pairs found, skipping")
                continue

            print(f"\n  [{subset}] Processing {len(pairs)} subjects...")
            work_items = [
                (ct_path, mask_path, subject_id, output_dir, args.force, num_angles)
                for ct_path, mask_path, subject_id in pairs
            ]
            subset_samples = parallel_generate(work_items, args.gen_workers)
            all_samples.extend(subset_samples)
            print(f"  [{subset}] Done: {len(subset_samples)} samples")

        proc_time = _time.time() - t1
        print(f"\n  Processing complete in {proc_time:.0f}s")
        print(f"  Total time: {dl_time + proc_time:.0f}s "
              f"(download: {dl_time:.0f}s, process: {proc_time:.0f}s)")

        # Clean up raw data
        print(f"\n  Cleaning up raw downloads...")
        shutil.rmtree(cads_dir, ignore_errors=True)

    else:
        # Batch download + process (low disk usage mode)
        from download_cads import list_subjects

        progress = load_progress(output_dir)
        total_skipped = 0

        for subset in subsets:
            print(f"\n{'#'*50}")
            print(f"Subset: {subset}")
            print(f"{'#'*50}")

            print(f"Listing available subjects...")
            subjects = list_subjects(subset)
            print(f"Found {len(subjects)} subjects on HuggingFace")

            if not subjects:
                print(f"  Skipping {subset} (no subjects found)")
                continue

            if args.max_subjects > 0:
                subjects = subjects[:args.max_subjects]

            # Filter out already-completed subjects
            remaining = get_remaining_subjects(progress, subset, subjects)
            skipped = len(subjects) - len(remaining)
            total_skipped += skipped
            if skipped > 0:
                print(f"  Skipping {skipped} already-completed subjects")
            if not remaining:
                print(f"  All {len(subjects)} subjects already done, skipping subset")
                continue

            print(f"Will process {len(remaining)} subjects in batches of {args.batch_size}\n")

            subset_start = len(all_samples)
            for i in range(0, len(remaining), args.batch_size):
                batch = remaining[i:i + args.batch_size]
                batch_num = i // args.batch_size + 1
                total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
                print(f"{'='*50}")
                print(f"[{subset}] Batch {batch_num}/{total_batches}: subjects {i+1}-{i+len(batch)}")
                print(f"{'='*50}")

                batch_samples = download_and_process_batch(
                    batch, cads_dir, output_dir, subset,
                    num_angles, args.gen_workers, args.force,
                    progress=progress)
                all_samples.extend(batch_samples)

                print(f"  Batch {batch_num} done: {len(batch_samples)} samples "
                      f"(total so far: {len(all_samples)})")

            subset_count = len(all_samples) - subset_start
            print(f"\n  [{subset}] Done: {subset_count} samples")

        if total_skipped > 0:
            print(f"\nResumed: skipped {total_skipped} previously completed subjects")

    # Collect any existing samples from previous runs (for complete manifest)
    existing_dirs = [d for d in output_dir.iterdir()
                     if d.is_dir() and (d / "voxels.npy").exists()]
    existing_ids = set(s["id"] for s in all_samples)
    for d in existing_dirs:
        if d.name not in existing_ids:
            bone_ratio = float(np.load(d / "voxels.npy").mean())
            # Parse subject from sample ID (strip _aXXX suffix if present)
            subject = d.name.rsplit("_a", 1)[0] if "_a" in d.name else d.name
            all_samples.append({
                "id": d.name, "subject": subject, "source": "cads",
                "body_part": "full_body", "bone_ratio": bone_ratio, "angle": 0,
            })

    if not all_samples:
        print("\nNo training samples generated.")
        sys.exit(1)

    n_train, n_val = write_manifest(all_samples, output_dir, args.val_ratio)

    print(f"\n{'='*50}")
    print(f"Done! {len(all_samples)} total samples ({n_train} train, {n_val} val)")
    print(f"  Subjects processed: {len(set(s['subject'] for s in all_samples))}")
    print(f"  Subsets: {', '.join(subsets)}")
    print(f"  Angle variations: {num_angles}")
    avg_bone = np.mean([s["bone_ratio"] for s in all_samples])
    print(f"  Average bone ratio: {avg_bone:.4f}")
    print(f"Training data: {output_dir}")
    print(f"\nNext step — train the model:")
    print(f"  python -m model.train --data {output_dir}")


if __name__ == "__main__":
    main()
