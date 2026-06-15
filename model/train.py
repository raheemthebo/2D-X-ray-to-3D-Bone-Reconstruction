"""
Train the X2BR-inspired X-ray to 3D bone reconstruction model.

Training uses point-sampled occupancy prediction (following X2BR):
- 2048 random 3D points sampled per iteration (50% near-surface, 50% uniform)
- Ground-truth occupancy looked up from the 64^3 voxel grid
- BCE loss with balanced sampling
- Validation evaluates dense Dice on the full 64^3 voxel grid
- Validation samples saved as PNG (X-rays) + GLB (3D mesh) every N epochs
- Checkpoints backed up to --backup-dir continuously

Usage:
    python -m model.train                                  # Train with defaults
    python -m model.train --epochs 200 --batch-size 4      # Custom settings
    python -m model.train --device cuda                    # Force device
    python -m model.train --resume model/checkpoints/best.pt  # Resume training
    python -m model.train --backup-dir /home/user/backups  # Continuous backup
"""

import argparse
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architecture import X2BRBiplanarModel
from .dataset import XrayVoxelDataset, worker_init_fn
from .losses import OccupancyBCELoss

TRAINING_DATA_DIR = Path(__file__).parent.parent / "data_factory" / "training_data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
LOG_DIR = Path(__file__).parent / "logs"

NUM_QUERY_POINTS = 2048  # points sampled per sample per iteration (X2BR uses 2048)
VAL_SAMPLE_INTERVAL = 10  # save validation samples every N epochs


def seed_everything(seed: int):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster for fixed input sizes (224x224)


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_query_points(voxels: torch.Tensor, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample balanced 3D query points — 50% near bone, 50% uniform random.

    This addresses the severe class imbalance (bone is ~3-5% of volume).
    Half the points are sampled uniformly, half near occupied voxels with jitter.
    """
    B, _, D, H, W = voxels.shape
    device = voxels.device
    scale = torch.tensor([D - 1, H - 1, W - 1], device=device, dtype=torch.float32)

    n_uniform = num_points // 2
    n_near_surface = num_points - n_uniform

    all_points = []
    all_targets = []

    for b in range(B):
        vol = voxels[b, 0]  # (D, H, W)

        # Uniform random points
        uniform_pts = torch.rand(n_uniform, 3, device=device)

        # Near-surface points: sample from occupied voxels + add jitter
        occupied = torch.nonzero(vol, as_tuple=False).float()  # (K, 3)
        if occupied.shape[0] > 0:
            idx = torch.randint(0, occupied.shape[0], (n_near_surface,), device=device)
            near_pts = occupied[idx] / scale
            jitter = (torch.rand(n_near_surface, 3, device=device) - 0.5) * (4.0 / scale)
            near_pts = (near_pts + jitter).clamp(0, 1)
        else:
            near_pts = torch.rand(n_near_surface, 3, device=device)

        pts = torch.cat([uniform_pts, near_pts], dim=0)

        # Look up occupancy
        grid_idx = (pts * scale).long()
        grid_idx[:, 0] = grid_idx[:, 0].clamp(0, D - 1)
        grid_idx[:, 1] = grid_idx[:, 1].clamp(0, H - 1)
        grid_idx[:, 2] = grid_idx[:, 2].clamp(0, W - 1)
        tgt = vol[grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]]

        all_points.append(pts)
        all_targets.append(tgt)

    points = torch.stack(all_points, dim=0)
    targets = torch.stack(all_targets, dim=0).unsqueeze(-1)
    return points, targets


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice score between prediction and target voxel grids."""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection / total).item()


def train_one_epoch(model, loader, criterion, optimizer, device, num_points, epoch, total_epochs,
                    scaler=None):
    model.train()
    total_loss = 0
    n = 0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for ap, lat, voxels in pbar:
        ap, lat, voxels = ap.to(device), lat.to(device), voxels.to(device)

        query_points, targets = sample_query_points(voxels, num_points)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(ap, lat, query_points)
            loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}")

    return total_loss / n


@torch.no_grad()
def validate(model, loader, device, use_amp=False):
    """Validate by computing dense voxel Dice (full 64^3 grid evaluation)."""
    model.eval()
    total_dice = 0
    n = 0

    for ap, lat, voxels in loader:
        ap, lat, voxels = ap.to(device), lat.to(device), voxels.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(ap, lat)
        total_dice += dice_score(pred, voxels)
        n += 1

    return total_dice / n


def voxels_to_glb(voxels: np.ndarray, output_path: Path, threshold: float = 0.5):
    """Convert voxel grid to GLB mesh file."""
    import trimesh
    from skimage.measure import marching_cubes

    binary = (voxels > threshold).astype(np.float32)
    if binary.sum() < 10:
        return False

    try:
        verts, faces, _, _ = marching_cubes(binary, level=0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        trimesh.smoothing.filter_laplacian(mesh, iterations=3)
        mesh.export(str(output_path))
        return True
    except Exception:
        return False


@torch.no_grad()
def save_val_samples(model, dataset, device, output_dir: Path, epoch: int, n_samples: int = 5):
    """Save validation samples: PNG X-rays + GLB 3D meshes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = min(n_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    for idx in indices:
        ap, lat, voxels = dataset[idx]
        ap_in = ap.unsqueeze(0).to(device)
        lat_in = lat.unsqueeze(0).to(device)

        pred = model(ap_in, lat_in)
        pred = pred.cpu().squeeze().numpy()
        pred_bin = (pred > 0.5).astype(np.float32)

        gt = voxels.squeeze().numpy()
        ap_img = ap.squeeze().numpy()
        lat_img = lat.squeeze().numpy()

        d = dice_score(torch.from_numpy(pred_bin).unsqueeze(0),
                       torch.from_numpy(gt).unsqueeze(0))

        sample_id = dataset.sample_ids[idx]
        prefix = f"e{epoch:03d}_{sample_id}"

        # Save raw input X-rays as clean PNGs (no annotations, suitable for inference)
        from PIL import Image
        Image.fromarray((ap_img * 255).clip(0, 255).astype(np.uint8)).save(
            output_dir / f"{prefix}_ap.png")
        Image.fromarray((lat_img * 255).clip(0, 255).astype(np.uint8)).save(
            output_dir / f"{prefix}_lat.png")

        # Save predicted 3D mesh as GLB
        glb_saved = voxels_to_glb(pred, output_dir / f"{prefix}_pred.glb")

        # Save ground truth 3D mesh as GLB
        voxels_to_glb(gt, output_dir / f"{prefix}_gt.glb")

        # Save comparison figure (overview)
        mid = pred.shape[0] // 2
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes[0, 0].imshow(ap_img, cmap="bone")
        axes[0, 0].set_title("AP Input")
        axes[0, 1].imshow(lat_img, cmap="bone")
        axes[0, 1].set_title("LAT Input")
        axes[0, 2].imshow(gt.max(axis=2), cmap="gray")
        axes[0, 2].set_title("GT (AP proj)")
        axes[0, 3].imshow(gt[mid], cmap="gray")
        axes[0, 3].set_title(f"GT (slice {mid})")
        axes[1, 0].imshow(pred.max(axis=2), cmap="hot")
        axes[1, 0].set_title("Pred (AP proj)")
        axes[1, 1].imshow(pred[mid], cmap="hot")
        axes[1, 1].set_title(f"Pred (slice {mid})")
        axes[1, 2].imshow(pred_bin.max(axis=2), cmap="gray")
        axes[1, 2].set_title("Pred binary (AP)")
        axes[1, 3].imshow(pred_bin[mid], cmap="gray")
        axes[1, 3].set_title(f"Pred binary (slice {mid})")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"Epoch {epoch} | {sample_id} | Dice: {d:.4f}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_overview.png", dpi=150)
        plt.close(fig)

        glb_status = "GLB saved" if glb_saved else "GLB failed"
        print(f"  Val sample: {sample_id} (Dice: {d:.4f}, {glb_status})")


def backup_checkpoint(backup_dir: Path):
    """Backup checkpoints and logs to a safe directory (e.g., home disk)."""
    if not backup_dir:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup latest checkpoint
    for name in ["best.pt", "latest.pt"]:
        src = CHECKPOINT_DIR / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)

    # Backup logs
    log_backup = backup_dir / "logs"
    log_backup.mkdir(exist_ok=True)
    for log_file in LOG_DIR.glob("*"):
        if log_file.is_file():
            shutil.copy2(log_file, log_backup / log_file.name)

    # Backup val_samples
    src_samples = LOG_DIR / "val_samples"
    if src_samples.exists():
        dst = backup_dir / "val_samples"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_samples, dst)


def main():
    parser = argparse.ArgumentParser(description="Train X2BR-inspired X-ray to 3D model")
    parser.add_argument("--data", default=str(TRAINING_DATA_DIR), help="Training data directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (X2BR: 1e-4)")
    parser.add_argument("--device", type=str, help="Force device (cuda/mps/cpu)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-points", type=int, default=NUM_QUERY_POINTS,
                        help=f"Query points per sample (default: {NUM_QUERY_POINTS})")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Directory to continuously backup checkpoints/samples to (e.g., home disk)")
    parser.add_argument("--val-sample-interval", type=int, default=VAL_SAMPLE_INTERVAL,
                        help=f"Save validation samples every N epochs (default: {VAL_SAMPLE_INTERVAL})")
    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"Seed: {args.seed}")

    device = get_device(args.device)
    print(f"Device: {device}")

    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    if backup_dir:
        print(f"Backup dir: {backup_dir}")

    # Data
    train_ds = XrayVoxelDataset(args.data, split="train", augment=True)
    val_ds = XrayVoxelDataset(args.data, split="val", augment=False)
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val samples")

    g = torch.Generator()
    g.manual_seed(args.seed)
    num_workers = min(8, args.batch_size * 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, generator=g, worker_init_fn=worker_init_fn,
                              pin_memory=True, persistent_workers=num_workers > 0,
                              prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0,
                            prefetch_factor=2)

    # Model
    model = X2BRBiplanarModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: X2BRBiplanarModel — {params:,} parameters ({params/1e6:.1f}M)")

    # Loss
    criterion = OccupancyBCELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 1
    best_val_dice = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_dice = ckpt.get("best_val_dice", 0)
            print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, best dice {best_val_dice:.4f})")
        else:
            model.load_state_dict(ckpt)
            print(f"Resumed model weights from {args.resume} (no optimizer state)")

    # Mixed precision training (AMP) — ~2x speedup on modern GPUs
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision: enabled (AMP)")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Validation samples directory
    val_samples_dir = LOG_DIR / "val_samples"
    val_samples_dir.mkdir(parents=True, exist_ok=True)

    # Training log
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{run_id}.json"
    run_log = {
        "run_id": run_id,
        "architecture": "X2BRBiplanarModel",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "parameters": params,
            "num_query_points": args.num_points,
            "resumed_from": args.resume,
            "start_epoch": start_epoch,
            "loss": "BCE (balanced sampling)",
            "optimizer": "AdamW (wd=1e-4)",
            "scheduler": "CosineAnnealingLR (eta_min=1e-6)",
            "sampling": "50% uniform + 50% near-surface",
            "voxel_resolution": 64,
            "xray_resolution": 224,
        },
        "epochs": [],
    }

    start_time = time.time()

    print(f"\nTraining for epochs {start_epoch}-{args.epochs} ({args.num_points} query points/sample)...")
    print(f"Val samples every {args.val_sample_interval} epochs\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Dice':>9} {'LR':>10} {'Time':>6}")
    print("-" * 52)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, args.num_points, epoch, args.epochs,
                                     scaler=scaler)
        val_dice = validate(model, val_loader, device, use_amp=use_amp)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        # Build checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "val_dice": val_dice,
            "train_loss": train_loss,
            "seed": args.seed,
            "architecture": "X2BRBiplanarModel",
        }

        # Always save latest checkpoint (for resume on interruption)
        torch.save(checkpoint, CHECKPOINT_DIR / "latest.pt")

        # Save best model
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            checkpoint["best_val_dice"] = best_val_dice
            torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")
            marker = " *"
        else:
            marker = ""

        # Log — write after every epoch so it's always up to date
        run_log["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_dice": round(val_dice, 6),
            "lr": round(lr, 8),
            "elapsed_s": round(elapsed, 1),
            "is_best": is_best,
        })
        run_log["summary"] = {
            "best_val_dice": round(best_val_dice, 6),
            "total_time_s": round(elapsed, 1),
            "final_train_loss": round(train_loss, 6),
            "current_epoch": epoch,
        }
        log_path.write_text(json.dumps(run_log, indent=2))

        print(f"{epoch:6d} {train_loss:11.4f} {val_dice:9.4f} {lr:10.6f} {elapsed:5.0f}s{marker}")

        # Save validation samples periodically
        if epoch % args.val_sample_interval == 0 or epoch == args.epochs or is_best:
            print(f"\n  Saving validation samples (epoch {epoch})...")
            save_val_samples(model, val_ds, device, val_samples_dir, epoch)
            print()

        # Continuous backup to safe storage (every 5 epochs, on best, and at end)
        if backup_dir and (epoch % 5 == 0 or is_best or epoch == args.epochs):
            backup_checkpoint(backup_dir)

    # Save final model
    torch.save(checkpoint, CHECKPOINT_DIR / "final.pt")

    # Final backup
    if backup_dir:
        backup_checkpoint(backup_dir)

    print(f"\nDone! Best val Dice: {best_val_dice:.4f}")
    print(f"Checkpoints: {CHECKPOINT_DIR}/best.pt, {CHECKPOINT_DIR}/latest.pt")
    print(f"Training log: {log_path}")
    print(f"Val samples: {val_samples_dir}/")


if __name__ == "__main__":
    main()
