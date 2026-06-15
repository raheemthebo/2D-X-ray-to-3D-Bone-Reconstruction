#!/bin/bash
# Sync model outputs from the SLURM cluster to local machine.
#
# Usage:
#   bash scripts/sync_cluster.sh              # Sync everything
#   bash scripts/sync_cluster.sh --best-only  # Best-epoch val samples + best.pt only
#   bash scripts/sync_cluster.sh --dry-run    # Preview what would be synced
#
# What it syncs:
#   - model/checkpoints/best.pt         Best model checkpoint
#   - model/checkpoints/latest.pt       Latest checkpoint (for resume)
#   - model/logs/*.json, *.out          Training & SLURM logs
#   - model/logs/val_samples/           Validation visualisations (PNG + GLB)
#   - val_samples/                      25 random validation data samples (ap.npy, lat.npy, voxels.npy)

set -e

REMOTE="mlp"
REMOTE_HOME="Expo-AI"
REMOTE_BACKUP="${REMOTE_HOME}/model/backups"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DRY_RUN=""
BEST_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN="--dry-run" ;;
        --best-only) BEST_ONLY=true ;;
    esac
done

[ -n "$DRY_RUN" ] && echo "[DRY RUN] Previewing what would be synced..." && echo ""

# ---- Best-only mode: fetch best-epoch samples + best.pt ----
if $BEST_ONLY; then
    echo "Finding best epoch from checkpoint..."
    BEST_EPOCH=$(ssh "${REMOTE}" "source ${REMOTE_HOME}/.venv/bin/activate && python3 -c \"
import torch
ckpt = torch.load('${REMOTE_BACKUP}/best.pt', map_location='cpu', weights_only=False)
print(ckpt.get('epoch', 0))
\"")
    echo "Best epoch: ${BEST_EPOCH}"
    PREFIX=$(printf "e%03d_" "${BEST_EPOCH}")

    echo ""
    echo "=== Best-epoch validation samples ==="
    mkdir -p "${LOCAL_DIR}/val_samples"
    rsync -av --progress ${DRY_RUN} \
        --include="${PREFIX}*" --exclude="*" \
        "${REMOTE}:${REMOTE_BACKUP}/val_samples/" \
        "${LOCAL_DIR}/val_samples/"

    echo ""
    echo "=== Best checkpoint ==="
    mkdir -p "${LOCAL_DIR}/model/checkpoints"
    rsync -av --progress ${DRY_RUN} \
        "${REMOTE}:${REMOTE_BACKUP}/best.pt" \
        "${LOCAL_DIR}/model/checkpoints/best.pt"

    echo ""
    echo "=== Metrics ==="
    mkdir -p "${LOCAL_DIR}/model/logs"
    rsync -avz --progress ${DRY_RUN} \
        --include="*.json" --exclude="*" \
        "${REMOTE}:${REMOTE_HOME}/model/logs/" \
        "${LOCAL_DIR}/model/logs/" 2>/dev/null || true
    rsync -avz --progress ${DRY_RUN} \
        --include="*.json" --exclude="*" \
        "${REMOTE}:${REMOTE_BACKUP}/logs/" \
        "${LOCAL_DIR}/model/logs/" 2>/dev/null || true

    echo ""
    echo "Done! Best-epoch files saved."
    exit 0
fi

# ---- Full sync mode ----
echo "Syncing from cluster → ${LOCAL_DIR}"
echo ""

echo "=== Checkpoints ==="
mkdir -p "${LOCAL_DIR}/model/checkpoints"
rsync -avz --progress ${DRY_RUN} \
    "${REMOTE}:${REMOTE_HOME}/model/checkpoints/best.pt" \
    "${REMOTE}:${REMOTE_HOME}/model/checkpoints/latest.pt" \
    "${LOCAL_DIR}/model/checkpoints/" 2>/dev/null || true
# Also try from backup dir
rsync -avz --progress ${DRY_RUN} \
    "${REMOTE}:${REMOTE_BACKUP}/best.pt" \
    "${REMOTE}:${REMOTE_BACKUP}/latest.pt" \
    "${LOCAL_DIR}/model/checkpoints/" 2>/dev/null || true
echo ""

echo "=== Training logs & metrics ==="
mkdir -p "${LOCAL_DIR}/model/logs"
rsync -avz --progress ${DRY_RUN} \
    --include="*.json" --include="*.out" --exclude="*" \
    "${REMOTE}:${REMOTE_HOME}/model/logs/" \
    "${LOCAL_DIR}/model/logs/" 2>/dev/null || true
rsync -avz --progress ${DRY_RUN} \
    --include="*.json" --exclude="*" \
    "${REMOTE}:${REMOTE_BACKUP}/logs/" \
    "${LOCAL_DIR}/model/logs/" 2>/dev/null || true
echo ""

echo "=== Validation samples (model outputs) ==="
rsync -avz --progress ${DRY_RUN} \
    "${REMOTE}:${REMOTE_HOME}/model/logs/val_samples/" \
    "${LOCAL_DIR}/model/logs/val_samples/" 2>/dev/null || true
rsync -avz --progress ${DRY_RUN} \
    "${REMOTE}:${REMOTE_BACKUP}/val_samples/" \
    "${LOCAL_DIR}/model/logs/val_samples/" 2>/dev/null || true
echo ""

echo "=== Validation data (diverse samples backed up during training) ==="
# Training job backs up 25 diverse samples (round-robin across subsets) to ~/backups/val_data/
# Clear local dir and rsync from there
if [ -z "$DRY_RUN" ]; then
    rm -rf "${LOCAL_DIR}/val_samples"
else
    echo "  [DRY RUN] Would clear ${LOCAL_DIR}/val_samples/"
fi
mkdir -p "${LOCAL_DIR}/val_samples"
rsync -avz --progress ${DRY_RUN} \
    "${REMOTE}:${REMOTE_BACKUP}/val_data/" \
    "${LOCAL_DIR}/val_samples/" 2>/dev/null || echo "  No val_data in backups yet (run a training job first)."
echo ""

echo "=== Done ==="
echo ""
echo "Checkpoints:  ${LOCAL_DIR}/model/checkpoints/"
echo "Logs:         ${LOCAL_DIR}/model/logs/"
echo "Val outputs:  ${LOCAL_DIR}/model/logs/val_samples/"
echo "Val data:     ${LOCAL_DIR}/val_samples/"
echo ""
echo "Run inference:"
echo "  uv run python -m model.inference --ap test_images/xray_ap.png --lat test_images/xray_lat.png -o output/"
