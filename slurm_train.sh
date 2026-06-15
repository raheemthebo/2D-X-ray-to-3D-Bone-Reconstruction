#!/bin/bash
#SBATCH --job-name=xray3d_train
#SBATCH --output=model/logs/slurm_%j.out
#SBATCH --error=model/logs/slurm_%j.out
#SBATCH --account=general-teaching
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1g.16gb:1
#SBATCH --nodelist=saxa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s2891607@ed.ac.uk
#SBATCH --exclude=landonia01

set -e
export PYTHONUNBUFFERED=1

# ---- Load .env (HF_TOKEN, etc.) ----
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env (HF_TOKEN=${HF_TOKEN:+set})"
fi

# ---- Paths ----
HOME_DIR="/home/s2891607/Expo-AI"
SCRATCH_DIR="/disk/scratch/s2891607"
BACKUP_DIR="${HOME_DIR}/model/backups"
TRAINING_DIR="${SCRATCH_DIR}/training_data"
GCS_DATA_URL="${GCS_DATA_URL:?Set GCS_DATA_URL in .env}"

echo "=========================================="
echo "X-ray to 3D Bone — Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# ---- Environment setup ----
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Python: $(which python) ($(python --version))"

pip install --quiet --upgrade pip
pip install --quiet -r <(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
print('\n'.join(deps))
")

mkdir -p model/logs model/checkpoints

# ---- Clean scratch ----
echo "Cleaning scratch..."
rm -rf "${SCRATCH_DIR}"
mkdir -p "${SCRATCH_DIR}"

# ---- GPU diagnostics ----
nvidia-smi
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
"
echo ""

# ---- Download training data from GCS ----
echo "Downloading training data from GCS..."
mkdir -p "${SCRATCH_DIR}"
gcloud storage cp "${GCS_DATA_URL}" "${SCRATCH_DIR}/training_data.tar.gz"
tar xzf "${SCRATCH_DIR}/training_data.tar.gz" -C "${SCRATCH_DIR}"
rm -f "${SCRATCH_DIR}/training_data.tar.gz"
echo "Downloaded $(ls "${TRAINING_DIR}" | wc -l) items."
echo ""

# ---- Back up diverse validation samples to home ----
echo "Backing up diverse validation samples to home..."
VAL_DATA_DIR="${BACKUP_DIR}/val_data"
rm -rf "${VAL_DATA_DIR}"
mkdir -p "${VAL_DATA_DIR}"

python3 -c "
import os, re, random, shutil
from collections import defaultdict

src = '${TRAINING_DIR}'
dst = '${VAL_DATA_DIR}'
N = 25

dirs = [d for d in os.listdir(src)
        if os.path.isdir(f'{src}/{d}') and os.path.isfile(f'{src}/{d}/voxels.npy')]

# Group by subset (body region) from dir name
def get_subset(name):
    m = re.match(r'^(.+?)_s\d+', name)
    return m.group(1) if m else 'unknown'

groups = defaultdict(list)
for d in dirs:
    groups[get_subset(d)].append(d)

for v in groups.values():
    random.shuffle(v)

# Round-robin across subsets for variety
picked = []
keys = sorted(groups.keys())
idx = 0
while len(picked) < N and any(groups[k] for k in keys):
    k = keys[idx % len(keys)]
    if groups[k]:
        picked.append(groups[k].pop())
    idx += 1

for d in picked:
    shutil.copytree(f'{src}/{d}', f'{dst}/{d}')

print(f'Backed up {len(picked)} samples from {len(set(get_subset(d) for d in picked))} subsets')
"
echo ""

# ---- Restore checkpoints ----
if [ ! -f "model/checkpoints/latest.pt" ] && [ ! -f "model/checkpoints/best.pt" ]; then
    if [ -f "${BACKUP_DIR}/latest.pt" ]; then
        echo "Restoring checkpoints from backup..."
        cp "${BACKUP_DIR}/latest.pt" model/checkpoints/latest.pt
        [ -f "${BACKUP_DIR}/best.pt" ] && cp "${BACKUP_DIR}/best.pt" model/checkpoints/best.pt
        [ -d "${BACKUP_DIR}/logs" ] && cp -r "${BACKUP_DIR}/logs/"* model/logs/ 2>/dev/null || true
        [ -d "${BACKUP_DIR}/val_samples" ] && cp -r "${BACKUP_DIR}/val_samples" model/logs/ 2>/dev/null || true
    else
        echo "Training from scratch."
    fi
fi
echo ""

# ---- Train ----
RESUME_FLAG=""
if [ -f "model/checkpoints/latest.pt" ]; then
    echo "Resuming from model/checkpoints/latest.pt"
    RESUME_FLAG="--resume model/checkpoints/latest.pt"
elif [ -f "model/checkpoints/best.pt" ]; then
    echo "Resuming from model/checkpoints/best.pt"
    RESUME_FLAG="--resume model/checkpoints/best.pt"
fi

python -m model.train \
    --data "${TRAINING_DIR}" \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda \
    --backup-dir "${BACKUP_DIR}" \
    --val-sample-interval 10 \
    ${RESUME_FLAG}

echo "=========================================="
echo "Training finished at $(date)"
echo "=========================================="
