import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def worker_init_fn(worker_id: int):
    """Seed each DataLoader worker for reproducibility."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


class XrayVoxelDataset(Dataset):
    """Dataset of paired (AP X-ray, Lateral X-ray, 3D voxel grid) samples."""

    def __init__(self, data_dir: str, split: str = "train", augment: bool = False):
        self.data_dir = Path(data_dir)
        self.augment = augment

        manifest = json.loads((self.data_dir / "manifest.json").read_text())
        self.sample_ids = manifest[split]
        self.voxel_res = manifest["voxel_resolution"]
        self.xray_res = manifest["xray_resolution"]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_dir = self.data_dir / sample_id

        ap = np.load(sample_dir / "ap.npy").astype(np.float32)
        lat = np.load(sample_dir / "lat.npy").astype(np.float32)
        voxels = np.load(sample_dir / "voxels.npy").astype(np.float32)

        if self.augment:
            ap, lat, voxels = self._augment(ap, lat, voxels)

        # Convert to tensors: add channel dim
        ap = torch.from_numpy(ap).unsqueeze(0)       # (1, 224, 224)
        lat = torch.from_numpy(lat).unsqueeze(0)      # (1, 224, 224)
        voxels = torch.from_numpy(voxels).unsqueeze(0)  # (1, 64, 64, 64)

        return ap, lat, voxels

    def _augment(self, ap, lat, voxels):
        # Random horizontal flip (flip X axis consistently across all three)
        if np.random.random() > 0.5:
            ap = np.flip(ap, axis=1).copy()
            lat = np.flip(lat, axis=1).copy()
            voxels = np.flip(voxels, axis=2).copy()  # flip X in voxel space

        # Random intensity jitter on X-rays
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            ap = np.clip(ap * scale, 0, 1)
            lat = np.clip(lat * scale, 0, 1)

        # Random Gaussian noise on X-rays
        if np.random.random() > 0.5:
            noise_std = np.random.uniform(0.01, 0.05)
            ap = np.clip(ap + np.random.normal(0, noise_std, ap.shape).astype(np.float32), 0, 1)
            lat = np.clip(lat + np.random.normal(0, noise_std, lat.shape).astype(np.float32), 0, 1)

        # Random 90-degree rotation around Y axis (swap X and Z)
        if np.random.random() > 0.5:
            voxels = np.transpose(voxels, (2, 1, 0)).copy()  # swap X and Z
            ap, lat = lat, ap  # swap the views to match

        return ap, lat, voxels
