"""
X2BR-inspired biplanar X-ray to 3D bone reconstruction via neural implicit fields.

Architecture (following X2BR: High-Fidelity 3D Bone Reconstruction, 2025):
    AP X-ray (1, 224, 224)  → ConvNeXt Encoder → 1024-dim global features
    LAT X-ray (1, 224, 224) → ConvNeXt Encoder (shared weights)
                                    ↓
                              Biplanar Feature Fusion
                                    ↓
                              Implicit Decoder with CBN + DenseNet blocks
                                    ↓
                              Query 3D points → occupancy (0-1)

Key design choices from X2BR:
- ConvNeXt-based encoder → 1024-dim latent representation
- Neural implicit occupancy decoder (point-based, not voxel grid)
- Conditional Batch Normalization (CBN) for image-conditioned decoding
- DenseNet-style connections in decoder for rich feature propagation
- Training at 64³ voxel resolution
- Binary cross-entropy loss on 2048 sampled points

Biplanar extension (our addition):
- Shared ConvNeXt encoder processes both AP and LAT views
- Features fused via concatenation + projection before conditioning decoder
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ConvNeXt Encoder (following ConvNeXt V1 design)
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv → LayerNorm → pointwise expand → GELU → pointwise shrink."""

    def __init__(self, dim: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        return residual + x


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-based hierarchical encoder for grayscale X-ray images.

    Produces a 1024-dim global feature vector (matching X2BR).

    Input:  (B, 1, 224, 224)
    Output: (B, 1024) global feature vector
    """

    def __init__(
        self,
        in_channels: int = 1,
        dims: tuple[int, ...] = (96, 192, 384, 1024),
        depths: tuple[int, ...] = (3, 3, 9, 3),
    ):
        super().__init__()
        self.feature_dim = dims[-1]

        # Stem: patchify with 4×4 conv, stride 4 (224→56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),  # LayerNorm equivalent for conv
        )

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(dims)):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

            # Downsample between stages (except last)
            if i < len(dims) - 1:
                ds = nn.Sequential(
                    nn.GroupNorm(1, dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamples.append(ds)

        self.norm = nn.LayerNorm(dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)  # (B, 96, 56, 56)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # Global average pool → (B, 1024)
        x = x.mean(dim=[2, 3])
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Implicit Decoder with Conditional Batch Normalization (X2BR-style)
# ---------------------------------------------------------------------------

class ConditionalBatchNorm(nn.Module):
    """Conditional Batch Normalization: BN whose affine params come from a conditioning vector.

    Following X2BR: the CBN layers are parameterized by β_i and γ_i derived from
    the image feature vector, dynamically adjusting normalization per-sample.
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.gamma_fc = nn.Linear(cond_dim, num_features)
        self.beta_fc = nn.Linear(cond_dim, num_features)

        # Initialize near-identity transform
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) point features where N is num_points, D is num_features
            cond: (B, cond_dim) conditioning features from encoder
        Returns:
            (B, N, D) normalized and conditioned features
        """
        B, N, D = x.shape
        # BN operates on (B*N, D)
        x_flat = x.reshape(B * N, D)
        x_norm = self.bn(x_flat).reshape(B, N, D)

        gamma = self.gamma_fc(cond).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_fc(cond).unsqueeze(1)     # (B, 1, D)
        return gamma * x_norm + beta


class DenseBlock(nn.Module):
    """DenseNet-style block with CBN conditioning (X2BR decoder component).

    Each layer receives concatenated features from all previous layers,
    enabling rich feature propagation through the decoder.
    """

    def __init__(self, in_dim: int, growth: int, cond_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, growth)
        self.cbn = ConditionalBatchNorm(growth, cond_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = self.cbn(out, cond)
        out = self.act(out)
        return torch.cat([x, out], dim=-1)


class ImplicitDecoder(nn.Module):
    """Neural implicit occupancy decoder with DenseNet blocks and CBN.

    Maps (3D query points, image conditioning) → occupancy probability.
    Following X2BR: uses DenseNet-style connections with CBN layers
    parameterized by the encoder features.

    Input:
        points: (B, N, 3) query point coordinates in [0, 1]^3
        cond:   (B, cond_dim) image feature conditioning
    Output:
        (B, N, 1) occupancy logits
    """

    def __init__(self, cond_dim: int = 1024, hidden_dim: int = 256, num_dense_blocks: int = 5):
        super().__init__()

        # Fourier positional encoding for query points
        self.num_freq = 6  # 6 frequencies → 3 + 6*2*3 = 39 dims
        point_encoded_dim = 3 + self.num_freq * 2 * 3

        # Initial point projection
        self.point_proj = nn.Linear(point_encoded_dim, hidden_dim)
        self.point_cbn = ConditionalBatchNorm(hidden_dim, cond_dim)
        self.point_act = nn.GELU()

        # DenseNet blocks with CBN
        self.dense_blocks = nn.ModuleList()
        current_dim = hidden_dim
        growth = hidden_dim // 2
        for _ in range(num_dense_blocks):
            self.dense_blocks.append(DenseBlock(current_dim, growth, cond_dim))
            current_dim += growth

        # Output head
        self.head = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _positional_encode(self, points: torch.Tensor) -> torch.Tensor:
        """Fourier feature encoding for 3D coordinates."""
        freq_bands = 2.0 ** torch.arange(self.num_freq, device=points.device, dtype=points.dtype)
        pts_freq = points.unsqueeze(-1) * freq_bands  # (B, N, 3, num_freq)
        sin_feat = torch.sin(pts_freq * torch.pi)
        cos_feat = torch.cos(pts_freq * torch.pi)
        encoded = torch.cat([sin_feat, cos_feat], dim=-1).reshape(*points.shape[:2], -1)
        return torch.cat([points, encoded], dim=-1)  # (B, N, 3 + 3*num_freq*2)

    def forward(self, points: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self._positional_encode(points)
        x = self.point_proj(x)
        x = self.point_cbn(x, cond)
        x = self.point_act(x)

        for block in self.dense_blocks:
            x = block(x, cond)

        return self.head(x)  # (B, N, 1) — raw logits


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class X2BRBiplanarModel(nn.Module):
    """X2BR-inspired biplanar X-ray to 3D bone reconstruction model.

    Combines:
    - ConvNeXt encoder (shared for AP and LAT views) → 1024-dim
    - Biplanar feature fusion (concat + project)
    - Neural implicit decoder with CBN (point-based occupancy prediction)

    Training mode:
        forward(ap, lat, query_points) → occupancy logits (B, N, 1)

    Inference mode (no query_points):
        forward(ap, lat) → dense voxel grid (B, 1, R, R, R)
    """

    def __init__(
        self,
        encoder_dims: tuple[int, ...] = (96, 192, 384, 1024),
        encoder_depths: tuple[int, ...] = (3, 3, 9, 3),
        decoder_hidden: int = 256,
        decoder_dense_blocks: int = 5,
        voxel_resolution: int = 64,
    ):
        super().__init__()
        self.voxel_resolution = voxel_resolution
        encoder_feat_dim = encoder_dims[-1]  # 1024
        fused_dim = encoder_feat_dim  # after projection

        # Shared ConvNeXt encoder for both views
        self.encoder = ConvNeXtEncoder(in_channels=1, dims=encoder_dims, depths=encoder_depths)

        # Biplanar fusion: concat AP+LAT features → project to fused_dim
        self.fusion = nn.Sequential(
            nn.Linear(encoder_feat_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )

        # Implicit decoder
        self.decoder = ImplicitDecoder(
            cond_dim=fused_dim,
            hidden_dim=decoder_hidden,
            num_dense_blocks=decoder_dense_blocks,
        )

    def encode(self, ap: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        """Encode biplanar X-rays into a fused conditioning vector.

        Args:
            ap:  (B, 1, 224, 224)
            lat: (B, 1, 224, 224)
        Returns:
            (B, 1024) conditioning vector
        """
        feat_ap = self.encoder(ap)    # (B, 1024)
        feat_lat = self.encoder(lat)  # (B, 1024)
        fused = self.fusion(torch.cat([feat_ap, feat_lat], dim=-1))
        return fused

    def decode_points(self, cond: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Predict occupancy for query points given conditioning.

        Args:
            cond:   (B, fused_dim) image conditioning
            points: (B, N, 3) query points in [0, 1]^3
        Returns:
            (B, N, 1) occupancy logits
        """
        return self.decoder(points, cond)

    def decode_dense(self, cond: torch.Tensor, resolution: int | None = None) -> torch.Tensor:
        """Evaluate occupancy on a dense 3D grid.

        Args:
            cond:       (B, fused_dim) image conditioning
            resolution: grid resolution (default: self.voxel_resolution)
        Returns:
            (B, 1, R, R, R) occupancy probabilities in [0, 1]
        """
        R = resolution or self.voxel_resolution
        device = cond.device
        B = cond.shape[0]

        # Create grid points in [0, 1]^3
        coords = torch.linspace(0, 1, R, device=device)
        gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing="ij")
        # Match training point order: (D, H, W) — same as torch.nonzero on (D,H,W) volume
        grid_points = torch.stack([gz, gy, gx], dim=-1).reshape(1, -1, 3)  # (1, R³, 3)
        grid_points = grid_points.expand(B, -1, -1)  # (B, R³, 3)

        # Evaluate in chunks to avoid OOM
        chunk_size = 32768
        total_points = grid_points.shape[1]
        logits_list = []

        for start in range(0, total_points, chunk_size):
            end = min(start + chunk_size, total_points)
            chunk = grid_points[:, start:end, :]
            logits_chunk = self.decoder(chunk, cond)  # (B, chunk, 1)
            logits_list.append(logits_chunk)

        logits = torch.cat(logits_list, dim=1)  # (B, R³, 1)
        probs = torch.sigmoid(logits)
        return probs.reshape(B, 1, R, R, R)

    @torch.no_grad()
    def decode_mise(
        self,
        cond: torch.Tensor,
        initial_resolution: int = 32,
        final_resolution: int = 256,
        threshold: float = 0.5,
        chunk_size: int = 32768,
    ) -> torch.Tensor:
        """Multiresolution IsoSurface Extraction (MISE) — X2BR-style.

        Starts with a coarse grid evaluation, then iteratively subdivides only
        the voxels near the surface (where occupancy is uncertain). This produces
        a high-resolution occupancy field while evaluating far fewer points than
        a full dense grid at final_resolution.

        Args:
            cond:                (B, fused_dim) conditioning features
            initial_resolution:  starting grid resolution (default: 32)
            final_resolution:    target resolution (default: 256)
            threshold:           occupancy threshold for surface detection
            chunk_size:          max points per decoder call

        Returns:
            (B, 1, final_resolution, final_resolution, final_resolution) occupancy probs
        """
        device = cond.device
        B = cond.shape[0]

        # We process one sample at a time for MISE (batch dim handled in loop)
        all_grids = []

        for b in range(B):
            cond_b = cond[b:b+1]  # (1, fused_dim)

            # Step 1: Evaluate coarse grid
            R = initial_resolution
            grid = torch.zeros(final_resolution, final_resolution, final_resolution,
                               device=device)
            evaluated = torch.zeros_like(grid, dtype=torch.bool)

            # Coarse grid points
            stride = final_resolution // R
            coords_coarse = torch.arange(0, final_resolution, stride, device=device)
            gz, gy, gx = torch.meshgrid(coords_coarse, coords_coarse, coords_coarse, indexing="ij")
            coarse_indices = torch.stack([gz.flatten(), gy.flatten(), gx.flatten()], dim=1)  # (R³, 3)

            # Evaluate coarse points
            points_01 = coarse_indices.float() / (final_resolution - 1)  # [0, 1]
            points_01 = points_01.unsqueeze(0)  # (1, R³, 3)

            probs = self._eval_points_chunked(cond_b, points_01, chunk_size)  # (1, R³, 1)
            probs = probs.squeeze(0).squeeze(-1)  # (R³,)

            for i, idx in enumerate(coarse_indices):
                grid[idx[0], idx[1], idx[2]] = probs[i]
                evaluated[idx[0], idx[1], idx[2]] = True

            # Step 2: Iteratively subdivide near-surface regions
            current_stride = stride
            while current_stride > 1:
                current_stride //= 2

                # Find coarse voxels near the surface (occupancy between 0.1 and 0.9)
                # Check which evaluated points are "uncertain" (near threshold)
                margin = 0.4  # evaluate neighbors of voxels with occ in [0.1, 0.9]
                uncertain = evaluated & (grid > threshold - margin) & (grid < threshold + margin)

                if not uncertain.any():
                    break

                # For each uncertain voxel, add its un-evaluated neighbors at finer stride
                uncertain_indices = torch.nonzero(uncertain, as_tuple=False)  # (K, 3)
                new_points_set = set()

                for idx in uncertain_indices:
                    z, y, x = idx[0].item(), idx[1].item(), idx[2].item()
                    # Add neighboring points at current_stride resolution
                    for dz in range(-current_stride, current_stride + 1, current_stride):
                        for dy in range(-current_stride, current_stride + 1, current_stride):
                            for dx in range(-current_stride, current_stride + 1, current_stride):
                                nz = z + dz
                                ny = y + dy
                                nx = x + dx
                                if (0 <= nz < final_resolution and
                                    0 <= ny < final_resolution and
                                    0 <= nx < final_resolution and
                                    not evaluated[nz, ny, nx]):
                                    new_points_set.add((nz, ny, nx))

                if not new_points_set:
                    break

                new_indices = torch.tensor(list(new_points_set), device=device, dtype=torch.long)
                new_points_01 = new_indices.float() / (final_resolution - 1)
                new_points_01 = new_points_01.unsqueeze(0)  # (1, N, 3)

                new_probs = self._eval_points_chunked(cond_b, new_points_01, chunk_size)
                new_probs = new_probs.squeeze(0).squeeze(-1)  # (N,)

                for i, idx in enumerate(new_indices):
                    grid[idx[0], idx[1], idx[2]] = new_probs[i]
                    evaluated[idx[0], idx[1], idx[2]] = True

            # Step 3: Fill unevaluated interior voxels (clearly inside or outside)
            # Voxels surrounded by all-occupied neighbors → 1, all-empty → 0
            # Use nearest-neighbor interpolation from evaluated points
            if not evaluated.all():
                from scipy.ndimage import distance_transform_edt
                eval_np = evaluated.cpu().numpy()
                grid_np = grid.cpu().numpy()

                # For unevaluated points, find nearest evaluated point's value
                _, nearest_indices = distance_transform_edt(~eval_np, return_distances=True,
                                                             return_indices=True)
                filled = grid_np[nearest_indices[0], nearest_indices[1], nearest_indices[2]]
                grid = torch.from_numpy(filled).to(device)

            all_grids.append(grid)

        result = torch.stack(all_grids, dim=0).unsqueeze(1)  # (B, 1, R, R, R)
        return result

    def _eval_points_chunked(self, cond: torch.Tensor, points: torch.Tensor,
                              chunk_size: int) -> torch.Tensor:
        """Evaluate occupancy for points in chunks. Returns sigmoid probabilities."""
        total = points.shape[1]
        probs_list = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            logits = self.decoder(points[:, start:end, :], cond)
            probs_list.append(torch.sigmoid(logits))
        return torch.cat(probs_list, dim=1)

    def forward(
        self,
        ap: torch.Tensor,
        lat: torch.Tensor,
        query_points: torch.Tensor | None = None,
        mise: bool = False,
        mise_resolution: int = 256,
    ) -> torch.Tensor:
        """
        Training:  forward(ap, lat, query_points) → (B, N, 1) occupancy logits
        Inference: forward(ap, lat)                → (B, 1, R, R, R) occupancy probs
        Inference: forward(ap, lat, mise=True)     → (B, 1, 256, 256, 256) via MISE
        """
        cond = self.encode(ap, lat)

        if query_points is not None:
            return self.decode_points(cond, query_points)

        if mise:
            return self.decode_mise(cond, final_resolution=mise_resolution)

        return self.decode_dense(cond)


# ---------------------------------------------------------------------------
# Legacy alias for backward compatibility
# ---------------------------------------------------------------------------
XrayTo3DModel = X2BRBiplanarModel
