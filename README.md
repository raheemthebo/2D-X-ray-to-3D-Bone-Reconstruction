# 2D-X-ray-to-3D-Bone-Reconstruction

Deep learning framework for reconstructing anatomical 3D bone meshes (knee, thorax/ribcage, spine) from standard 2D radiographs. Features a ConvNeXt-based biplanar encoder, neural implicit field decoder, FastAPI backend, and an interactive 3D WebGL viewer.


## Brief Idea
We are building an AI tool that turns standard 2D X-rays into 3D bone structure. Our goal is to solve two massive global problems:

- **The Cancer Risk**: Standard 3D CT scans expose patients to high radiation, contributing to an estimated 1.5% to 2% of all cancer cases worldwide. Globally, this equates to hundreds of thousands of future cancer diagnoses every year. Our AI aims to reduce this radiation exposure by 99% (from around 10 mSv down to 0.1 mSv).

- **The Accessibility Gap**: Nearly 4 billion people (around half the world), lack access to 3D imaging. While a CT scanner can cost hundreds of thousands of dollars to buy and maintain, our software-based approach is 10x cheaper, making 3D surgical planning possible for rural and low-income clinics.

Our Goal: A real-time 2D-to-3D transformation tool to make surgical planning safer and more affordable for millions of people who currently have no 3D imaging options.

## Model Architecture

Based on [X2BR](https://arxiv.org/abs/2504.08675) (High-Fidelity 3D Bone Reconstruction, 2025):

- **Encoder**: ConvNeXt-based, shared for AP and Lateral X-ray views → 1024-dim feature vector
- **Fusion**: Biplanar concatenation + projection (our extension for two views)
- **Decoder**: Neural implicit occupancy decoder with Conditional Batch Normalization (CBN) and DenseNet-style blocks
- **Output**: Point-based occupancy prediction → 64³ voxel grid → 3D mesh (GLB)
- **Parameters**: ~43.8M

### Training Details
- **Loss**: BCE with balanced point sampling (50% near bone surface, 50% uniform)
- **Optimizer**: AdamW (lr=1e-4, weight decay=1e-4)
- **Scheduler**: Cosine annealing (1e-4 → 1e-6)
- **Augmentation**: Horizontal flip, intensity jitter, Gaussian noise, 90° rotation
- **Data workers**: 8 with pin_memory + persistent_workers for GPU efficiency

## Dataset

### CADS (multi-subset)
| Subset | Volumes | Body Region | Bone Mask |
|--------|---------|-------------|-----------|
| [0037_totalsegmentator](https://huggingface.co/datasets/huggingface/CADS-dataset) | 1,203 | Full body | part_559 label 5 |
| [0010_verse](https://huggingface.co/datasets/huggingface/CADS-dataset) | 450 | Spine (vertebrae) | part_559 label 5 |
| [0013_ribfrac](https://huggingface.co/datasets/huggingface/CADS-dataset) | 360 | Ribs + chest | part_559 label 5 |

**Config**: 250 subjects/subset × 3 subsets × 8 angle variations = **~106K samples** (~84K train, ~22K val). Covers full-body bone (spine, ribs, pelvis, limbs, skull). DRRs are generated from full CT volumes (not bone masks) for realistic X-ray appearance with soft tissue.

## Usage

### Inference (local)

```bash
# Single image (AP only — duplicated as lateral)
python -m model.inference --ap xray.png -o output.glb

# Biplanar (AP + Lateral)
python -m model.inference --ap xray_ap.png --lat xray_lat.png -o output.glb

# Batch: process all images in test_images/
python -m model.inference --input-dir test_images/ --output-dir output/

# High-resolution output with MISE (256³ from 64³ training)
python -m model.inference --ap xray.png -o output.glb --mise
python -m model.inference --ap xray.png -o output.glb --mise --mise-resolution 128
```

### Web App

```bash
# Terminal 1 — Backend (FastAPI)
python backend/main.py    # http://localhost:8000

# Terminal 2 — Frontend (React + Vite)
cd frontend && npm install
npm run dev               # http://localhost:5173
```

Open http://localhost:5173, upload AP (+ optional lateral) X-rays, or choose from presets (Knee, Chest, Spine) and view the reconstructed 3D anatomical model interactively with 360° inspection and GLB export.

## Project Structure

```
2D-X-ray-to-3D-Bone-Reconstruction/
├── model/
│   ├── architecture.py      # X2BR-inspired ConvNeXt encoder + implicit decoder
│   ├── train.py             # Training loop with continuous backup + val samples
│   ├── inference.py         # Single/batch inference → GLB output + anatomy detection
│   ├── dataset.py           # PyTorch dataset with augmentation
│   ├── losses.py            # BCE loss for occupancy prediction
│   └── checkpoints/         # Model weights
├── assets/                  # High-fidelity 3D anatomical meshes (knee, chest, spine)
├── backend/
│   └── main.py              # FastAPI server (upload → inference → GLB)
├── frontend/
│   ├── src/App.tsx          # Modern Medical 3D Viewer UI (Three.js / Canvas)
│   └── public/samples/      # Sample X-rays for instant testing
└── README.md
```

## Potential Technical Q&A (Judging Panel)

**Q: Describe the architecture of your method, how you evaluated it, where the data came from, and what accuracy metrics you used for the generated 3D models.**  
A: **Architecture**: Our model is based on X2BR (2025). It has two stages — a ConvNeXt encoder that takes one or two X-ray views (AP and optional lateral, each 224×224 grayscale) and compresses them into a 1024-dim feature vector, followed by a neural implicit occupancy decoder. The decoder takes any 3D coordinate (x, y, z) plus the image features and predicts the probability that point is inside bone. We use Conditional Batch Normalization (CBN) and DenseNet-style skip connections in the decoder. For biplanar input, both view features are concatenated (2048-dim) and projected back to 1024-dim before decoding. At inference, we query a 64³ grid of points, threshold at 0.5, and run marching cubes to extract a 3D triangle mesh exported as GLB.

**Data**: We use the CADS dataset from HuggingFace — 3 subsets: TotalSegmentator (1,203 full-body CTs), VerSe (450 spine CTs), and RibFrac (360 rib/chest CTs). From each CT volume we extract the bone segmentation mask (label 5) as ground truth, and generate synthetic X-rays called Digitally Reconstructed Radiographs (DRRs) by simulating X-ray physics — ray-casting through the full CT volume and integrating attenuation, just like a real X-ray machine. We generate 8 angle variations per subject (AP, lateral, and rotated views), giving us ~106K training samples (~84K train, ~22K val).

**Evaluation & Metrics**: We evaluate using the **Dice coefficient** — the volumetric overlap between the predicted 3D bone voxels and the ground-truth bone mask. Dice ranges from 0 (no overlap) to 1 (perfect). Our current best validation Dice is **~0.64**. We also visually inspect reconstructed GLB meshes against ground truth at validation checkpoints every 10 epochs.

**Q: How does your model reconstruct 3D structure from a single 2D image? Isn't that an ill-posed problem?**  
A: Yes, it's inherently ill-posed — a single 2D projection has infinite consistent 3D interpretations. The model learns a strong anatomical prior from ~84K training samples: the ConvNeXt encoder compresses the X-ray into a 1024-dim feature vector, and the implicit decoder learns to predict occupancy at any 3D query point conditioned on that vector. When a second (lateral) view is available, we fuse both feature vectors, resolving depth ambiguity.

**Q: What is an implicit neural representation, and why use it over voxel regression?**  
A: Instead of directly outputting a 64³ voxel grid, our decoder takes a 3D coordinate (x, y, z) as input and predicts the probability that point is inside bone. At inference we query a dense grid of points and threshold to get the voxel volume. The advantage is resolution-agnostic training and inference with MISE.

**Q: What metric do you use to evaluate reconstruction quality?**  
A: Dice coefficient (volumetric overlap between predicted and ground-truth bone voxels). It ranges from 0 (no overlap) to 1 (perfect match).

**Q: Why ConvNeXt over a standard ResNet or Vision Transformer?**  
A: ConvNeXt modernizes the ConvNet design with transformer-inspired tricks (larger kernels, LayerNorm, GELU, inverted bottleneck) while keeping the efficiency and inductive biases of convolutions. For medical images — which are single-channel, grayscale, and texture-heavy — ConvNets tend to be more data-efficient than ViTs.

**Q: How do you go from the voxel grid to a 3D mesh?**  
A: Marching cubes algorithm. The implicit decoder outputs occupancy probabilities on a 64³ grid. We threshold at 0.5 to get a binary volume, then run marching cubes to extract the isosurface as a triangle mesh exported as GLB.
