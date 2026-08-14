<div align="center">
  
# 🔬 KLA-Restore: AI-Based Restoration of Degraded Semiconductor Inspection Images

**Joint Denoising + Super-Resolution for Semiconductor Inspection Imagery**

*Built for the KLA AI Hackathon 2026*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

In semiconductor manufacturing, inspection images must be extremely sharp and clean — a single pixel of noise or a small loss of detail can hide a defect that causes a chip to fail. In practice, inspection images are frequently degraded by:

- **Speckle noise** — signal-dependent, multiplicative grain that can push pixel intensities beyond the true signal range
- **Gaussian noise** — softens edges and washes out fine structural detail
- **Spatial resolution reduction** — downsampling (e.g. 512×512 → 256×256, or 256×256 → 128×128) that permanently discards high-frequency detail

**KLA-Restore** is an AI pipeline that takes a degraded (noisy, low-resolution) inspection image and restores it — removing noise while reconstructing detail lost during downsampling — to match the original high-resolution, high-SNR ground truth.

---

## 🎯 Problem Statement

> Given a degraded image (noisy + low-resolution), train a model that reverses the degradation and produces a restored image matching the ground truth as closely as possible — generalizing to image sources and degradation patterns not seen during training.

**Constraints the solution must satisfy:**
- Handle speckle noise, Gaussian noise, and resolution loss **simultaneously** (a single image may carry all three)
- Generalize to out-of-distribution test samples from unseen sources — not just memorize training data
- Be **fast**: the full pipeline (model load → I/O → inference → write-out) is benchmarked end-to-end on an NVIDIA H100 GPU, and faster pipelines are preferred when quality is comparable

**Evaluation metrics:**
| Metric | Measures |
|---|---|
| **SSIM** | Structural similarity |
| **pSNR** | Pixel-level fidelity |
| **LPIPS** | Perceptual similarity (deep-feature based) |
| **Inference time** | End-to-end pipeline latency on H100 |

---

## 🏗️ Architecture

A lightweight **U-Net encoder-decoder** with a **PixelShuffle super-resolution head**, trained end-to-end for joint denoising + upsampling:

```
Degraded Image (e.g. 256×256)
        │
        ├──────────────────────────────┐
        ▼                              │ (bicubic upsample —
   U-Net Encoder-Decoder               │  residual base)
   (denoising context,                 │
    skip connections)                  │
        │                              │
        ▼                              │
   PixelShuffle SR Head                │
   (learned upsampling)                │
        │                              │
        ▼                              ▼
      Residual  ────────────────────  Add
        │
        ▼
Restored Image (e.g. 512×512)
```

**Design choices:**
- **Residual learning** — the model predicts a *correction* on top of a bicubic-upsampled base, rather than reconstructing the image from scratch. This trains faster and more stably.
- **PixelShuffle upsampling** — the network learns the upsampling kernel instead of relying on fixed interpolation.
- **GroupNorm over BatchNorm** — restoration models often train with small batches (large images eat VRAM); GroupNorm stays stable at low batch sizes.
- **Lightweight by design** (~1–4M params, tunable) — inference speed is scored, so this deliberately avoids heavy transformer backbones unless the quality gain justifies the latency cost.

**Loss function** — a weighted combination matching the evaluation metrics themselves:

```
L = λ₁ · L1(pred, target) + λ₂ · (1 − SSIM(pred, target)) + λ₃ · LPIPS(pred, target)
```

| Term | Why it's included |
|---|---|
| L1 | Pixel-level fidelity — correlates with pSNR |
| SSIM | Structural/contrast fidelity |
| LPIPS | Perceptual/texture realism that pixel losses miss entirely |

---

## 📂 Repository Structure

```
kla_restoration/
├── models/
│   ├── unet_restoration.py   # U-Net + PixelShuffle SR head
│   └── losses.py               # Combined L1 + SSIM + LPIPS loss
├── data/
│   └── dataset.py               # Paired dataset loader + synthetic data generator
├── scripts/
│   ├── train.py                  # Training loop
│   ├── infer.py                  # Standalone inference script (submission format)
│   └── compute_metrics.py        # Self-evaluation: SSIM / pSNR / LPIPS
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & install
```bash
git clone https://github.com/ShubhamAjayTiwari/kla-restore.git
cd kla-restore
pip install -r requirements.txt
```

### 2. Prepare the dataset
Place KLA's paired dataset as:
```
your_dataset/
├── ground_truth/   0001.png, 0002.png, ...
└── degraded/        0001.png, 0002.png, ...   (matching filenames)
```

### 3. Train
```bash
python scripts/train.py \
    --data_root /path/to/your_dataset \
    --epochs 50 --batch_size 16 --patch_size 128 --scale 2 --base_ch 48
```

### 4. Run inference (submission format)
```bash
python scripts/infer.py \
    --input_dir /path/to/test_images \
    --output_dir outputs \
    --checkpoint checkpoints/best_model.pt --scale 2
```

### 5. Self-evaluate
```bash
python scripts/compute_metrics.py \
    --restored_dir outputs --gt_dir /path/to/test_ground_truth --use_lpips
```

---

## ✅ Status

- [x] Model architecture — verified forward pass, correct shapes
- [x] Dataset loader + synthetic multiplicative-speckle data generator
- [x] Combined L1 + SSIM + LPIPS loss
- [x] Training loop — loss decreases, checkpointing works
- [x] Standalone inference script matching submission requirements
- [x] Metrics self-evaluation script
- [ ] Training on real KLA dataset
- [ ] Hyperparameter tuning (loss weights, `base_ch`, augmentation strength)
- [ ] Inference-speed optimization (mixed precision, batching) for H100 benchmark
- [ ] Out-of-distribution generalization testing

---

## 👥 Team

Built by **Shubham** ([@ShubhamAjayTiwari](https://github.com/ShubhamAjayTiwari)) and teammate for the KLA AI Hackathon 2026.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
