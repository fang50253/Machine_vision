"""
DnCNN Denoising - Configuration
"""
import torch
import os

# ── Device ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model Architecture ──
NUM_LAYERS = 17           # DnCNN depth (original: 17)
NUM_FEATURES = 64         # feature channels (original: 64)
USE_ATTENTION = False     # channel attention (SELayer) — off by default for backward compat
NUM_LAYERS_DEEP = 20      # deeper variant depth
NUM_FEATURES_DEEP = 96    # deeper variant channels
CHANNELS = 3              # RGB input

# ── Training ──
TRAIN = {
    'patch_size': 128,
    'batch_size': 16,
    'epochs': 100,
    'lr': 1e-3,
    'lr_min': 1e-5,
    'weight_decay': 1e-4,
    # FIXED: clip_grad_value_ (per-param) instead of clip_grad_norm_ (total norm)
    # Original DnCNN paper uses 0.01 per parameter
    'grad_clip_value': 0.01,
    'noise_sigma': 25,
    'val_split': 0.05,
    'num_workers': 0,
    'seed': 42,
    # ── Data augmentation ──
    'aug_color_jitter': 0.05,       # color jitter strength (0 = off)
    'aug_gaussian_blur': 0.1,       # random blur probability
    # ── Curriculum learning ──
    'sigma_ramp_epochs': 20,        # linearly ramp sigma from 0 to noise_sigma over N epochs
    # ── Multi-scale ──
    'scale_aug_prob': 0.3,          # random scaling probability
    'scale_range': (0.5, 1.0),      # scale range for multi-scale
    # ── Perceptual loss (VGG) ──
    'perceptual_weight': 0.0,       # perceptual loss weight (0 = off; set 0.1~0.5 to enable)
    'frequency_weight': 0.0,        # FFT frequency loss weight (0 = off)
    # ── Joint fine-tuning ──
    'joint_ft_epochs': 20,
    'joint_ft_lr': 5e-5,
}

# ── Benchmark ──
BENCHMARK = {
    'noise_sigma': 25,
    'max_size': 1024,
    'tta_enabled': False,            # test-time augmentation (8-way flip/rotate)
    'tta_methods': 8,                # 2 (H flip only) or 8 (H+V+rotations)
}

# ── Paths ──
MODEL_SAVE_DIR = "trained_models"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
