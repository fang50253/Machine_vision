#!/usr/bin/env python
"""
train_stage2.py
Stage 2: Edge enhancement training (DnCNN frozen).

Pipeline:
   noisy image → [DnCNN (frozen)] → denoised → [EdgeEnhance (training)] → enhanced

Only EdgeEnhance weights are updated. DnCNN's best model is loaded and frozen.

Usage:
    python train_stage2.py --data ../Div2k/DIV2K_train_HR \
        --dncnn trained_models/dncnn_best_20260720_201132.pth

    # With separate validation set
    python train_stage2.py --data ../Div2k/DIV2K_train_HR \
        --val ../Div2k/DIV2K_valid_HR \
        --dncnn trained_models/dncnn_best_20260720_201132.pth
"""
import argparse, os, sys, time, math, random, glob
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import config
from models.dncnn import DnCNN
from models.edge_enhancer import EdgeEnhancementNetwork
from models.losses import CombinedLoss


# ── dataset (same as train.py) ──

class DenoiseDataset(Dataset):
    """On-the-fly patch extraction + Gaussian noise addition."""

    def __init__(self, root: str, patch_size: int = 128, sigma: float = 25.0,
                 max_samples: int | None = None,
                 sigma_ramp_epochs: int = 0,
                 deterministic: bool = False, noise_seed: int | None = None):
        super().__init__()
        self.patch_size = patch_size
        self.sigma = sigma / 255.0
        self.deterministic = deterministic
        self.noise_seed = noise_seed
        self.files = self._collect_images(root)
        if max_samples is not None and max_samples < len(self.files):
            self.files = self.files[:max_samples]
        tag = " (deterministic)" if deterministic else ""
        print(f"Dataset [{root}]: {len(self.files)} images{tag}")

    @staticmethod
    def _collect_images(root: str) -> list[str]:
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
        files: list[str] = []
        for ext in exts:
            pattern = os.path.join(root, '**', ext)
            files.extend(glob.glob(pattern, recursive=True))
        return sorted(set(files))

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT101)
            h, w = img.shape[:2]
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        return img[y:y + self.patch_size, x:x + self.patch_size]

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
        k = random.randint(0, 3)
        if k:
            patch = np.rot90(patch, k).copy()
        return patch

    def __getitem__(self, idx: int):
        img = self._load_image(self.files[idx])
        patch = self._random_crop(img)
        if not self.deterministic:
            patch = self._augment(patch)
        clean = patch / 255.0

        if self.noise_seed is not None:
            rng = np.random.RandomState(self.noise_seed + idx)
            noise = rng.randn(*clean.shape).astype(np.float32) * self.sigma
        else:
            noise = np.random.randn(*clean.shape).astype(np.float32) * self.sigma

        noisy = np.clip(clean + noise, 0.0, 1.0)
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1))
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1))
        return noisy_t, clean_t


# ── edge enhancement loss ──

class EdgeEnhanceLoss(nn.Module):
    """
    Loss for edge enhancement training:
        L = λ_recon * L1(enhanced, clean) + λ_edge * edge_preserve(enhanced, clean)
    """
    def __init__(self, lambda_recon=1.0, lambda_edge=0.3):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_edge = lambda_edge
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _edge_loss(self, enhanced, clean):
        g1 = torch.mean(enhanced, dim=1, keepdim=True)
        g2 = torch.mean(clean, dim=1, keepdim=True)
        dev = enhanced.device
        sx = self.sobel_x.to(dev)
        sy = self.sobel_y.to(dev)
        g1x = F.conv2d(g1, sx, padding=1)
        g1y = F.conv2d(g1, sy, padding=1)
        g2x = F.conv2d(g2, sx, padding=1)
        g2y = F.conv2d(g2, sy, padding=1)
        g1_mag = torch.sqrt(g1x**2 + g1y**2 + 1e-6)
        g2_mag = torch.sqrt(g2x**2 + g2y**2 + 1e-6)
        return F.l1_loss(g1_mag, g2_mag)

    def forward(self, enhanced, clean):
        recon = F.l1_loss(enhanced, clean)
        edge = self._edge_loss(enhanced, clean)
        total = self.lambda_recon * recon + self.lambda_edge * edge
        return total, {'recon': recon.item(), 'edge': edge.item()}


# ── helpers ──

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dncnn(ckpt_path: str, device: torch.device) -> DnCNN:
    model = DnCNN(channels=config.CHANNELS, num_layers=config.NUM_LAYERS,
                  num_features=config.NUM_FEATURES)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'model_state' in state:
        state = state['model_state']
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


# ── main ──

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: Train EdgeEnhance on frozen DnCNN outputs')
    parser.add_argument('--data', required=True,
                        help='Path to training images folder')
    parser.add_argument('--val', default=None,
                        help='Path to validation images (optional, auto-split 5%)')
    parser.add_argument('--dncnn', required=True,
                        help='Path to pre-trained DnCNN best model (.pth)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=config.TRAIN['batch_size'])
    parser.add_argument('--patch-size', type=int, default=config.TRAIN['patch_size'])
    parser.add_argument('--sigma', type=float, default=config.TRAIN['noise_sigma'])
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for edge network (default: 3e-4)')
    parser.add_argument('--lambda-recon', type=float, default=1.0,
                        help='Reconstruction L1 loss weight')
    parser.add_argument('--lambda-edge', type=float, default=0.3,
                        help='Edge preservation loss weight')
    parser.add_argument('--seed', type=int, default=config.TRAIN['seed'])
    parser.add_argument('--out', default=config.MODEL_SAVE_DIR)
    parser.add_argument('--perceptual', action='store_true',
                        help='Enable VGG perceptual loss on top of edge loss')
    args = parser.parse_args()

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device}  |  CUDA: {torch.cuda.is_available()}")
    print(f"DnCNN model: {args.dncnn}")

    # ── load frozen DnCNN ──
    dncnn = load_dncnn(args.dncnn, device)
    dncnn.eval()
    for p in dncnn.parameters():
        p.requires_grad = False
    n_d = sum(p.numel() for p in dncnn.parameters())
    print(f"DnCNN frozen: {n_d:,} params")

    # ── create edge network ──
    edge_net = EdgeEnhancementNetwork(in_channels=config.CHANNELS, base_channels=64)
    n_e = sum(p.numel() for p in edge_net.parameters())
    print(f"EdgeEnhance: {n_e:,} params (trainable)")

    # ── datasets ──
    full_dataset = DenoiseDataset(
        root=args.data, patch_size=args.patch_size, sigma=args.sigma)

    if args.val is not None and os.path.isdir(args.val):
        train_dataset = full_dataset
        val_dataset = DenoiseDataset(
            root=args.val, patch_size=args.patch_size, sigma=args.sigma,
            sigma_ramp_epochs=0,
            deterministic=True, noise_seed=42)
    else:
        val_size = max(1, int(len(full_dataset) * config.TRAIN['val_split']))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed))
        print(f"Auto-split: train={train_size}  val={val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config.TRAIN['num_workers'], pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.TRAIN['num_workers'], pin_memory=True)

    # ── optimiser & loss ──
    optimizer = optim.Adam(edge_net.parameters(), lr=args.lr,
                           weight_decay=config.TRAIN['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=config.TRAIN['lr_min'])
    criterion = EdgeEnhanceLoss(args.lambda_recon, args.lambda_edge)

    # Optional perceptual loss on top
    perceptual_loss = None
    w_perc = 0.0
    if args.perceptual and config.TRAIN['perceptual_weight'] > 0:
        from models.losses import PerceptualLoss
        perceptual_loss = PerceptualLoss(device=device)
        w_perc = config.TRAIN['perceptual_weight'] * 0.5  # scale down for edge task
        print(f"Perceptual loss enabled (weight={w_perc})")

    # ── training loop ──
    edge_net.to(device)
    best_loss = float('inf')
    prev_best: str | None = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*70}")
    print(f"Stage 2: Edge Enhancement Training")
    print(f"epochs={args.epochs}  lr={args.lr}  "
          f"lambda_recon={args.lambda_recon}  lambda_edge={args.lambda_edge}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train one epoch
        edge_net.train()
        train_total, train_count = 0.0, 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            with torch.no_grad():
                noise_pred = dncnn(noisy)
                denoised = noisy - noise_pred

            optimizer.zero_grad()
            enhanced, _ = edge_net(denoised)
            loss, details = criterion(enhanced, clean)

            # Add perceptual loss if enabled
            if perceptual_loss is not None:
                perc = perceptual_loss(enhanced, clean)
                loss = loss + w_perc * perc
                details['perceptual'] = perc.item()

            loss.backward()
            # FIXED: use clip_grad_value_ (consistent with Stage 1)
            nn.utils.clip_grad_value_(
                edge_net.parameters(), config.TRAIN['grad_clip_value'])
            optimizer.step()

            train_total += loss.item()
            train_count += 1

        train_loss = train_total / max(train_count, 1)

        # evaluate
        edge_net.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                noise_pred = dncnn(noisy)
                denoised = noisy - noise_pred
                enhanced, _ = edge_net(denoised)
                loss, _ = criterion(enhanced, clean)
                val_total += loss.item()
                val_count += 1
        val_loss = val_total / max(val_count, 1)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # save best
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            edge_path = os.path.join(args.out, f'edge_stage2_best_{timestamp}.pth')
            if prev_best is not None and os.path.exists(prev_best):
                os.remove(prev_best)
            torch.save({'model_state_dict': edge_net.state_dict()}, edge_path)
            prev_best = edge_path

        elapsed = time.time() - t0
        perc_str = f"  perc={details.get('perceptual', 0):.2e}" if 'perceptual' in details else ""
        print(f"[{epoch:3d}/{args.epochs}]  "
              f"train={train_loss:.2e}  val={val_loss:.2e}  "
              f"(recon={details['recon']:.2e}  edge={details['edge']:.2e}{perc_str})  "
              f"lr={current_lr:.2e}  {elapsed:.1f}s"
              f"{'  ★ best' if is_best else ''}")

    print(f"\nDone. Best val loss: {best_loss:.2e}")
    print(f"Best edge model: {prev_best}")


if __name__ == '__main__':
    main()
