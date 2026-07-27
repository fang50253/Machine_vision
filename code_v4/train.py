#!/usr/bin/env python
"""
train_joint_finetune.py
Stage 3: End-to-end joint fine-tuning of DnCNN + EdgeEnhance.

Pipeline:
   noisy ? [DnCNN (unfrozen, small lr)] ? denoised ? [EdgeEnhance (unfrozen)] ? enhanced

Both networks are trained jointly with a small learning rate to let them
adapt to each other after Stage 1 (DnCNN) + Stage 2 (EdgeEnhance) training.

Usage:
    python train.py --data ../Div2k/DIV2K_train_HR \
        --dncnn trained_models/dncnn_best_20260720_221351.pth \
        --edge trained_models/edge_stage2_best_20260720_221351.pth \
        --val ../Div2k/DIV2K_valid_HR
"""
import argparse, os, sys, time, random, glob, json, math
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import config
from models.dncnn import DnCNN
from models.edge_enhancer import EdgeEnhancementNetwork
from models.losses import CombinedLoss


# ?? dataset ??

class DenoiseDataset(Dataset):
    """On-the-fly patch extraction + Gaussian noise addition."""

    def __init__(self, root: str, patch_size: int = 128, sigma: float = 25.0,
                 max_samples: int | None = None,
                 deterministic: bool = False, noise_seed: int | None = None,
                 sigma_ramp_epochs: int = 0,
                 center_crop: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.sigma_target = sigma / 255.0
        self.sigma_ramp_epochs = sigma_ramp_epochs
        self.epoch = 0
        self.deterministic = deterministic
        self.noise_seed = noise_seed
        self.center_crop = center_crop
        self._fixed_sigma: float | None = None  # override for multi-sigma eval
        self.files = self._collect_images(root)
        if max_samples is not None and max_samples < len(self.files):
            self.files = self.files[:max_samples]
        tag = " (deterministic)" if deterministic else ""
        if center_crop:
            tag += " center-crop"
        print(f"Dataset [{root}]: {len(self.files)} images{tag}")

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch (for sigma ramp)."""
        self.epoch = epoch

    def set_fixed_sigma(self, sigma: float) -> None:
        self._fixed_sigma = sigma

    def clear_fixed_sigma(self) -> None:
        self._fixed_sigma = None

    @property
    def current_sigma(self) -> float:
        if self._fixed_sigma is not None:
            return self._fixed_sigma
        if self.deterministic:
            return self.sigma_target
        if self.sigma_ramp_epochs <= 0:
            max_sigma = self.sigma_target
        else:
            progress = min(self.epoch / self.sigma_ramp_epochs, 1.0)
            max_sigma = self.sigma_target * progress
        return random.random() * max_sigma

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

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT101)
            h, w = img.shape[:2]
        y = (h - self.patch_size) // 2
        x = (w - self.patch_size) // 2
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
        patch = self._center_crop(img) if self.center_crop else self._random_crop(img)
        if not self.deterministic:
            patch = self._augment(patch)
        clean = patch / 255.0

        sigma = self.current_sigma
        if self.noise_seed is not None:
            rng = np.random.RandomState(self.noise_seed + idx)
            noise = rng.randn(*clean.shape).astype(np.float32) * sigma
        else:
            noise = np.random.randn(*clean.shape).astype(np.float32) * sigma

        noisy = np.clip(clean + noise, 0.0, 1.0)
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1))
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1))
        return noisy_t, clean_t


# ?? helpers ??

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: nn.Module, ckpt_path: str, key: str | None = None) -> None:
    """Load state dict from checkpoint, handling various formats."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if key is not None and key in state:
        state = state[key]
    elif 'model_state' in state:
        state = state['model_state']
    elif 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'state_dict' in state:
        state = state['state_dict']
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


# ?? joint training ??

class JointTrainer:
    def __init__(self, dncnn: nn.Module, edge_net: nn.Module,
                 device: torch.device, out_dir: str):
        self.dncnn = dncnn.to(device)
        self.edge_net = edge_net.to(device)
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Both networks trainable
        self.dncnn.train()
        self.edge_net.train()

        # Joint optimizer with separate LR per network
        self.optimizer = optim.Adam([
            {'params': self.dncnn.parameters(), 'lr': config.TRAIN['joint_ft_lr']},
            {'params': self.edge_net.parameters(), 'lr': config.TRAIN['joint_ft_lr'] * 2},
        ], weight_decay=config.TRAIN['weight_decay'])

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.TRAIN['joint_ft_epochs'],
            eta_min=config.TRAIN['lr_min'],
        )

        # Loss: MSE + edge-aware loss
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.best_loss = float('inf')
        self._prev_best: str | None = None

    def _edge_loss(self, enhanced, clean):
        """Sobel gradient magnitude L1 loss."""
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                               dtype=torch.float32, device=enhanced.device)
        sobel_y = torch.tensor([[[[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]],
                               dtype=torch.float32, device=enhanced.device)
        g1 = torch.mean(enhanced, dim=1, keepdim=True)
        g2 = torch.mean(clean, dim=1, keepdim=True)
        g1x = nn.functional.conv2d(g1, sobel_x, padding=1)
        g1y = nn.functional.conv2d(g1, sobel_y, padding=1)
        g2x = nn.functional.conv2d(g2, sobel_x, padding=1)
        g2y = nn.functional.conv2d(g2, sobel_y, padding=1)
        return nn.functional.l1_loss(
            torch.sqrt(g1x**2 + g1y**2 + 1e-6),
            torch.sqrt(g2x**2 + g2y**2 + 1e-6),
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader,
                 eval_sigmas: list[float] | None = None
                 ) -> tuple[float, dict[str, float]]:
        """
        Run validation loop. Supports multi-sigma evaluation.
        Returns (avg_loss, psnr_dict).
        """
        self.dncnn.eval()
        self.edge_net.eval()
        dataset = loader.dataset
        orig_bs = loader.batch_size
        has_fixed_sigma = hasattr(dataset, 'set_fixed_sigma') and hasattr(dataset, 'clear_fixed_sigma')

        if eval_sigmas is None:
            eval_sigmas = [None]

        all_losses: list[float] = []
        psnr_dict: dict[str, float] = {}

        for sigma in eval_sigmas:
            if sigma is not None and has_fixed_sigma:
                dataset.set_fixed_sigma(sigma / 255.0)
            elif has_fixed_sigma:
                dataset.clear_fixed_sigma()

            temp_loader = DataLoader(
                dataset, batch_size=orig_bs, shuffle=False,
                num_workers=0, pin_memory=False,
            )

            total, count, total_mse = 0.0, 0, 0.0
            for noisy, clean in temp_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                noise_pred = self.dncnn(noisy)
                denoised = noisy - noise_pred
                enhanced, _ = self.edge_net(denoised)
                loss = self.mse(enhanced, clean)
                total += loss.item()
                total_mse += nn.functional.mse_loss(enhanced, clean).item() * clean.size(0)
                count += 1

            avg_loss = total / max(count, 1)
            all_losses.append(avg_loss)

            if sigma is not None:
                avg_mse = total_mse / max(count * orig_bs, 1)
                psnr_val = 20 * math.log10(1.0 / math.sqrt(avg_mse)) if avg_mse > 1e-10 else 100.0
                key = f'psnr_{sigma:.0f}'
                psnr_dict[key] = round(psnr_val, 2)

        if has_fixed_sigma:
            dataset.clear_fixed_sigma()

        self.dncnn.train()
        self.edge_net.train()
        return sum(all_losses) / max(len(all_losses), 1), psnr_dict

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, sigma_target: float = 25.0):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"\n{'='*60}")
        print("Joint Fine-tuning: DnCNN + EdgeEnhance (end-to-end)")
        print(f"DnCNN lr={config.TRAIN['joint_ft_lr']}  "
              f"Edge lr={config.TRAIN['joint_ft_lr'] * 2}")
        print(f"epochs={epochs}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train with per-batch sigma (override dataset's per-sample noise)
            train_total, train_count = 0.0, 0
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                # Per-batch uniform sigma: all samples share same noise level
                sigma_val = random.uniform(0, sigma_target)
                sigma_t = torch.tensor(sigma_val / 255.0,
                                       device=self.device).view(1, 1, 1, 1)
                noise = torch.randn_like(clean) * sigma_t
                noisy = torch.clamp(clean + noise, 0.0, 1.0)

                self.optimizer.zero_grad()

                # Forward through both networks
                noise_pred = self.dncnn(noisy)
                denoised = noisy - noise_pred
                enhanced, _ = self.edge_net(denoised)

                # Loss: MSE + edge preservation
                mse_loss = self.mse(enhanced, clean)
                edge_loss = self._edge_loss(enhanced, clean)
                total_loss = mse_loss + 0.1 * edge_loss

                total_loss.backward()

                # Gradient clipping for both networks
                nn.utils.clip_grad_value_(self.dncnn.parameters(),
                                          config.TRAIN['grad_clip_value'])
                nn.utils.clip_grad_value_(self.edge_net.parameters(),
                                          config.TRAIN['grad_clip_value'])

                self.optimizer.step()

                train_total += total_loss.item()
                train_count += 1

            train_loss = train_total / max(train_count, 1)
            val_loss, psnr_dict = self.evaluate(val_loader)
            self.scheduler.step()

            # Save best
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                # Save joint checkpoint
                joint_path = os.path.join(self.out_dir,
                                          f'joint_best_{timestamp}.pth')
                if self._prev_best is not None and os.path.exists(self._prev_best):
                    os.remove(self._prev_best)
                torch.save({
                    'dncnn_state_dict': self.dncnn.state_dict(),
                    'edge_state_dict': self.edge_net.state_dict(),
                }, joint_path)
                self._prev_best = joint_path

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']
            psnr_str = '  '.join(f'{k}={v:.1f}dB' for k, v in psnr_dict.items())
            print(f"[{epoch:3d}/{epochs}]  "
                  f"train={train_loss:.2e}  val={val_loss:.2e}  "
                  f"lr={lr:.2e}  {psnr_str}  {elapsed:.1f}s"
                  f"{'  ★ best' if is_best else ''}")

            # Emit structured metrics for pipeline
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": round(lr, 8),
            }
            if psnr_dict:
                metrics.update(psnr_dict)
            if not sys.stdout.isatty():
                print("__METRICS__" + json.dumps(metrics))

        print(f"\nDone. Best val loss: {self.best_loss:.2e}")
        print(f"Best joint model: {self._prev_best}")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: Joint fine-tuning of DnCNN + EdgeEnhance')
    parser.add_argument('--data', required=True,
                        help='Path to training images folder')
    parser.add_argument('--val', default=None,
                        help='Path to validation images (optional)')
    parser.add_argument('--dncnn', required=True,
                        help='Path to DnCNN best model (.pth)')
    parser.add_argument('--edge', required=True,
                        help='Path to EdgeEnhance best model (.pth)')
    parser.add_argument('--epochs', type=int, default=config.TRAIN['joint_ft_epochs'])
    parser.add_argument('--batch-size', type=int, default=config.TRAIN['batch_size'])
    parser.add_argument('--lr', type=float, default=None,
                        help='DnCNN fine-tuning lr (default: config joint_ft_lr)')
    parser.add_argument('--seed', type=int, default=config.TRAIN['seed'])
    parser.add_argument('--out', default=config.MODEL_SAVE_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device}  |  CUDA: {torch.cuda.is_available()}")
    print(f"DnCNN model: {os.path.basename(args.dncnn)}")
    print(f"Edge model:  {os.path.basename(args.edge)}")

    # Override joint_ft_lr from config if provided
    if args.lr is not None:
        config.TRAIN['joint_ft_lr'] = args.lr

    # ?? datasets ??
    full_dataset = DenoiseDataset(
        root=args.data, patch_size=config.TRAIN['patch_size'],
        sigma=config.TRAIN['noise_sigma'])

    if args.val is not None and os.path.isdir(args.val):
        train_dataset = full_dataset
        val_dataset = DenoiseDataset(
            root=args.val, patch_size=config.TRAIN['patch_size'],
            sigma=config.TRAIN['noise_sigma'],
            sigma_ramp_epochs=0,
            deterministic=True, noise_seed=42,
            center_crop=True)
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

    # ?? models ??
    # Try loading deep variant first, fall back to standard
    dncnn = DnCNN(channels=config.CHANNELS, num_layers=config.NUM_LAYERS,
                  num_features=config.NUM_FEATURES)
    try:
        load_checkpoint(dncnn, args.dncnn, key='model_state')
        print("DnCNN loaded: standard arch")
    except Exception as e:
        print(f"Standard DnCNN load failed ({e}), trying deep arch...")
        raise

    edge_net = EdgeEnhancementNetwork(in_channels=3, base_channels=64)
    try:
        load_checkpoint(edge_net, args.edge, key='model_state_dict')
        print("EdgeEnhance loaded")
    except Exception as e:
        print(f"EdgeEnhance load error: {e}")
        raise

    # ?? trainer ??
    trainer = JointTrainer(dncnn, edge_net, device, args.out)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)


if __name__ == '__main__':
    main()
