#!/usr/bin/env python
"""
train.py
Train a DnCNN model on a dataset (e.g. DIV2K).

Improvements over baseline:
  - clip_grad_value_ (per-param) instead of clip_grad_norm_ (total norm)
  - CosineAnnealingLR (stable decay) instead of WarmRestarts
  - Stronger data augmentation: color jitter, random blur, random scale
  - Curriculum learning: sigma ramps from 0 to target over N epochs
  - Optional perceptual + frequency loss (CombinedLoss)
  - Optional channel attention (USE_ATTENTION)

Usage:
    python train.py --data /path/to/train/images --val /path/to/val/images
    python train.py --data /path/to/images              # auto-splits 5% for val
"""
import argparse, os, sys, time, math, random, glob, json
from datetime import datetime

# ensure project root is on sys.path (for running from training/ subdir)
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
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from models.dncnn import DnCNN, create_dncnn_deep
from models.losses import CombinedLoss


# ── helpers ──

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two uint8 [0,255] images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ── dataset ──

class DenoiseDataset(Dataset):
    """
    On-the-fly patch extraction + Gaussian noise addition + augmentation.

    Each item: (noisy_patch, clean_patch)  both in [0,1].

    Augmentations:
      - Random 90° rotation + horizontal flip (baseline)
      - Color jitter (brightness, contrast, saturation)
      - Random Gaussian blur
      - Random scaling (multi-scale training)
      - Curriculum learning: sigma ramps from 0 to target
    """

    def __init__(self, root: str, patch_size: int = 128, sigma: float = 25.0,
                 max_samples: int | None = None, epoch: int = 0,
                 sigma_ramp_epochs: int = 0,
                 deterministic: bool = False, noise_seed: int | None = None,
                 center_crop: bool = False):
        """
        Args:
            deterministic: If True, disable random augmentations (for validation).
            noise_seed: If set, use fixed noise per image index (reproducible).
            center_crop: If True, use center crop instead of random crop
                         (makes val loss reproducible across epochs).
        """
        super().__init__()
        self.patch_size = patch_size
        self.sigma_target = sigma / 255.0
        self.sigma_ramp_epochs = sigma_ramp_epochs
        self.epoch = epoch
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
        """Update current epoch (for curriculum learning)."""
        self.epoch = epoch

    def set_fixed_sigma(self, sigma: float) -> None:
        self._fixed_sigma = sigma

    def clear_fixed_sigma(self) -> None:
        self._fixed_sigma = None

    @property
    def current_sigma(self) -> float:
        """Return noise sigma for current sample.

        - If _fixed_sigma set (multi-sigma eval): return that value.
        - Training (deterministic=False): Uniform[0, max_sigma].
          max_sigma linearly ramps from 0 to sigma_target over ramp_epochs.
        - Validation (deterministic=True): fixed sigma_target.
        """
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

    # ── file collection ──
    @staticmethod
    def _collect_images(root: str) -> list[str]:
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
        files: list[str] = []
        for ext in exts:
            pattern = os.path.join(root, '**', ext)
            files.extend(glob.glob(pattern, recursive=True))
        files = sorted(set(files))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        """Randomly crop a patch of size patch_size×patch_size."""
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_REFLECT101)
            h, w = img.shape[:2]
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)
        return img[y:y + self.patch_size, x:x + self.patch_size]

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        """Center crop a patch — deterministic location for reproducible val."""
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_REFLECT101)
            h, w = img.shape[:2]
        y = (h - self.patch_size) // 2
        x = (w - self.patch_size) // 2
        return img[y:y + self.patch_size, x:x + self.patch_size]

    def _augment_basic(self, patch: np.ndarray) -> np.ndarray:
        """Random 90° rotation + horizontal flip."""
        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
        k = random.randint(0, 3)
        if k:
            patch = np.rot90(patch, k).copy()
        return patch

    def _augment_color(self, patch: np.ndarray, strength: float = 0.05) -> np.ndarray:
        """Color jitter: brightness, contrast, saturation."""
        if strength <= 0:
            return patch
        # brightness
        if random.random() < 0.5:
            b = 1.0 + random.uniform(-strength, strength)
            patch = np.clip(patch * b, 0, 255)
        # contrast
        if random.random() < 0.5:
            c = 1.0 + random.uniform(-strength, strength)
            mean = np.mean(patch, axis=(0, 1), keepdims=True)
            patch = np.clip((patch - mean) * c + mean, 0, 255)
        # saturation (HSV S channel)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            s = 1.0 + random.uniform(-strength * 2, strength * 2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
            patch = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        return patch

    def _augment_blur(self, patch: np.ndarray, prob: float = 0.1) -> np.ndarray:
        """Random Gaussian blur."""
        if prob <= 0 or random.random() >= prob:
            return patch
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        return cv2.GaussianBlur(patch, (ksize, ksize), sigma)

    def _augment_scale(self, patch: np.ndarray) -> np.ndarray:
        """Multi-scale: random resize back to patch_size."""
        scale = random.uniform(*config.TRAIN['scale_range'])
        if scale >= 0.99:
            return patch
        h, w = patch.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx: int):
        img = self._load_image(self.files[idx])
        patch = self._center_crop(img) if self.center_crop else self._random_crop(img)

        if not self.deterministic:
            # Augmentations only for training
            if random.random() < config.TRAIN.get('scale_aug_prob', 0):
                patch = self._augment_scale(patch)
            patch = self._augment_basic(patch)
            patch = self._augment_color(patch, config.TRAIN.get('aug_color_jitter', 0))
            patch = self._augment_blur(patch, config.TRAIN.get('aug_gaussian_blur', 0))

        # Normalise to [0,1]
        clean = patch / 255.0

        # Add Gaussian noise (curriculum sigma)
        sigma = self.current_sigma

        if self.noise_seed is not None:
            # Deterministic noise: same image → same noise every epoch
            rng = np.random.RandomState(self.noise_seed + idx)
            noise = rng.randn(*clean.shape).astype(np.float32) * sigma
        else:
            noise = np.random.randn(*clean.shape).astype(np.float32) * sigma

        noisy = np.clip(clean + noise, 0.0, 1.0)

        # HWC → CHW
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1))
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1))
        return noisy_t, clean_t


# ── training ──

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, out_dir: str,
                 use_perceptual: bool = False, use_frequency: bool = False):
        self.model = model.to(device)
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAIN['lr'],
            weight_decay=config.TRAIN['weight_decay'],
        )

        # Use CosineAnnealingLR for stable decay (no warm restarts)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.TRAIN['epochs'],
            eta_min=config.TRAIN['lr_min'],
        )

        # Loss: CombinedLoss with optional perceptual + frequency
        w_perc = config.TRAIN['perceptual_weight'] if use_perceptual else 0.0
        w_freq = config.TRAIN['frequency_weight'] if use_frequency else 0.0
        self.criterion = CombinedLoss(
            w_mse=1.0, w_perc=w_perc, w_freq=w_freq, device=device,
        )

        self.best_loss = float('inf')
        self._prev_best: str | None = None
        self.history: dict[str, list[float]] = {'train_loss': [], 'val_loss': []}

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'config': {k: v for k, v in config.TRAIN.items()},
        }, path)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader,
                 eval_sigmas: list[float] | None = None
                 ) -> tuple[float, dict[str, float]]:
        """
        Run validation loop.

        Args:
            eval_sigmas: If set, evaluate at each sigma level (e.g. [25, 50, 75])
                         and return per-sigma PSNR. When None, use dataset's
                         default sigma.
        Returns:
            (avg_loss, psnr_dict) — avg_loss is mean across all sigma levels.
        """
        self.model.eval()
        dataset = loader.dataset
        orig_bs = loader.batch_size
        has_fixed_sigma = hasattr(dataset, 'set_fixed_sigma') and hasattr(dataset, 'clear_fixed_sigma')

        if eval_sigmas is None:
            eval_sigmas = [None]  # use dataset default

        all_losses: list[float] = []
        psnr_dict: dict[str, float] = {}

        for sigma in eval_sigmas:
            if sigma is not None and has_fixed_sigma:
                dataset.set_fixed_sigma(sigma / 255.0)
            elif has_fixed_sigma:
                dataset.clear_fixed_sigma()

            # Use num_workers=0 so fixed_sigma propagates (multiprocess workers
            # don't see attribute changes after fork)
            temp_loader = DataLoader(
                dataset, batch_size=orig_bs, shuffle=False,
                num_workers=0, pin_memory=False,
            )

            total_loss, total_mse, count = 0.0, 0.0, 0
            for noisy, clean in temp_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                noise_pred = self.model(noisy)
                denoised = noisy - noise_pred
                loss, _ = self.criterion(denoised, clean)
                total_loss += loss.item()
                total_mse += F.mse_loss(denoised, clean).item() * clean.size(0)
                count += clean.size(0)

            avg_loss = total_loss / max(len(temp_loader), 1)
            all_losses.append(avg_loss)

            if sigma is not None:
                avg_mse = total_mse / max(count, 1)
                psnr_val = 20 * math.log10(1.0 / math.sqrt(avg_mse)) if avg_mse > 1e-10 else 100.0
                key = f'psnr_{sigma:.0f}'
                psnr_dict[key] = round(psnr_val, 2)

        if has_fixed_sigma:
            dataset.clear_fixed_sigma()

        return sum(all_losses) / max(len(all_losses), 1), psnr_dict

    def train_one_epoch(self, loader: DataLoader, epoch: int,
                         sigma_ramp_epochs: int = 0,
                         sigma_target: float = 25.0) -> float:
        self.model.train()
        total, count = 0.0, 0

        # Per-batch sigma: all samples in one batch share the same sigma
        # (much lower gradient variance than per-sample random sigma).
        if sigma_ramp_epochs <= 0:
            max_sigma = sigma_target
        else:
            progress = min(epoch / sigma_ramp_epochs, 1.0)
            max_sigma = sigma_target * progress

        for noisy, clean in loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)

            # Override noisy with per-batch uniform sigma noise.
            # The dataset already added per-sample noise, but we discard it
            # and re-add with a single sigma for the whole batch → stable grads.
            sigma_val = random.uniform(0, max_sigma)  # 0 … sigma_target
            sigma_t = torch.tensor(sigma_val / 255.0,
                                   device=self.device).view(1, 1, 1, 1)
            noise = torch.randn_like(clean) * sigma_t
            noisy = torch.clamp(clean + noise, 0.0, 1.0)

            self.optimizer.zero_grad()
            noise_pred = self.model(noisy)
            denoised = noisy - noise_pred
            loss, _ = self.criterion(denoised, clean)
            loss.backward()
            # FIXED: clip_grad_value_ (per-param) per DnCNN paper
            nn.utils.clip_grad_value_(
                self.model.parameters(), config.TRAIN['grad_clip_value'])
            self.optimizer.step()
            total += loss.item()
            count += 1
        return total / max(count, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, eval_sigmas: list[float] | None = None
            ) -> dict[str, list[float]]:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Read sigma_ramp_epochs from dataset (respects CLI --sigma-ramp)
        sigma_ramp_epochs = getattr(train_loader.dataset, 'sigma_ramp_epochs',
                                    config.TRAIN.get('sigma_ramp_epochs', 0))
        # Train sigma covers the full eval range so the model sees high noise
        sigma_target = max(eval_sigmas) if eval_sigmas else config.TRAIN['noise_sigma']

        print(f"\n{'='*60}")
        print(f"Training DnCNN  |  device: {self.device}")
        print(f"epochs={epochs}  lr={config.TRAIN['lr']}  "
              f"sigma=Uniform[0,{sigma_target}] per-batch  "
              f"patches={config.TRAIN['patch_size']}×{config.TRAIN['patch_size']}")
        if sigma_ramp_epochs > 0:
            print(f"curriculum: max sigma ramps over {sigma_ramp_epochs} epochs")
        if eval_sigmas:
            print(f"eval sigmas: {eval_sigmas}")
        if config.TRAIN.get('perceptual_weight', 0) > 0:
            print(f"perceptual loss: weight={config.TRAIN['perceptual_weight']}")
        if config.TRAIN.get('frequency_weight', 0) > 0:
            print(f"frequency loss: weight={config.TRAIN['frequency_weight']}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Update curriculum sigma for training set only
            if hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)

            train_loss = self.train_one_epoch(
                train_loader, epoch=epoch,
                sigma_ramp_epochs=sigma_ramp_epochs,
                sigma_target=sigma_target)
            val_loss, psnr_dict = self.evaluate(val_loader, eval_sigmas=eval_sigmas)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # checkpoint (based on primary val loss)
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                best_path = os.path.join(self.out_dir,
                                         f'dncnn_best_{timestamp}.pth')
                if self._prev_best is not None and os.path.exists(self._prev_best):
                    os.remove(self._prev_best)
                self.save_checkpoint(best_path)
                self._prev_best = best_path

            # Log max sigma for current epoch (per-batch sigma upper bound)
            sigma_info = ""
            current_sigma_val = None
            if sigma_ramp_epochs <= 0:
                max_sigma_display = sigma_target
            else:
                progress = min(epoch / sigma_ramp_epochs, 1.0)
                max_sigma_display = sigma_target * progress
            sigma_info = f"  σ_max={max_sigma_display:.1f}"
            current_sigma_val = round(max_sigma_display, 1)

            elapsed = time.time() - t0
            print(f"[{epoch:3d}/{epochs}]  "
                  f"train={train_loss:.2e}  val={val_loss:.2e}  "
                  f"lr={current_lr:.2e}{sigma_info}  "
                  f"{elapsed:.1f}s  {'★ best' if is_best else ''}")

            # —— emit structured metrics for web UI ——
            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": round(current_lr, 8),
            }
            if current_sigma_val is not None:
                metrics["sigma"] = current_sigma_val
            if psnr_dict:
                metrics.update(psnr_dict)
            # Only emit JSON line when run under server (stdout is pipe),
            # not in interactive terminal (one readable line/epoch is enough).
            if not sys.stdout.isatty():
                print("__METRICS__" + json.dumps(metrics))

        # final
        final_path = os.path.join(self.out_dir, f'dncnn_final_{timestamp}.pth')
        self.save_checkpoint(final_path)
        print(f"\nDone. Best val loss: {self.best_loss:.2e}")
        print(f"Best checkpoint: {best_path}")
        return self.history


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description='Train DnCNN')
    parser.add_argument('--data', required=True,
                        help='Path to training images (folder)')
    parser.add_argument('--val', default=None,
                        help='Path to validation images (optional)')
    parser.add_argument('--epochs', type=int, default=config.TRAIN['epochs'])
    parser.add_argument('--batch-size', type=int, default=config.TRAIN['batch_size'])
    parser.add_argument('--patch-size', type=int, default=config.TRAIN['patch_size'])
    parser.add_argument('--sigma', type=float, default=config.TRAIN['noise_sigma'],
                        help='Gaussian noise std')
    parser.add_argument('--lr', type=float, default=config.TRAIN['lr'])
    parser.add_argument('--seed', type=int, default=config.TRAIN['seed'])
    parser.add_argument('--out', default=config.MODEL_SAVE_DIR)
    parser.add_argument('--deep', action='store_true',
                        help='Use deeper DnCNN (20 layers / 96 channels + attention)')
    parser.add_argument('--perceptual', action='store_true',
                        help='Enable VGG perceptual loss')
    parser.add_argument('--frequency', action='store_true',
                        help='Enable FFT frequency loss')
    parser.add_argument('--sigma-ramp', type=int, default=None,
                        help='Curriculum sigma ramp epochs (default: config value)')
    parser.add_argument('--eval-sigmas', type=str, default=None,
                        help='Comma-separated sigma levels for multi-noise eval, e.g. "25,50,75,100"')
    args = parser.parse_args()

    set_seed(args.seed)
    device = config.DEVICE
    print(f"Device: {device}  |  CUDA: {torch.cuda.is_available()}")

    # ── datasets ──
    sigma_ramp = args.sigma_ramp if args.sigma_ramp is not None else config.TRAIN.get('sigma_ramp_epochs', 0)

    full_dataset = DenoiseDataset(
        root=args.data,
        patch_size=args.patch_size,
        sigma=args.sigma,
        sigma_ramp_epochs=sigma_ramp,
    )

    if args.val is not None and os.path.isdir(args.val):
        train_dataset = full_dataset
        val_dataset = DenoiseDataset(
            root=args.val,
            patch_size=args.patch_size,
            sigma=args.sigma,
            sigma_ramp_epochs=0,          # no ramp → always σ=25
            deterministic=True,
            noise_seed=42,                # fixed seed → same image always gets same noise
            center_crop=True,             # same patch location every epoch → stable val loss
        )
    else:
        # auto-split
        val_size = max(1, int(len(full_dataset) * config.TRAIN['val_split']))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"Auto-split: train={train_size}  val={val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config.TRAIN['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.TRAIN['num_workers'], pin_memory=True,
    )

    # ── model ──
    if args.deep:
        model = create_dncnn_deep()
        print("Using DEEP DnCNN (20 layers / 96 channels + attention)")
    else:
        model = DnCNN(
            channels=config.CHANNELS,
            num_layers=config.NUM_LAYERS,
            num_features=config.NUM_FEATURES,
            use_attention=config.USE_ATTENTION,
        )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    eval_sigmas: list[float] | None = None
    if args.eval_sigmas:
        eval_sigmas = [float(s.strip()) for s in args.eval_sigmas.split(',') if s.strip()]

    trainer = Trainer(model, device, args.out,
                      use_perceptual=args.perceptual,
                      use_frequency=args.frequency)
    trainer.fit(train_loader, val_loader, epochs=args.epochs, eval_sigmas=eval_sigmas)


if __name__ == '__main__':
    main()
