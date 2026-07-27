"""
models/joint_trainer.py
Joint training module: DnCNN denoising + EdgeEnhance enhancement end-to-end.

Usage:
    from models.joint_trainer import JointModel, JointTrainer

    model = JointModel(dncnn, edge_net)
    trainer = JointTrainer(model, device, out_dir)
    trainer.fit(train_loader, val_loader, epochs=50)
"""
import os, time, math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from models.dncnn import DnCNN
from models.edge_enhancer import EdgeEnhancementNetwork


class JointModel(nn.Module):
    """
    End-to-end: DnCNN denoise → EdgeEnhance enhance.

    Forward returns:
        denoised: DnCNN output (noise removed)
        enhanced: EdgeEnhance output (edges sharpened)
    """

    def __init__(self, dncnn: DnCNN, edge_net: EdgeEnhancementNetwork):
        super().__init__()
        self.dncnn = dncnn
        self.edge_net = edge_net

    def forward(self, x: torch.Tensor):
        # DnCNN residual learning: predict noise, subtract
        noise_pred = self.dncnn(x)
        denoised = x - noise_pred

        # Edge enhancement on denoised result
        enhanced, edge_features = self.edge_net(denoised)

        return denoised, enhanced


class JointLoss(nn.Module):
    """
    Combined loss for joint training:
        L = λ_d * L_denoise + λ_e * L_enhance + λ_edge * L_edgePreserve

    Where:
        L_denoise = MSE(denoised, clean)
        L_enhance = L1(enhanced, clean)  ← L1 preserves edges better
        L_edgePreserve = Sobel gradient loss between enhanced and clean
    """

    def __init__(self, lambda_d=0.5, lambda_e=1.0, lambda_edge=0.3):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_e = lambda_e
        self.lambda_edge = lambda_edge

        # Sobel kernels
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _edge_loss(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Gradient magnitude L1 loss for edge preservation."""
        g1 = torch.mean(img1, dim=1, keepdim=True)
        g2 = torch.mean(img2, dim=1, keepdim=True)

        # ensure sobel kernels on same device as input
        device = img1.device
        sx = self.sobel_x.to(device)
        sy = self.sobel_y.to(device)
        g1x = F.conv2d(g1, sx, padding=1)
        g1y = F.conv2d(g1, sy, padding=1)
        g1_mag = torch.sqrt(g1x ** 2 + g1y ** 2 + 1e-6)

        g2x = F.conv2d(g2, sx, padding=1)
        g2y = F.conv2d(g2, sy, padding=1)
        g2_mag = torch.sqrt(g2x ** 2 + g2y ** 2 + 1e-6)

        return F.l1_loss(g1_mag, g2_mag)

    def forward(self, denoised, enhanced, clean):
        d_loss = F.mse_loss(denoised, clean)
        e_loss = F.l1_loss(enhanced, clean)
        edge_loss = self._edge_loss(enhanced, clean)

        total = self.lambda_d * d_loss + self.lambda_e * e_loss + self.lambda_edge * edge_loss
        return total, {'denoise': d_loss.item(), 'enhance': e_loss.item(), 'edge': edge_loss.item()}


class JointTrainer:
    """
    Trainer for JointModel.

    Saves three types of checkpoints at best validation loss:
        - dncnn_best_joint_{ts}.pth  (DnCNN sub-network, loadable by benchmark.py)
        - edge_best_joint_{ts}.pth   (EdgeEnhance sub-network, loadable by benchmark_full.py)
        - joint_best_{ts}.pth        (full combined checkpoint)
    """

    def __init__(self, model: JointModel, device: torch.device, out_dir: str,
                 lambda_d=0.5, lambda_e=1.0, lambda_edge=0.3):
        self.model = model.to(device)
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Optimise ALL parameters (both sub-networks)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAIN['lr'],
            weight_decay=config.TRAIN['weight_decay'],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.TRAIN['epochs'],
            eta_min=config.TRAIN['lr_min'],
        )
        self.criterion = JointLoss(lambda_d, lambda_e, lambda_edge)
        self.best_loss = float('inf')
        self._prev_best: dict[str, str | None] = {
            'joint': None, 'dncnn': None, 'edge': None,
        }
        self.history: dict[str, list] = {'train_loss': [], 'val_loss': [],
                                         'denoise_loss': [], 'enhance_loss': [], 'edge_loss': []}

    def save_sub_models(self, tag: str) -> None:
        """Save DnCNN and EdgeEnhance checkpoints separately."""
        dncnn_path = os.path.join(self.out_dir, f'dncnn_best_joint_{tag}.pth')
        edge_path = os.path.join(self.out_dir, f'edge_best_joint_{tag}.pth')

        # DnCNN state
        torch.save({
            'model_state': self.model.dncnn.state_dict(),
            'config': {
                'channels': self.model.dncnn.channels,
                'num_layers': self.model.dncnn.num_layers,
                'num_features': self.model.dncnn.num_features,
            },
        }, dncnn_path)

        # EdgeEnhance state
        torch.save({
            'model_state_dict': self.model.edge_net.state_dict(),
        }, edge_path)

        return dncnn_path, edge_path

    def save_checkpoint(self, tag: str) -> str:
        """Save full joint checkpoint."""
        path = os.path.join(self.out_dir, f'joint_best_{tag}.pth')
        torch.save({
            'model_state_dncnn': self.model.dncnn.state_dict(),
            'model_state_edge': self.model.edge_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'config': {k: v for k, v in config.TRAIN.items()},
        }, path)
        return path

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            denoised, enhanced = self.model(noisy)
            loss, _ = self.criterion(denoised, enhanced, clean)
            total += loss.item()
            count += 1
        return total / max(count, 1)

    def train_one_epoch(self, loader) -> dict:
        self.model.train()
        total, count = 0.0, 0
        accum = {'denoise': 0.0, 'enhance': 0.0, 'edge': 0.0}
        for noisy, clean in loader:
            noisy, clean = noisy.to(self.device), clean.to(self.device)

            self.optimizer.zero_grad()
            denoised, enhanced = self.model(noisy)
            loss, details = self.criterion(denoised, enhanced, clean)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), config.TRAIN['grad_clip_norm'])
            self.optimizer.step()

            total += loss.item()
            count += 1
            for k in accum:
                accum[k] += details[k]

        avg = total / max(count, 1)
        self.history['train_loss'].append(avg)
        for k in accum:
            self.history[f'{k}_loss'].append(accum[k] / max(count, 1))
        return avg

    def fit(self, train_loader, val_loader, epochs: int, print_freq: int = 1):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        n_d = sum(p.numel() for p in self.model.dncnn.parameters())
        n_e = sum(p.numel() for p in self.model.edge_net.parameters())

        print(f"\n{'='*70}")
        print(f"Joint Training  |  device: {self.device}")
        print(f"DnCNN: {n_d:,} params  |  EdgeEnhance: {n_e:,} params  |  Total: {n_d + n_e:,}")
        print(f"epochs={epochs}  lr={config.TRAIN['lr']}  "
              f"sigma={config.TRAIN['noise_sigma']}  "
              f"patches={config.TRAIN['patch_size']}×{config.TRAIN['patch_size']}")
        print(f"Loss weights: λ_d={self.criterion.lambda_d}  λ_e={self.criterion.lambda_e}  λ_edge={self.criterion.lambda_edge}")
        print(f"{'='*70}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['val_loss'].append(val_loss)

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                # delete previous bests before saving new ones
                for k, p in self._prev_best.items():
                    if p is not None and os.path.exists(p):
                        os.remove(p)
                self.save_checkpoint(timestamp)
                dncnn_p, edge_p = self.save_sub_models(timestamp)
                self._prev_best = {
                    'joint': os.path.join(self.out_dir, f'joint_best_{timestamp}.pth'),
                    'dncnn': dncnn_p,
                    'edge': edge_p,
                }
                best_note = f"★ best  [saved: {os.path.basename(dncnn_p)}, {os.path.basename(edge_p)}]"
            else:
                best_note = ''

            elapsed = time.time() - t0
            if epoch % print_freq == 0 or is_best:
                d_l = self.history['denoise_loss'][-1]
                e_l = self.history['enhance_loss'][-1]
                eg_l = self.history['edge_loss'][-1]
                print(f"[{epoch:3d}/{epochs}]  "
                      f"train={train_loss:.2e}  val={val_loss:.2e}  "
                      f"(d={d_l:.2e}  e={e_l:.2e}  edge={eg_l:.2e})  "
                      f"lr={current_lr:.2e}  "
                      f"{elapsed:.1f}s  {best_note}")

        # final save
        final_path = os.path.join(self.out_dir, f'joint_final_{timestamp}.pth')
        self.save_checkpoint(timestamp)
        print(f"\nDone. Best val loss: {self.best_loss:.2e}")
        print(f"Best joint: joint_best_{timestamp}.pth")
        print(f"Best DnCNN: dncnn_best_joint_{timestamp}.pth")
        print(f"Best Edge:  edge_best_joint_{timestamp}.pth")
        return self.history


def create_joint_model(device=None,
                       dncnn_pretrained=None, edge_pretrained=None,
                       freeze_dncnn=False) -> JointModel:
    """
    Factory: creates JointModel, optionally loading pre-trained weights.

    Args:
        dncnn_pretrained: Path to DnCNN .pth checkpoint (optional)
        edge_pretrained:  Path to EdgeEnhance .pth checkpoint (optional)
        freeze_dncnn:     If True, freeze DnCNN weights (train only Edge)
    """
    dncnn = DnCNN(channels=config.CHANNELS, num_layers=config.NUM_LAYERS,
                  num_features=config.NUM_FEATURES)
    edge_net = EdgeEnhancementNetwork(in_channels=config.CHANNELS, base_channels=64)

    if dncnn_pretrained:
        state = torch.load(dncnn_pretrained, map_location='cpu', weights_only=True)
        if 'model_state' in state:
            state = state['model_state']
        if all(k.startswith('module.') for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        dncnn.load_state_dict(state, strict=False)
        print(f"  [Joint] Loaded DnCNN from: {dncnn_pretrained}")

    if edge_pretrained:
        state = torch.load(edge_pretrained, map_location='cpu', weights_only=True)
        for key in ('model_state_dict', 'state_dict'):
            if key in state:
                state = state[key]
                break
        edge_net.load_state_dict(state, strict=False)
        print(f"  [Joint] Loaded EdgeEnhance from: {edge_pretrained}")

    model = JointModel(dncnn, edge_net)

    if freeze_dncnn:
        for p in model.dncnn.parameters():
            p.requires_grad = False
        print("  [Joint] DnCNN frozen (training Edge only)")

    if device is not None:
        model = model.to(device)

    return model
