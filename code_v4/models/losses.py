"""
models/losses.py
Advanced loss functions for image denoising / enhancement.

PerceptualLoss  — VGG16-based feature reconstruction loss
FrequencyLoss   — FFT-based frequency-domain loss
CombinedLoss    — MSE + Perceptual + Frequency (configurable weights)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Perceptual Loss ──

class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual (feature reconstruction) loss.

    Extracts features from layers relu1_2, relu2_2, relu3_3, relu4_3
    and computes L1 loss between feature maps of output and target.

    Reference: Johnson et al. "Perceptual Losses for Real-Time Style Transfer"
    """
    def __init__(self, device: torch.device = None,
                 weights: list[float] | None = None):
        super().__init__()
        # Load pretrained VGG16 and freeze
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Extract feature layers up to conv4_3
        self.layers = nn.Sequential(*list(vgg.features)[:23])
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.layer_indices = [3, 8, 15, 22]  # last index per layer group

        # Per-layer loss weights (match spatial resolution importance)
        if weights is None:
            self.weights = [1.0, 0.8, 0.6, 0.4]
        else:
            self.weights = weights

        # Normalization: VGG was trained on ImageNet mean/std
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])
                             .view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])
                             .view(1, 3, 1, 1))

        if device is not None:
            self.to(device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [0,1] to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: (B,3,H,W) in [0,1] range
            target: (B,3,H,W) in [0,1] range
        Returns:
            Scalar perceptual loss
        """
        # Normalize inputs
        out_norm = self._normalize(output)
        tgt_norm = self._normalize(target)

        loss = 0.0
        for idx, weight in zip(self.layer_indices, self.weights):
            # Forward up to this layer
            out_feat = self.layers[:idx + 1](out_norm)
            tgt_feat = self.layers[:idx + 1](tgt_norm)
            loss += weight * F.l1_loss(out_feat, tgt_feat)

        return loss


# ── Frequency Loss ──

class FrequencyLoss(nn.Module):
    """
    FFT-based frequency domain loss.

    Encourages the output to match the target's frequency spectrum,
    which helps preserve high-frequency details (edges, textures).

    L_freq = ||FFT(output) - FFT(target)||_1
    """
    def __init__(self, weight_mag: float = 1.0, weight_phase: float = 0.1):
        super().__init__()
        self.weight_mag = weight_mag
        self.weight_phase = weight_phase

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: (B,3,H,W) tensor
            target: (B,3,H,W) tensor
        Returns:
            Scalar frequency loss
        """
        # Compute FFT for each channel
        out_fft = torch.fft.fft2(output, norm='ortho')
        tgt_fft = torch.fft.fft2(target, norm='ortho')

        # Magnitude and phase
        out_mag = torch.abs(out_fft)
        tgt_mag = torch.abs(tgt_fft)
        out_phase = torch.angle(out_fft)
        tgt_phase = torch.angle(tgt_fft)

        mag_loss = F.l1_loss(out_mag, tgt_mag)
        phase_loss = F.l1_loss(out_phase, tgt_phase)

        return self.weight_mag * mag_loss + self.weight_phase * phase_loss


# ── Combined Loss ──

class CombinedLoss(nn.Module):
    """
    Combined training loss:
        L = w_mse * L_mse + w_perc * L_perceptual + w_freq * L_frequency

    All sub-losses are optional (disabled when weight = 0).
    """
    def __init__(self, w_mse: float = 1.0, w_perc: float = 0.0,
                 w_freq: float = 0.0, device: torch.device = None):
        super().__init__()
        self.w_mse = w_mse
        self.w_perc = w_perc
        self.w_freq = w_freq

        self.mse = nn.MSELoss()
        self.perceptual = PerceptualLoss(device=device) if w_perc > 0 else None
        self.frequency = FrequencyLoss() if w_freq > 0 else None

    def forward(self, output: torch.Tensor, target: torch.Tensor
                ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss, dict of individual losses
        """
        details = {}

        mse_loss = self.mse(output, target)
        details['mse'] = mse_loss.item()
        total = self.w_mse * mse_loss

        perc_loss = 0.0
        if self.perceptual is not None:
            perc_loss = self.perceptual(output, target)
            details['perceptual'] = perc_loss.item()
            total += self.w_perc * perc_loss

        freq_loss = 0.0
        if self.frequency is not None:
            freq_loss = self.frequency(output, target)
            details['frequency'] = freq_loss.item()
            total += self.w_freq * freq_loss

        details['total'] = total.item()
        return total, details


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test perceptual loss
    perc = PerceptualLoss(device=device)
    x = torch.randn(2, 3, 128, 128, device=device)
    y = torch.randn(2, 3, 128, 128, device=device)
    p_loss = perc(x, y)
    print(f"Perceptual loss: {p_loss.item():.4f}")

    # Test frequency loss
    freq = FrequencyLoss()
    f_loss = freq(x, y)
    print(f"Frequency loss: {f_loss.item():.4f}")

    # Test combined loss
    combined = CombinedLoss(w_mse=1.0, w_perc=0.1, w_freq=0.1, device=device)
    c_loss, details = combined(x, y)
    print(f"Combined loss: {c_loss.item():.4f}  details: {details}")
