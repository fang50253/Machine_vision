"""
models/dncnn.py
Clean DnCNN implementation with residual learning (predicts noise).
Uses Kaiming normal initialisation for stable training.

Optional features (config-controlled):
  - Channel attention (SELayer) for improved feature selection
  - Deeper variant (20 layers / 96 channels)
"""
import torch
import torch.nn as nn


# ── optional channel attention ──

class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ── DnCNN ──

class DnCNN(nn.Module):
    """
    DnCNN with optional channel attention.

    Architecture:
        - 1x Conv(3→64, 3×3) + ReLU
        - (N-2)x Conv(64→64, 3×3) + BN + ReLU  (+ optional SE per block)
        - 1x Conv(64→3, 3×3)

    Forward: predicts noise → denoised = input - noise_pred (handled by caller).
    """

    def __init__(self, channels: int = 3, num_layers: int = 17,
                 num_features: int = 64, use_attention: bool = False):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.num_features = num_features
        self.use_attention = use_attention

        # first layer: Conv + ReLU (no BN)
        layers = [
            nn.Conv2d(channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        # middle layers: Conv + BN + ReLU (+ optional SE)
        for _ in range(num_layers - 2):
            block = [
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
            if use_attention:
                block.append(SELayer(num_features))
            layers.extend(block)

        # last layer: Conv (no BN, no ReLU)
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1))

        self.dncnn = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialisation – critical for DnCNN convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict residual noise map (shape same as input)."""
        return self.dncnn(x)


# ── convenience factory ──

def create_dncnn(cfg=None, **kwargs) -> DnCNN:
    """Build a DnCNN from a config module / dict or explicit kwargs."""
    if cfg is not None:
        return DnCNN(
            channels=getattr(cfg, 'CHANNELS', 3),
            num_layers=getattr(cfg, 'NUM_LAYERS', 17),
            num_features=getattr(cfg, 'NUM_FEATURES', 64),
            use_attention=getattr(cfg, 'USE_ATTENTION', False),
        )
    return DnCNN(**kwargs)


def create_dncnn_deep(cfg=None) -> DnCNN:
    """Build a deeper DnCNN (20 layers / 96 channels) with attention."""
    if cfg is not None:
        return DnCNN(
            channels=getattr(cfg, 'CHANNELS', 3),
            num_layers=getattr(cfg, 'NUM_LAYERS_DEEP', 20),
            num_features=getattr(cfg, 'NUM_FEATURES_DEEP', 96),
            use_attention=True,
        )
    return DnCNN(channels=3, num_layers=20, num_features=96, use_attention=True)


if __name__ == '__main__':
    net = DnCNN(use_attention=False)
    x = torch.randn(2, 3, 128, 128)
    y = net(x)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"DnCNN (base): output={y.shape}  params={n_params:,}")

    net2 = DnCNN(use_attention=True)
    y2 = net2(x)
    n_params2 = sum(p.numel() for p in net2.parameters())
    print(f"DnCNN (attn): output={y2.shape}  params={n_params2:,}")

    net3 = create_dncnn_deep()
    y3 = net3(x)
    n_params3 = sum(p.numel() for p in net3.parameters())
    print(f"DnCNN (deep): output={y3.shape}  params={n_params3:,}")
