#!/usr/bin/env python
"""
run.py
Denoise / enhance a single image with trained model(s).

Supports:
  1. DnCNN only       (--model)
  2. DnCNN + Edge     (--model + --edge-model)
  3. Joint fine-tuned (--joint-model, contains both networks)

Denoising strength (--strength):
  Control how aggressively noise is removed.
  0.0 = no denoising, 1.0 = full denoising (default), >1.0 = over-denoising.

Large image handling:
  --resize N   : downscale so longest side <= N pixels (aspect ratio preserved)
  --tile S,O   : split into tiles of size S with O-pixel overlap, process
                 independently, then stitch with linear blending at seams.

Uncertainty estimation (--uncertainty):
  Runs N=4 TTA passes, computes per-pixel variance as uncertainty proxy.
  Outputs:
    - _uncertainty.jpg : JET heatmap (blue=confident, red=uncertain)
    - _uncertainty.raw : raw float32 stream [HxWxC, C-order, no header]

Usage:
    python run.py --model dncnn_best.pth --input noise.png
    python run.py --joint-model joint_best.pth --input photo.png --uncertainty
    python run.py --model dncnn_best.pth --input noise.png --strength 0.7
    python run.py --model dncnn_best.pth --input large.png --resize 1024
    python run.py --model dncnn_best.pth --input huge.png --tile 256,32
"""
import argparse, os, math

import cv2
import numpy as np
import torch

import config
from models.dncnn import DnCNN
from models.edge_enhancer import EdgeEnhancementNetwork


# -- helpers --

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


# -- noise estimation / EXIF helpers --

def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estimate noise std dev using Donoho's wavelet method (HH subband)."""
    import pywt
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coeffs = pywt.dwt2(gray.astype(np.float32), 'db8')
    _, (_, _, HH) = coeffs
    return float(np.median(np.abs(HH)) / 0.6745)


def read_exif_iso(image_path: str) -> int | None:
    """Read ISO speed rating from EXIF metadata (tag 34855 = ISOSpeedRatings)."""
    try:
        from PIL import Image
        iso = Image.open(image_path).getexif().get(34855)
        if iso is not None:
            return int(iso)
    except Exception:
        pass
    return None


def sigma_to_strength(sigma: float, cap: float = 3.0) -> float:
    """Piecewise linear noise-to-strength mapping.

    Breakpoints (sigma, strength):
      (5,  0.0) — noise-free threshold
      (25, 1.0) — training noise baseline
      (50, 1.5)
      (100, 2.5)
      cap at `cap` (default 3.0).
    """
    if sigma < 5:
        return 0.0
    points = [(5, 0.0), (25, 1.0), (50, 1.5), (100, 2.5)]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if sigma <= x2:
            return y1 + (sigma - x1) * (y2 - y1) / (x2 - x1)
    # Extrapolate beyond 100 with last segment slope (0.02)
    slope = (points[-1][1] - points[-2][1]) / (points[-1][0] - points[-2][0])
    raw = points[-1][1] + (sigma - points[-1][0]) * slope
    return min(cap, raw)


# -- model loaders --

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


def load_edge(ckpt_path: str, device: torch.device) -> 'EdgeEnhancementNetwork':
    net = EdgeEnhancementNetwork(in_channels=3, base_channels=64)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'state_dict' in state:
        state = state['state_dict']
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return net


def load_joint(ckpt_path: str, device: torch.device):
    """Load joint checkpoint -> returns (dncnn_model, edge_net)."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    dncnn = DnCNN()
    if 'dncnn_state_dict' in state:
        dncnn.load_state_dict(state['dncnn_state_dict'], strict=False)
    dncnn.to(device).eval()
    edge_net = EdgeEnhancementNetwork(in_channels=3, base_channels=64)
    if 'edge_state_dict' in state:
        edge_net.load_state_dict(state['edge_state_dict'], strict=False)
    edge_net.to(device).eval()
    return dncnn, edge_net


# -- pre/post processing --

def _preprocess(image: np.ndarray, pad: tuple[int, int]) -> np.ndarray:
    ph, pw = pad
    if ph or pw:
        image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT101)
    return image.astype(np.float32) / 255.0


def _postprocess(tensor: torch.Tensor, pad: tuple[int, int],
                 h: int, w: int) -> np.ndarray:
    out = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    ph, pw = pad
    if ph or pw:
        out = out[:h, :w]
    return out


def _compute_padding(h: int, w: int, has_edge: bool) -> tuple[int, int]:
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if has_edge:
        pad_h += (32 - (h + pad_h) % 32) % 32
        pad_w += (32 - (w + pad_w) % 32) % 32
    return pad_h, pad_w


# -- single-pass inference (strength-aware) --

@torch.no_grad()
def _forward_dncnn(model: DnCNN, inp_t: torch.Tensor,
                   strength: float = 1.0) -> torch.Tensor:
    noise_pred = model(inp_t)
    return inp_t - noise_pred * strength


@torch.no_grad()
def _forward_dncnn_edge(model: DnCNN, edge_net: EdgeEnhancementNetwork,
                        inp_t: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    noise_pred = model(inp_t)
    denoised_t = inp_t - noise_pred * strength
    enhanced_t, _ = edge_net(denoised_t)
    return enhanced_t


# -- main inference --

@torch.no_grad()
def run_dncnn(model: DnCNN, image: np.ndarray,
              device: torch.device, strength: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    pad = _compute_padding(h, w, has_edge=False)
    inp = _preprocess(image, pad)
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    out_t = _forward_dncnn(model, inp_t, strength=strength)
    return _postprocess(out_t, pad, h, w)


@torch.no_grad()
def run_dncnn_edge(model: DnCNN, edge_net: EdgeEnhancementNetwork,
                   image: np.ndarray, device: torch.device,
                   strength: float = 1.0) -> np.ndarray:
    h, w = image.shape[:2]
    pad = _compute_padding(h, w, has_edge=True)
    inp = _preprocess(image, pad)
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    out_t = _forward_dncnn_edge(model, edge_net, inp_t, strength=strength)
    return _postprocess(out_t, pad, h, w)


# -- large image: tiling --

def _tile_weight_map(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """Create 2D weight for a tile: 1.0 in interior, linear fade to 0 at edges."""
    y = np.ones(tile_h, dtype=np.float32)
    if overlap > 0 and overlap < tile_h // 2:
        ramp = np.linspace(0, 1, overlap + 1)[1:]
        y[:overlap] = ramp
        y[-overlap:] = ramp[::-1]
    x = np.ones(tile_w, dtype=np.float32)
    if overlap > 0 and overlap < tile_w // 2:
        ramp = np.linspace(0, 1, overlap + 1)[1:]
        x[:overlap] = ramp
        x[-overlap:] = ramp[::-1]
    return y[:, None] * x[None, :]


def _tile_process(image: np.ndarray, tile_size: int, overlap: int,
                  process_fn, **fn_kwargs) -> np.ndarray:
    """
    Split image into overlapping tiles, process each, stitch with blending.

    Blending: linear weight ramp over overlap region ensures seamless seams.
    """
    H, W = image.shape[:2]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be < tile_size ({tile_size})")

    weight_map = _tile_weight_map(tile_size, tile_size, overlap)
    weight_map_3 = weight_map[:, :, None]

    accum = np.zeros((H, W, 3), dtype=np.float64)
    weight_accum = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)

            tile = image[y1:y2, x1:x2].copy()
            pad_b = tile_size - (y2 - y1)
            pad_r = tile_size - (x2 - x1)
            if pad_b or pad_r:
                tile = cv2.copyMakeBorder(tile, 0, pad_b, 0, pad_r,
                                          cv2.BORDER_REFLECT101)

            processed = process_fn(tile, **fn_kwargs)
            if pad_b or pad_r:
                processed = processed[:tile_size - pad_b, :tile_size - pad_r]

            w = weight_map_3[:y2 - y1, :x2 - x1]
            accum[y1:y2, x1:x2] += processed.astype(np.float64) * w
            weight_accum[y1:y2, x1:x2] += weight_map[:y2 - y1, :x2 - x1]

    weight_accum = np.maximum(weight_accum, 1e-10)
    return (accum / weight_accum[:, :, None]).astype(np.uint8)


# -- uncertainty estimation --

def _tta_transforms(inp_t: torch.Tensor):
    yield inp_t, lambda x: x
    yield torch.flip(inp_t, dims=[3]), lambda x: torch.flip(x, dims=[3])
    yield torch.flip(inp_t, dims=[2]), lambda x: torch.flip(x, dims=[2])
    yield torch.flip(torch.flip(inp_t, dims=[3]), dims=[2]), \
        lambda x: torch.flip(torch.flip(x, dims=[3]), dims=[2])


@torch.no_grad()
def run_with_uncertainty(model, edge_net, image: np.ndarray,
                         device: torch.device, strength: float = 1.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    has_edge = edge_net is not None
    pad = _compute_padding(h, w, has_edge)
    inp = _preprocess(image, pad)
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

    accum = torch.zeros(1, 3, inp_t.shape[2], inp_t.shape[3], device='cpu')
    accum_sq = torch.zeros(1, 3, inp_t.shape[2], inp_t.shape[3], device='cpu')
    count = 0

    for x_t, inverse_fn in _tta_transforms(inp_t):
        if has_edge:
            out_t = _forward_dncnn_edge(model, edge_net, x_t, strength=strength)
        else:
            out_t = _forward_dncnn(model, x_t, strength=strength)
        out_aligned = inverse_fn(out_t).cpu().float()
        accum += out_aligned
        accum_sq += out_aligned ** 2
        count += 1

    mean_t = accum / count
    var_t = torch.clamp((accum_sq / count) - mean_t ** 2, min=0.0)
    out_np = _postprocess(mean_t, pad, h, w)
    var_np = var_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    if pad[0] or pad[1]:
        var_np = var_np[:h, :w]
    return out_np, var_np.astype(np.float32)


def save_uncertainty(var_map: np.ndarray, base_path: str):
    gray_var = np.mean(var_map, axis=2)
    vmin, vmax = np.percentile(gray_var, [1, 99])
    if vmax - vmin < 1e-10:
        vmax = vmin + 1e-6
    norm = np.clip((gray_var - vmin) / (vmax - vmin), 0, 1)
    heatmap_color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(base_path + '_uncertainty.jpg', heatmap_color)
    print(f"  Uncertainty heatmap: {base_path}_uncertainty.jpg")

    raw_path = base_path + '_uncertainty.raw'
    with open(raw_path, 'wb') as f:
        f.write(var_map.tobytes())
    sz = var_map.shape
    print(f"  Uncertainty stream : {raw_path}  ({sz[0]}x{sz[1]}x{sz[2]} = {var_map.nbytes} bytes)")


# -- CLI --

def main():
    parser = argparse.ArgumentParser(description='Denoise/enhance a single image')
    parser.add_argument('--model', '-m', default=None)
    parser.add_argument('--edge-model', default=None)
    parser.add_argument('--joint-model', default=None)
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--sigma', '-s', type=float, default=0)
    parser.add_argument('--noise-seed', type=int, default=None)
    parser.add_argument('--strength', type=float, default=1.0,
                        help='Denoising strength: 0.0=off, 1.0=full (default), >1.0=over')
    parser.add_argument('--auto-strength', action='store_true',
                        help='Auto-determine strength from EXIF ISO or wavelet estimation')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--uncertainty', action='store_true')
    parser.add_argument('--resize', type=int, default=0,
                        help='Resize longest side to N pixels')
    parser.add_argument('--tile', type=str, default=None,
                        help='Tile processing: "tile_size,overlap" e.g. "256,32"')
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    # -- validate args --
    if args.joint_model:
        if not os.path.exists(args.joint_model):
            raise FileNotFoundError(f"Joint model not found: {args.joint_model}")
        print(f"Mode: joint model ({os.path.basename(args.joint_model)})")
        model, edge_net = load_joint(args.joint_model, device)
    elif args.model:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}")
        model = load_dncnn(args.model, device)
        n_p = sum(p.numel() for p in model.parameters())
        edge_net = None
        if args.edge_model:
            if not os.path.exists(args.edge_model):
                raise FileNotFoundError(f"Edge model not found: {args.edge_model}")
            edge_net = load_edge(args.edge_model, device)
            e_p = sum(p.numel() for p in edge_net.parameters())
            print(f"Mode: DnCNN ({n_p:,} params) -> EdgeEnhance ({e_p:,} params)")
        else:
            print(f"Mode: DnCNN only ({n_p:,} params)")
    else:
        raise ValueError("Either --model or --joint-model is required")

    strength = max(0.0, args.strength)
    if abs(strength - 1.0) > 1e-6:
        print(f"Denoising strength: {strength:.2f}")
    if args.tta:
        print("TTA: enabled (4-way flip)")
    if args.uncertainty:
        print("Uncertainty: enabled")
    if args.resize:
        print(f"Resize: longest side <= {args.resize} px")
    if args.tile:
        try:
            ts, ov = map(int, args.tile.split(','))
        except ValueError:
            raise ValueError("--tile must be 'tile_size,overlap' e.g. '256,32'")
        if ov >= ts:
            raise ValueError(f"overlap ({ov}) must be < tile_size ({ts})")
        print(f"Tile: {ts}x{ts}, overlap={ov}px")

    # -- load image --
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {args.input}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original = img_rgb.copy()
    print(f"Image: {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    # -- auto strength (overrides --strength, piecewise mapping) --
    if args.auto_strength:
        iso = read_exif_iso(args.input)
        if iso is not None and iso > 0:
            if iso <= 400:
                strength = 0.0
                print(f"  Auto-strength: EXIF ISO={iso} -> strength=0 (ISO<=400)")
            else:
                iso_sigma = iso / 32.0  # rough ISO→sigma conversion
                strength = sigma_to_strength(iso_sigma)
                print(f"  Auto-strength: EXIF ISO={iso} -> sigma~{iso_sigma:.0f} -> strength={strength:.2f}")
        else:
            noise_sigma = estimate_noise_sigma(img_rgb)
            strength = sigma_to_strength(noise_sigma)
            print(f"  Auto-strength: wavelet sigma~{noise_sigma:.1f} -> strength={strength:.2f}")

    # -- resize (optional) --
    if args.resize:
        h, w = img_rgb.shape[:2]
        scale = args.resize / max(h, w)
        if scale < 1.0:
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            print(f"Resized to: {int(w * scale)}x{int(h * scale)}")

    # -- optionally add noise --
    if args.sigma > 0:
        if args.noise_seed is not None:
            np.random.seed(args.noise_seed)
        noise = np.random.randn(*img_rgb.shape).astype(np.float32) * args.sigma
        img_rgb = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        print(f"Added Gaussian noise sigma={args.sigma}")

    # -- inference dispatch --
    has_edge = edge_net is not None

    if args.uncertainty:
        result, var_map = run_with_uncertainty(model, edge_net, img_rgb, device,
                                               strength=strength)
    elif args.tile:
        ts, ov = map(int, args.tile.split(','))
        if has_edge:
            def tile_fn(tile_img):
                return run_dncnn_edge(model, edge_net, tile_img, device, strength=strength)
        else:
            def tile_fn(tile_img):
                return run_dncnn(model, tile_img, device, strength=strength)
        result = _tile_process(img_rgb, ts, ov, tile_fn)
    else:
        if has_edge:
            result = run_dncnn_edge(model, edge_net, img_rgb, device, strength=strength)
        else:
            result = run_dncnn(model, img_rgb, device, strength=strength)

    # -- metrics --
    if args.sigma > 0:
        noisy_psnr = psnr(original, img_rgb)
        clean_psnr = psnr(original, result)
        print(f"Noisy   PSNR: {noisy_psnr:.2f} dB")
        print(f"Output  PSNR: {clean_psnr:.2f} dB  (+{clean_psnr - noisy_psnr:.2f})")

    # -- save --
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        pipeline = "joint" if args.joint_model else "dncnn"
        args.output = f"{base}_{pipeline}{ext}"

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    print(f"Saved: {args.output}")

    if args.sigma > 0:
        noisy_path = args.output.rsplit('.', 1)[0] + '_noisy.' + args.output.rsplit('.', 1)[1]
        noisy_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(noisy_path, noisy_bgr)
        print(f"Saved noisy: {noisy_path}")

    if args.uncertainty:
        save_uncertainty(var_map, args.output.rsplit('.', 1)[0])


if __name__ == '__main__':
    main()