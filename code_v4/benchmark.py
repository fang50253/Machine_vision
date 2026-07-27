#!/usr/bin/env python
"""
benchmark_full.py
Full comparison benchmark: Traditional DnCNN vs DnCNN+EdgeEnhance vs Traditional methods.

Usage:
    python benchmark.py --data ../Div2k/DIV2K_valid_HR \\
        --model trained_models/dncnn_best_20251026.pth \\
        --edge-model trained_models/dncnn_to_original/best_model.pth

    python benchmark.py --data /path/to/test --sigma 25 --max-images 20
"""
import argparse, os, math, time, glob, csv
from datetime import datetime
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# ???????????? extra metrics ????????????

def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean squared error (lower = better)."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))

def mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean absolute error (lower = better)."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))

# ???????????? metrics ????????????

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        if img1.ndim == 3:
            g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            g1, g2 = img1, img2
        return float(sk_ssim(g1, g2, data_range=255))
    except ImportError:
        return 0.0

# ???????????? model loaders ????????????

def load_dncnn(ckpt_path: str, device: torch.device) -> nn.Module:
    from models.dncnn import DnCNN
    model = DnCNN()
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'model_state' in state:
        state = state['model_state']
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

def load_edge_network(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load EdgeEnhancementNetwork directly (no wrapper dependency)."""
    from models.edge_enhancer import EdgeEnhancementNetwork
    net = EdgeEnhancementNetwork(in_channels=3, base_channels=64)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return net

# ???????????? joint model loader ????????????

def load_joint_checkpoint(ckpt_path: str, device: torch.device):
    """Load joint fine-tuned model (DnCNN + EdgeEnhance checkpoints bundled)."""
    from models.dncnn import DnCNN
    from models.edge_enhancer import EdgeEnhancementNetwork
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    dncnn = DnCNN()
    if 'dncnn_state_dict' in state:
        dncnn.load_state_dict(state['dncnn_state_dict'], strict=False)
    print(f"  Joint model DnCNN loaded")
    dncnn.to(device).eval()

    edge_net = EdgeEnhancementNetwork(in_channels=3, base_channels=64)
    if 'edge_state_dict' in state:
        edge_net.load_state_dict(state['edge_state_dict'], strict=False)
    print(f"  Joint model EdgeEnhance loaded")
    edge_net.to(device).eval()

    return dncnn, edge_net

# ???????????? TTA inference ????????????

@torch.no_grad()
def run_dncnn_tta(image: np.ndarray, model: nn.Module,
                  device: torch.device, strength: float = 1.0) -> np.ndarray:
    """
    Test-time augmentation: 8-way flips + rotations, average outputs.
    """
    h, w = image.shape[:2]
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT101)
    inp = image.astype(np.float32) / 255.0
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

    denoised = None
    count = 0

    # Identity
    noise_pred = model(inp_t)
    denoised = (inp_t - noise_pred * strength).cpu()
    count = 1

    # Horizontal flip
    x_h = torch.flip(inp_t, dims=[3])
    noise_pred = model(x_h)
    out_h = torch.flip(x_h - noise_pred * strength, dims=[3])
    denoised = denoised + out_h.cpu()
    count += 1

    # Vertical flip
    x_v = torch.flip(inp_t, dims=[2])
    noise_pred = model(x_v)
    out_v = torch.flip(x_v - noise_pred * strength, dims=[2])
    denoised = denoised + out_v.cpu()
    count += 1

    # H+V flip
    x_hv = torch.flip(torch.flip(inp_t, dims=[3]), dims=[2])
    noise_pred = model(x_hv)
    out_hv = torch.flip(torch.flip(x_hv - noise_pred * strength, dims=[3]), dims=[2])
    denoised = denoised + out_hv.cpu()
    count += 1

    denoised = denoised / count

    out_np = denoised.squeeze(0).numpy().transpose(1, 2, 0)
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    if pad_h or pad_w:
        out_np = out_np[:h, :w]
    return out_np


@torch.no_grad()
def run_dncnn_edge_tta(image: np.ndarray, model: nn.Module,
                       edge: nn.Module, device: torch.device,
                       strength: float = 1.0) -> np.ndarray:
    """DnCNN + Edge pipeline with 4-way TTA."""
    h, w = image.shape[:2]
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    pad_h2 = (32 - (h + pad_h) % 32) % 32
    pad_w2 = (32 - (w + pad_w) % 32) % 32
    if pad_h or pad_w or pad_h2 or pad_w2:
        image = cv2.copyMakeBorder(image, 0, pad_h + pad_h2, 0, pad_w + pad_w2,
                                   cv2.BORDER_REFLECT101)
    inp = image.astype(np.float32) / 255.0
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

    accumulated = None
    count = 0

    for flip_h in [False, True]:
        for flip_v in [False, True]:
            x = inp_t
            if flip_h:
                x = torch.flip(x, dims=[3])
            if flip_v:
                x = torch.flip(x, dims=[2])

            noise_pred = model(x)
            denoised = x - noise_pred * strength
            enhanced, _ = edge(denoised)

            if flip_v:
                enhanced = torch.flip(enhanced, dims=[2])
            if flip_h:
                enhanced = torch.flip(enhanced, dims=[3])

            if accumulated is None:
                accumulated = enhanced.cpu()
            else:
                accumulated = accumulated + enhanced.cpu()
            count += 1

    out = accumulated / count
    out_np = out.squeeze(0).numpy().transpose(1, 2, 0)
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    total_pad = pad_h + pad_h2
    if total_pad > 0:
        out_np = out_np[:-(pad_h + pad_h2), :] if (pad_h + pad_h2) > 0 else out_np
    actual_h = h
    actual_w = w
    if total_pad:
        out_np = out_np[:actual_h, :actual_w]
    return out_np


# ???????????? inference functions ????????????

@torch.no_grad()
def run_dncnn(image: np.ndarray, model: nn.Module, device: torch.device,
              strength: float = 1.0) -> np.ndarray:
    """Denoise with plain DnCNN (strength: 0.0=off, 1.0=full, >1.0=over)."""
    h, w = image.shape[:2]
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT101)
    inp = image.astype(np.float32) / 255.0
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    noise_pred = model(inp_t)
    out_t = inp_t - noise_pred * strength
    out = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    if pad_h or pad_w:
        out = out[:h, :w]
    return out

@torch.no_grad()
def run_edge_enhance(image: np.ndarray, net: nn.Module, device: torch.device) -> np.ndarray:
    """Run edge enhancement only (for comparison)."""
    # pad to multiples of 32 (edge network requirement)
    h, w = image.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT101)
    inp = image.astype(np.float32) / 255.0
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    enhanced_t, _ = net(inp_t)
    out = enhanced_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    if pad_h or pad_w:
        out = out[:h, :w]
    return out

def run_dncnn_edge(image: np.ndarray, model: nn.Module,
                   edge: nn.Module, device: torch.device,
                   strength: float = 1.0) -> np.ndarray:
    """DnCNN denoise + edge enhancement pipeline."""
    dncnn_out = run_dncnn(image, model=model, device=device, strength=strength)
    enhanced = run_edge_enhance(dncnn_out, net=edge, device=device)
    return enhanced

def run_wavelet(image: np.ndarray) -> np.ndarray:
    from models.traditional_denoiser import TraditionalDenoiser
    return TraditionalDenoiser().wavelet_denoise_robust(image)

def run_bilateral(image: np.ndarray) -> np.ndarray:
    from models.traditional_denoiser import TraditionalDenoiser
    return TraditionalDenoiser().bilateral_denoise_basic(image)

def run_nlm(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def run_median(image: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(image, 5)

def run_gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (5, 5), 1.0)

# ???????????? data collection ????????????

def collect_images(root: str) -> list[str]:
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif',
            '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF', '*.TIF']
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
    return sorted(set(files))

# ???????????? main ????????????

def main():
    parser = argparse.ArgumentParser(
        description='Full benchmark: DnCNN vs DnCNN+Edge vs Traditional')
    parser.add_argument('--data', '-d', required=True,
                        help='Path to test images folder')
    parser.add_argument('--model', '-m', required=True,
                        help='Path to DnCNN checkpoint (.pth)')
    parser.add_argument('--edge-model', default=None,
                        help='Path to edge enhancement checkpoint (omit to skip DnCNN+Edge)')
    parser.add_argument('--sigma', type=float, default=25,
                        help='Gaussian noise sigma')
    parser.add_argument('--max-size', type=int, default=1024,
                        help='Max dimension (downscale larger)')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Limit images (0 = all)')
    parser.add_argument('--output', '-o', default=None,
                        help='CSV output path (default: auto)')
    parser.add_argument('--noise-seed', type=int, default=None,
                        help='Random seed for reproducible noise')
    parser.add_argument('--noise-type', default='gaussian', choices=['gaussian', 'uniform'],
                        help='Noise distribution type (default: gaussian)')
    parser.add_argument('--save-samples', type=int, default=0,
                        help='Number of sample comparison images to save (default: 0)')
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (4-way flip) for deep learning methods')
    parser.add_argument('--joint-model', default=None,
                        help='Path to joint fine-tuned model (.pth) for DnCNN+Edge pipeline')
    parser.add_argument('--auto-strength', action='store_true',
                        help='Auto-determine denoising strength from sigma: min(1.20, sigma/25*1.0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device   : {device}")
    print(f"Noise ?  : {args.sigma}")
    print(f"Noise type: {args.noise_type}")
    print(f"Data     : {args.data}")
    print(f"Model    : {args.model}")
    print(f"Edge model: {args.edge_model or '(not provided, skip DnCNN+Edge)'}")
    print(f"Joint model: {args.joint_model or '(not provided)'}")
    if args.tta:
        print(f"TTA      : enabled (4-way flip ensemble)")

    # ?? auto strength (piecewise mapping) ??
    def sigma_to_strength(sigma: float, cap: float = 3.0) -> float:
        if sigma < 5:
            return 0.0
        pts = [(5, 0.0), (25, 1.0), (50, 1.5), (100, 2.5)]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            if sigma <= x2:
                return y1 + (sigma - x1) * (y2 - y1) / (x2 - x1)
        slope = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
        return min(cap, pts[-1][1] + (sigma - pts[-1][0]) * slope)

    auto_strength_val: float | None = None
    if args.auto_strength:
        auto_strength_val = sigma_to_strength(args.sigma)
        print(f"Auto-strength: sigma={args.sigma} -> strength={auto_strength_val:.2f}")

    # ?? load models ??
    dncnn_model = load_dncnn(args.model, device)
    n_params = sum(p.numel() for p in dncnn_model.parameters())
    print(f"DnCNN parameters: {n_params:,}")

    # Load joint model (overrides separate models if provided)
    edge_net = None
    if args.joint_model and os.path.exists(args.joint_model):
        dncnn_model, edge_net = load_joint_checkpoint(args.joint_model, device)
    elif args.edge_model and os.path.exists(args.edge_model):
        edge_net = load_edge_network(args.edge_model, device)
        e_params = sum(p.numel() for p in edge_net.parameters())
        print(f"EdgeEnhance parameters: {e_params:,}")

    # ?? collect images ??
    image_paths = collect_images(args.data)
    if not image_paths:
        print("No images found!")
        return
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} images\n")

    # ?? define methods ??
    methods = OrderedDict()

    # Traditional
    methods['GaussianBlur'] = {'fn': lambda img, **kw: run_gaussian_blur(img)}
    methods['Median']       = {'fn': lambda img, **kw: run_median(img)}
    methods['Bilateral']    = {'fn': lambda img, **kw: run_bilateral(img)}
    methods['Wavelet']      = {'fn': lambda img, **kw: run_wavelet(img)}
    methods['NLM']          = {'fn': lambda img, **kw: run_nlm(img)}

    # Deep learning (with optional TTA)
    dncnn_fn = run_dncnn_tta if args.tta else run_dncnn
    dncnn_extra: dict = {'model': dncnn_model, 'device': device}
    if auto_strength_val is not None:
        dncnn_extra['strength'] = auto_strength_val
    methods['DnCNN'] = {'fn': dncnn_fn, 'extra': dncnn_extra}

    if edge_net is not None:
        edge_combined_fn = run_dncnn_edge_tta if args.tta else run_dncnn_edge
        edge_extra: dict = {'model': dncnn_model, 'edge': edge_net, 'device': device}
        if auto_strength_val is not None:
            edge_extra['strength'] = auto_strength_val
        methods['DnCNN+Edge'] = {'fn': edge_combined_fn, 'extra': edge_extra}
        methods['EdgeOnly'] = {
            'fn': run_edge_enhance,
            'extra': {'net': edge_net, 'device': device},
        }

    # ?? benchmark ??
    rows: list[dict] = []
    method_results: dict[str, list[float]] = {name: [] for name in methods}
    method_ssims: dict[str, list[float]] = {name: [] for name in methods}
    method_times: dict[str, list[float]] = {name: [] for name in methods}
    noisy_psnrs: list[float] = []   # noisy vs clean baseline

    header = f"{'Image':30s}" + "".join(f"  {name:>12s}" for name in methods)
    print(header)
    print('-' * len(header))

    for img_idx, img_path in enumerate(image_paths):
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        if max(h, w) > args.max_size:
            scale = args.max_size / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        # add noise (gaussian or uniform)
        if args.noise_seed is not None:
            np.random.seed(args.noise_seed + img_idx)
        if args.noise_type == 'uniform':
            # uniform with same std: scale = sigma * sqrt(3)
            scale = args.sigma * math.sqrt(3)
            noise = np.random.uniform(-scale, scale, rgb.shape).astype(np.float32)
        else:
            noise = np.random.randn(*rgb.shape).astype(np.float32) * args.sigma
        noisy = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        noisy_psnr = psnr(rgb, noisy)
        noisy_psnrs.append(noisy_psnr)

        line = f"{os.path.basename(img_path)[:28]:30s}"

        for name, method in methods.items():
            extra = method.get('extra', {})
            t0 = time.perf_counter()
            try:
                result = method['fn'](noisy, **extra)
            except Exception as e:
                result = noisy
                print(f"\n  {name} ERROR: {e}")
            elapsed = time.perf_counter() - t0

            p = psnr(rgb, result)
            s = ssim(rgb, result)
            mse_val = mse(rgb, result)
            mae_val = mae(rgb, result)
            method_results[name].append(p)
            method_ssims[name].append(s)
            method_times[name].append(elapsed)
            line += f"  {p:8.2f}  "

        print(line)

        row = {
            'image': os.path.basename(img_path),
            'width': rgb.shape[1],
            'height': rgb.shape[0],
        }
        for name in methods:
            row[f'{name}_psnr'] = round(method_results[name][-1], 2)
            row[f'{name}_ssim'] = round(method_ssims[name][-1], 4)
            row[f'{name}_mse'] = round(mse_val, 1)
            row[f'{name}_mae'] = round(mae_val, 1)
            row[f'{name}_time'] = round(method_times[name][-1], 3)
        rows.append(row)

    # ?? save sample comparison images ??
    if args.save_samples > 0:
        sample_dir = os.path.join(os.path.dirname(args.output or '.'), 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        # pick first N valid images
        saved = 0
        for img_idx, img_path in enumerate(image_paths):
            if saved >= args.save_samples:
                break
            if img_idx >= len(rows):
                break

            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if max(h, w) > args.max_size:
                scale = args.max_size / max(h, w)
                rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
            if args.noise_seed is not None:
                np.random.seed(args.noise_seed + img_idx)
            if args.noise_type == 'uniform':
                scale_n = args.sigma * math.sqrt(3)
                noise = np.random.uniform(-scale_n, scale_n, rgb.shape).astype(np.float32)
            else:
                noise = np.random.randn(*rgb.shape).astype(np.float32) * args.sigma
            noisy = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # collect results from all methods
            results = {}
            for name, method in methods.items():
                extra = method.get('extra', {})
                try:
                    res = method['fn'](noisy, **extra)
                except Exception:
                    res = noisy
                results[name] = res

            # build comparison canvas: rows = methods, cols = [clean, noisy, result]
            n_methods = len(methods)
            cell_h, cell_w = 200, 200
            canvas_h = cell_h * (1 + n_methods)  # clean row + method rows
            canvas_w = cell_w * 3  # clean | noisy | result
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            def put(img, row, col):
                """Place image at canvas position, auto-resize to cell."""
                resized = cv2.resize(img, (cell_w, cell_h))
                y0, x0 = row * cell_h, col * cell_w
                canvas[y0:y0+cell_h, x0:x0+cell_w] = resized

            def label(text, row, col):
                """Put label text on canvas."""
                y0 = row * cell_h + 12
                x0 = col * cell_w + 4
                cv2.putText(canvas, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)

            put(rgb, 0, 0); label('Clean', 0, 0)
            put(noisy, 0, 1); label('Noisy', 0, 1)
            # empty third column for clean row

            for i, (name, res) in enumerate(results.items()):
                row = i + 1
                put(rgb, row, 0)   # clean reference
                put(noisy, row, 1)  # noisy input
                put(res, row, 2)    # method output
                p = psnr(rgb, res)
                s = ssim(rgb, res)
                label(f'{name}  {p:.1f}dB/{s:.3f}', row, 2)

            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(sample_dir, f'{base}_comparison.png')
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, canvas_bgr)
            print(f"  [sample saved] {out_path}")
            saved += 1

    # ?? summary ??
    n = len(rows)
    avg_noisy_psnr = np.mean(noisy_psnrs)

    # rank methods by mean PSNR
    sorted_methods = sorted(
        methods.keys(),
        key=lambda m: np.mean(method_results[m]) if method_results[m] else 0,
        reverse=True,
    )

    medals = ['[#1]', '[#2]', '[#3]']  # rank badges (ASCII-safe)

    print('\n' + '=' * 120)
    print('  CONCLUSION TABLE ? Full Method Comparison')
    print('=' * 120)
    print(f"  Dataset       : {args.data}")
    print(f"  Images        : {n}")
    print(f"  Noise ?       : {args.sigma}")
    print(f"  Noisy baseline: {avg_noisy_psnr:.2f} dB")
    if args.edge_model:
        mname = os.path.basename(args.edge_model)
        if args.joint_model:
            mname = os.path.basename(args.joint_model) + ' (joint)'
        print(f"  Edge model    : {mname}")
    print(f"  DnCNN model   : {os.path.basename(args.model)}")
    if args.tta:
        print(f"  TTA           : enabled (4-way flip ensemble)")
    print()
    print(f"  {'Rank':<6s}  {'Method':<20s}  {'PSNR':>8s}  {'+/-PSNR':>8s}  {'SSIM':>8s}  {'Time':>8s}  {'Category':<16s}")
    print(f"  {'-'*4:6s}  {'-'*18:20s}  {'-'*6:8s}  {'-'*6:8s}  {'-'*6:8s}  {'-'*6:8s}  {'-'*14:16s}")
    for rank, name in enumerate(sorted_methods, 1):
        vals = method_results[name]
        svals = method_ssims[name]
        tvals = method_times[name]
        if not vals:
            continue
        mean_p = np.mean(vals)
        mean_s = np.mean(svals)
        mean_t = np.mean(tvals)
        delta = mean_p - avg_noisy_psnr

        if name in ('DnCNN', 'DnCNN+Edge'):
            cat = 'Deep Learning'
        elif name == 'EdgeOnly':
            cat = 'Enhancement'
        else:
            cat = 'Traditional'

        badge = medals[rank - 1] if rank <= 3 else '   '
        print(f"  {badge} {rank:<2d}  {name:<20s}  {mean_p:>7.2f}dB  {delta:+>+7.2f}dB  {mean_s:>7.4f}  {mean_t:>7.3f}s  {cat:<16s}")

    print('=' * 120)

    # ?? recommendation ??
    best_method = sorted_methods[0]
    best_psnr = np.mean(method_results[best_method])
    best_ssim = np.mean(method_ssims[best_method])
    best_time = np.mean(method_times[best_method])
    print()
    print(f"  *** RECOMMENDATION ***")
    print(f"  {'-'*60}")
    print(f"  Best method      : {best_method}")
    print(f"  Best PSNR        : {best_psnr:.2f} dB  (noisy baseline: {avg_noisy_psnr:.2f} dB)")
    print(f"  Net improvement  : +{best_psnr - avg_noisy_psnr:.2f} dB")
    print(f"  Best SSIM        : {best_ssim:.4f}")
    print(f"  Avg inference    : {best_time:.3f}s per image")
    if n >= 2:
        runner_up = sorted_methods[1]
        gap = best_psnr - np.mean(method_results[runner_up])
        if gap > 0.5:
            print(f"  Margin over 2nd  : +{gap:.2f} dB  (>> {best_method} dominates)")
        elif gap > 0:
            print(f"  Margin over 2nd  : +{gap:.2f} dB  (close race)")
        else:
            print(f"  Margin over 2nd  : {gap:+.2f} dB")
    print()

    # ?? save CSV ??
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"benchmark_full_{ts}.csv"
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nResults saved: {args.output}")


if __name__ == '__main__':
    main()
