#!/usr/bin/env python
"""
server.py
Flask web server — visual interface for DnCNN denoise / train / benchmark.
Serves a single-page HTML UI and provides REST API backends.

Usage:
    python server.py
    # → http://localhost:5000
"""
import argparse, os, sys, io, base64, json, time, math, glob, csv, threading, uuid
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
from flask import (
    Flask, request, jsonify, render_template, send_file, url_for,
)
from werkzeug.utils import secure_filename

# ── ensure project root is on sys.path ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from models.dncnn import DnCNN
from models.edge_enhancer import EdgeEnhancementNetwork

app = Flask(__name__)

# ── config ──
UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'uploads')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
TRANSLATIONS_DIR = os.path.join(PROJECT_ROOT, 'translations')
ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB

# ── i18n: load translations from JSON files ──
_translations: dict[str, dict[str, str]] = {}
for fname in os.listdir(TRANSLATIONS_DIR):
    if not fname.endswith('.json'):
        continue
    lang = fname[:-5]  # strip .json
    fpath = os.path.join(TRANSLATIONS_DIR, fname)
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            _translations[lang] = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load translation {fname}: {e}")
# fallback to empty so t() never crashes
if 'en' not in _translations:
    _translations['en'] = {}
if 'zh' not in _translations:
    _translations['zh'] = {}
_available_langs = sorted(_translations.keys())
print(f"i18n: loaded {len(_translations)} language(s): {', '.join(_available_langs)}")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── task tracking for async operations ──
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

# ── history directory ──
HISTORY_DIR = os.path.join(PROJECT_ROOT, 'history')
HISTORY_IMAGES_DIR = os.path.join(HISTORY_DIR, 'images')
os.makedirs(HISTORY_IMAGES_DIR, exist_ok=True)

_HISTORY_CACHE: dict[str, list[dict]] = {}  # 'denoise'|'benchmark'|'train' -> list
_HISTORY_MAX = 200  # max entries per type

def _load_history(kind: str) -> list[dict]:
    fpath = os.path.join(HISTORY_DIR, f'{kind}_log.json')
    if os.path.exists(fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _save_history(kind: str, entries: list[dict]) -> None:
    fpath = os.path.join(HISTORY_DIR, f'{kind}_log.json')
    _HISTORY_CACHE[kind] = entries[-_HISTORY_MAX:]
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(_HISTORY_CACHE[kind], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'Warning: failed to save {kind} history: {e}')

def _append_history(kind: str, entry: dict) -> None:
    entries = _load_history(kind)
    entries.append(entry)
    _save_history(kind, entries)


# ═══════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _safe_filename(name: str) -> str:
    name = secure_filename(name)
    if not name:
        name = f'upload_{uuid.uuid4().hex[:8]}'
    stem, ext = os.path.splitext(name)
    if ext.lower() not in ALLOWED_EXT:
        ext = '.png'
    return stem + ext


# ── model scanning ──

def scan_models() -> list[dict]:
    """Scan trained_models/ and group by type."""
    found: list[dict] = []
    if not os.path.isdir(MODEL_DIR):
        return found
    for fname in sorted(os.listdir(MODEL_DIR)):
        fpath = os.path.join(MODEL_DIR, fname)
        if not fname.endswith('.pth') or not os.path.isfile(fpath):
            continue
        low = fname.lower()
        if low.startswith('joint'):
            mtype = 'joint'
        elif low.startswith('edge'):
            mtype = 'edge'
        elif low.startswith('dncnn'):
            mtype = 'dncnn'
        else:
            mtype = 'unknown'
        found.append({
            'name': fname,
            'path': fpath,
            'type': mtype,
            'size': os.path.getsize(fpath),
            'mtime': datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
        })
    return found


# ── model loading ──

def load_dncnn(ckpt_path: str, device: torch.device) -> DnCNN:
    import torch
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


def load_edge(ckpt_path: str, device: torch.device) -> EdgeEnhancementNetwork:
    import torch
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
    import torch
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


# ── denoise inference ──

def _compute_padding(h: int, w: int, has_edge: bool) -> tuple[int, int]:
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if has_edge:
        pad_h += (32 - (h + pad_h) % 32) % 32
        pad_w += (32 - (w + pad_w) % 32) % 32
    return pad_h, pad_w


def _preprocess(image: np.ndarray, pad: tuple[int, int]) -> np.ndarray:
    ph, pw = pad
    if ph or pw:
        image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT101)
    return image.astype(np.float32) / 255.0


def _postprocess(tensor, pad: tuple[int, int], h: int, w: int) -> np.ndarray:
    out = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    ph, pw = pad
    if ph or pw:
        out = out[:h, :w]
    return out


@torch.no_grad()
def _forward_dncnn(model, inp_t, strength: float = 1.0):
    noise_pred = model(inp_t)
    return inp_t - noise_pred * strength


@torch.no_grad()
def _forward_dncnn_edge(model, edge_net, inp_t, strength: float = 1.0):
    noise_pred = model(inp_t)
    denoised_t = inp_t - noise_pred * strength
    enhanced_t, _ = edge_net(denoised_t)
    return enhanced_t


def run_inference(image_rgb: np.ndarray, model, edge_net=None,
                  device=None, strength: float = 1.0, tta: bool = False,
                  uncertainty: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Run denoising inference. Returns (result_rgb, variance_map_or_None)."""
    import torch
    if device is None:
        device = config.DEVICE

    h, w = image_rgb.shape[:2]
    has_edge = edge_net is not None
    pad = _compute_padding(h, w, has_edge)
    inp = _preprocess(image_rgb, pad)
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # ── TTA / uncertainty path ──
    if tta or uncertainty:
        import torch
        accum = torch.zeros(1, 3, inp_t.shape[2], inp_t.shape[3], device='cpu')
        accum_sq = torch.zeros(1, 3, inp_t.shape[2], inp_t.shape[3], device='cpu')
        count = 0

        transforms = [
            (lambda x: x, lambda x: x),
            (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
            (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
            (lambda x: torch.flip(torch.flip(x, dims=[3]), dims=[2]),
             lambda x: torch.flip(torch.flip(x, dims=[3]), dims=[2])),
        ]
        for aug_fn, inv_fn in transforms:
            x_t = aug_fn(inp_t)
            if has_edge:
                out_t = _forward_dncnn_edge(model, edge_net, x_t, strength=strength)
            else:
                out_t = _forward_dncnn(model, x_t, strength=strength)
            out_aligned = inv_fn(out_t).cpu().float()
            accum += out_aligned
            accum_sq += out_aligned ** 2
            count += 1

        mean_t = accum / count
        result = _postprocess(mean_t, pad, h, w)

        var_map = None
        if uncertainty:
            var_t = torch.clamp((accum_sq / count) - mean_t ** 2, min=0.0)
            var_np = var_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            if pad[0] or pad[1]:
                var_np = var_np[:h, :w]
            var_map = var_np.astype(np.float32)
        return result, var_map

    # ── single pass ──
    if has_edge:
        out_t = _forward_dncnn_edge(model, edge_net, inp_t, strength=strength)
    else:
        out_t = _forward_dncnn(model, inp_t, strength=strength)
    result = _postprocess(out_t, pad, h, w)
    return result, None


# ── tiled inference ──

def _tile_weight_map(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
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


def _tile_process(image_rgb: np.ndarray, tile_size: int, overlap: int,
                  model, edge_net=None, device=None, strength=1.0) -> np.ndarray:
    H, W = image_rgb.shape[:2]
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
            tile = image_rgb[y1:y2, x1:x2].copy()
            pad_b = tile_size - (y2 - y1)
            pad_r = tile_size - (x2 - x1)
            if pad_b or pad_r:
                tile = cv2.copyMakeBorder(tile, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT101)
            processed, _ = run_inference(tile, model, edge_net, device, strength)
            if pad_b or pad_r:
                processed = processed[:tile_size - pad_b, :tile_size - pad_r]
            w = weight_map_3[:y2 - y1, :x2 - x1]
            accum[y1:y2, x1:x2] += processed.astype(np.float64) * w
            weight_accum[y1:y2, x1:x2] += weight_map[:y2 - y1, :x2 - x1]

    weight_accum = np.maximum(weight_accum, 1e-10)
    return (accum / weight_accum[:, :, None]).astype(np.uint8)


# ═══════════════════════════════════════════════
#  routes
# ═══════════════════════════════════════════════

@app.route('/')
def index():
    # detect language
    lang = request.cookies.get('lang', '')
    if lang not in _available_langs:
        accept = request.headers.get('Accept-Language', 'en')
        lang = 'zh' if accept and 'zh' in accept else 'en'

    # current-lang translations for Jinja2 rendering
    t_cur = _translations.get(lang, {}) or _translations.get('en', {}) or {}
    # all translations for JS runtime language switching
    all_t = {}
    for l in _available_langs:
        all_t[l] = _translations.get(l, {})
    return render_template('index.html',
                           lang=lang,
                           t=t_cur,
                           all_t_json=json.dumps(all_t, ensure_ascii=False),
                           available_langs=_available_langs)


# ── list models ──

@app.route('/api/models')
def api_models():
    models = scan_models()
    return jsonify({'models': models, 'count': len(models)})


# ── denoise ──

@app.route('/api/denoise', methods=['POST'])
def api_denoise():
    import torch

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    # save uploaded file
    fname = _safe_filename(file.filename)
    in_path = os.path.join(UPLOAD_DIR, fname)
    file.save(in_path)

    # read image
    img_bgr = cv2.imread(in_path)
    if img_bgr is None:
        return jsonify({'error': f'Cannot read image: {fname}'}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # parse params
    model_path = request.form.get('model', '')
    edge_path = request.form.get('edge_model', '') or None
    joint_path = request.form.get('joint_model', '') or None
    strength = float(request.form.get('strength', 1.0))
    do_tta = request.form.get('tta', '0') == '1'
    do_uncertainty = request.form.get('uncertainty', '0') == '1'
    resize_max = int(request.form.get('resize', 0))
    tile_str = request.form.get('tile', '') or ''

    device = config.DEVICE

    # ── load model ──
    try:
        model = None
        edge_net = None
        if joint_path:
            model, edge_net = load_joint(joint_path, device)
            model_label = f"joint: {os.path.basename(joint_path)}"
        elif model_path:
            model = load_dncnn(model_path, device)
            if edge_path:
                edge_net = load_edge(edge_path, device)
                model_label = f"DnCNN+Edge: {os.path.basename(model_path)}"
            else:
                model_label = f"DnCNN: {os.path.basename(model_path)}"
        else:
            return jsonify({'error': 'No model specified'}), 400
    except Exception as e:
        return jsonify({'error': f'Model load failed: {e}'}), 500

    # ── resize ──
    if resize_max > 0 and max(img_rgb.shape[:2]) > resize_max:
        h, w = img_rgb.shape[:2]
        scale = resize_max / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    # ── run inference ──
    try:
        if tile_str:
            parts = tile_str.split(',')
            ts = int(parts[0])
            ov = int(parts[1]) if len(parts) > 1 else max(1, ts // 8)
            duration = time.time()
            result = _tile_process(img_rgb, ts, ov, model, edge_net, device, strength)
            duration = time.time() - duration
            var_map = None
        else:
            duration = time.time()
            result, var_map = run_inference(
                img_rgb, model, edge_net, device,
                strength=strength, tta=do_tta, uncertainty=do_uncertainty,
            )
            duration = time.time() - duration
    except Exception as e:
        return jsonify({'error': f'Inference failed: {e}'}), 500

    # ── metrics ──
    denoised_psnr = psnr(img_rgb, result)

    # ── encode result ──
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', result_bgr)
    result_b64 = base64.b64encode(buf).decode('utf-8')

    # ── encode uncertainty heatmap ──
    uncertainty_b64 = None
    if var_map is not None:
        gray_var = np.mean(var_map, axis=2)
        vmin, vmax = np.percentile(gray_var, [1, 99])
        if vmax - vmin < 1e-10:
            vmax = vmin + 1e-6
        norm = np.clip((gray_var - vmin) / (vmax - vmin), 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        # overlay on result
        overlay = (result * 0.5 + heat_rgb * 0.5).astype(np.uint8)
        _, buf2 = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        uncertainty_b64 = base64.b64encode(buf2).decode('utf-8')

    # ── encode input thumbnail ──
    _, buf3 = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    input_b64 = base64.b64encode(buf3).decode('utf-8')

    # ── save to history ──
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    hid = f'denoise_{ts}'
    try:
        cv2.imwrite(os.path.join(HISTORY_IMAGES_DIR, f'{hid}_input.jpg'), img_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        cv2.imwrite(os.path.join(HISTORY_IMAGES_DIR, f'{hid}_output.png'),
                    cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        has_unc = bool(uncertainty_b64)
        if has_unc:
            cv2.imwrite(os.path.join(HISTORY_IMAGES_DIR, f'{hid}_uncertainty.png'),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        _append_history('denoise', {
            'id': hid,
            'timestamp': ts,
            'model': model_label,
            'strength': strength,
            'tta': do_tta,
            'uncertainty': do_uncertainty,
            'tile': tile_str,
            'psnr': round(denoised_psnr, 2),
            'duration': round(duration, 3),
            'width': result.shape[1],
            'height': result.shape[0],
            'input_image': f'{hid}_input.jpg',
            'output_image': f'{hid}_output.png',
            'has_uncertainty': has_unc,
        })
    except Exception as e:
        print(f'Warning: failed to save denoise history: {e}')

    # cleanup uploaded file
    try:
        os.remove(in_path)
    except OSError:
        pass

    return jsonify({
        'result_image': result_b64,
        'input_image': input_b64,
        'uncertainty_image': uncertainty_b64,
        'psnr': round(denoised_psnr, 2),
        'duration': round(duration, 3),
        'width': result.shape[1],
        'height': result.shape[0],
        'model': model_label,
    })


# ── benchmark ──

@app.route('/api/benchmark', methods=['POST'])
def api_benchmark():
    import torch

    data_dir = request.json.get('data', '')
    model_path = request.json.get('model', '')
    edge_path = request.json.get('edge_model', '') or None
    joint_path = request.json.get('joint_model', '') or None
    sigma = float(request.json.get('sigma', 25))
    max_images = int(request.json.get('max_images', 0))
    use_tta = request.json.get('tta', False)

    if not data_dir or not os.path.isdir(data_dir):
        return jsonify({'error': f'Data directory not found: {data_dir}'}), 400
    if not model_path and not joint_path:
        return jsonify({'error': 'No model specified'}), 400

    device = config.DEVICE

    # ── load model ──
    try:
        if joint_path:
            model, edge_net = load_joint(joint_path, device)
        else:
            model = load_dncnn(model_path, device)
            edge_net = load_edge(edge_path, device) if edge_path else None
    except Exception as e:
        return jsonify({'error': f'Model load failed: {e}'}), 500

    # ── collect images ──
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    files = sorted(set(files))
    if not files:
        return jsonify({'error': 'No images found in data directory'}), 400
    if max_images > 0:
        files = files[:max_images]

    # ── run ──
    results = []
    psnr_list = []
    _t0 = time.time()

    for idx, fpath in enumerate(files):
        bgr = cv2.imread(fpath)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        # add noise
        noise = np.random.randn(*rgb.shape).astype(np.float32) * sigma
        noisy = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # denoise
        out, _ = run_inference(noisy, model, edge_net, device,
                               strength=1.0, tta=use_tta)
        p = psnr(rgb, out)
        psnr_list.append(p)
        results.append({
            'image': os.path.basename(fpath),
            'psnr': round(p, 2),
            'size': f'{w}x{h}',
        })

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0

    # ── save to history ──
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _append_history('benchmark', {
        'id': f'bench_{ts}',
        'timestamp': ts,
        'model': os.path.basename(joint_path or model_path),
        'sigma': sigma,
        'tta': use_tta,
        'avg_psnr': round(avg_psnr, 2),
        'count': len(results),
        'duration': round(time.time() - _t0, 1),
    })

    return jsonify({
        'results': results,
        'avg_psnr': round(avg_psnr, 2),
        'count': len(results),
        'sigma': sigma,
        'model': os.path.basename(joint_path or model_path),
        'tta': use_tta,
    })


# ── benchmark compare (multi-method) ──

_TRADITIONAL_FN = {
    'bilateral': lambda img: cv2.bilateralFilter(img, 9, 75, 75),
    'gaussian':  lambda img: cv2.GaussianBlur(img, (5, 5), 1.0),
    'median':    lambda img: cv2.medianBlur(img, 5),
    'nlm':       lambda img: cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
}

@app.route('/api/benchmark_compare', methods=['POST'])
def api_benchmark_compare():
    """Run benchmark across multiple selected methods on the same images."""
    import torch
    from models.traditional_denoiser import TraditionalDenoiser

    data_dir = request.json.get('data', '')
    sigma = float(request.json.get('sigma', 25))
    max_images = int(request.json.get('max_images', 0))
    methods = request.json.get('methods', [])
    dncnn_path = request.json.get('dncnn_model', '') or None
    edge_path = request.json.get('edge_model', '') or None

    if not data_dir or not os.path.isdir(data_dir):
        return jsonify({'error': 'Data directory not found'}), 400
    if not methods:
        return jsonify({'error': 'No methods selected'}), 400
    if any(m.startswith('dncnn') for m in methods) and not dncnn_path:
        return jsonify({'error': 'DnCNN model required for deep learning methods'}), 400

    device = config.DEVICE

    # ── load models once ──
    model = None
    edge_net = None
    if dncnn_path:
        try:
            model = load_dncnn(dncnn_path, device)
            if edge_path:
                edge_net = load_edge(edge_path, device)
        except Exception as e:
            return jsonify({'error': f'Model load failed: {e}'}), 500

    # ── collect images ──
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    files = sorted(set(files))
    if not files:
        return jsonify({'error': 'No images found'}), 400
    if max_images > 0:
        files = files[:max_images]

    # ── traditional denoiser (lazy) ──
    _trad = None
    def _get_trad():
        nonlocal _trad
        if _trad is None:
            _trad = TraditionalDenoiser()
        return _trad

    # ── run each method ──
    _t0 = time.time()
    method_results: dict[str, list[dict]] = {m: [] for m in methods}

    for idx, fpath in enumerate(files):
        bgr = cv2.imread(fpath)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        # fixed noise seed for fair comparison across methods
        rng = np.random.RandomState(idx)
        noise = rng.randn(*rgb.shape).astype(np.float32) * sigma
        noisy = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        for m in methods:
            try:
                if m == 'wavelet':
                    out = _get_trad().wavelet_denoise_robust(noisy)
                    out = np.clip(out, 0, 255).astype(np.uint8)
                elif m in _TRADITIONAL_FN:
                    out = _TRADITIONAL_FN[m](noisy)
                elif m == 'dncnn':
                    out, _ = run_inference(noisy, model, None, device, strength=1.0, tta=False)
                elif m == 'dncnn_tta':
                    out, _ = run_inference(noisy, model, None, device, strength=1.0, tta=True)
                elif m == 'dncnn_edge':
                    if edge_net is None:
                        raise ValueError('Edge model not loaded')
                    out, _ = run_inference(noisy, model, edge_net, device, strength=1.0, tta=False)
                elif m == 'dncnn_edge_tta':
                    if edge_net is None:
                        raise ValueError('Edge model not loaded')
                    out, _ = run_inference(noisy, model, edge_net, device, strength=1.0, tta=True)
                else:
                    continue

                p = psnr(rgb, out)
                method_results[m].append({
                    'image': os.path.basename(fpath),
                    'psnr': round(p, 2),
                })
            except Exception as e:
                method_results[m].append({
                    'image': os.path.basename(fpath),
                    'psnr': None,
                    'error': str(e),
                })

    # ── build response ──
    label_map = {
        'bilateral': 'Bilateral', 'gaussian': 'Gaussian', 'median': 'Median',
        'nlm': 'NLM', 'wavelet': 'Wavelet',
        'dncnn': 'DnCNN', 'dncnn_tta': 'DnCNN+TTA',
        'dncnn_edge': 'DnCNN+Edge', 'dncnn_edge_tta': 'DnCNN+Edge+TTA',
    }
    out_methods = []
    for m in methods:
        entries = method_results[m]
        psnr_vals = [e['psnr'] for e in entries if e['psnr'] is not None]
        avg = float(np.mean(psnr_vals)) if psnr_vals else None
        out_methods.append({
            'key': m,
            'name': label_map.get(m, m),
            'avg_psnr': round(avg, 2) if avg is not None else None,
            'results': entries,
        })

    # ── save to history ──
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        _append_history('benchmark', {
            'id': f'bench_{ts}',
            'timestamp': ts,
            'model': 'multi: ' + ', '.join(m for m in methods),
            'sigma': sigma,
            'tta': False,
            'avg_psnr': round(float(np.mean([m['avg_psnr'] for m in out_methods if m['avg_psnr'] is not None])), 2) if any(m['avg_psnr'] for m in out_methods) else 0,
            'count': len(files),
            'duration': round(time.time() - _t0, 1),
        })
    except Exception:
        pass

    return jsonify({
        'methods': out_methods,
        'count': len(files),
        'sigma': sigma,
        'duration': round(time.time() - _t0, 1),
    })


# ── train (async) ──

@app.route('/api/train', methods=['POST'])
def api_train():
    data_dir = request.json.get('data', '')
    val_dir = request.json.get('val', '') or None
    stage = request.json.get('stage', 1)
    epochs = int(request.json.get('epochs', 10))
    batch_size = int(request.json.get('batch_size', 8))
    lr = float(request.json.get('lr', 1e-3))
    sigma = float(request.json.get('sigma', 25))
    eval_sigmas = request.json.get('eval_sigmas', '') or ''
    deep = request.json.get('deep', False)
    dncnn_ckpt = request.json.get('dncnn_model', '') or None
    edge_ckpt = request.json.get('edge_model', '') or None

    if not data_dir or not os.path.isdir(data_dir):
        return jsonify({'error': f'Data directory not found: {data_dir}'}), 400

    task_id = uuid.uuid4().hex[:12]
    task_info = {
        'id': task_id,
        'stage': stage,
        'status': 'running',
        'progress': 0,
        'message': 'Starting...',
        'created': time.time(),
    }
    with _tasks_lock:
        _tasks[task_id] = task_info

    def _run_train(task_id, data_dir, val_dir, stage, epochs, batch_size, lr, sigma,
                   dncnn_ckpt, edge_ckpt):
        try:
            import sys, subprocess
            with _tasks_lock:
                _tasks[task_id]['message'] = 'Preparing...'

            if stage == 3:
                # Joint fine-tuning via train.py
                if not dncnn_ckpt or not edge_ckpt:
                    with _tasks_lock:
                        _tasks[task_id]['status'] = 'error'
                        _tasks[task_id]['message'] = 'Stage 3 requires --dncnn and --edge'
                    return
                script = os.path.join(PROJECT_ROOT, 'train.py')
                cmd = [
                    sys.executable, script,
                    '--data', data_dir,
                    '--dncnn', dncnn_ckpt,
                    '--edge', edge_ckpt,
                    '--epochs', str(epochs),
                    '--batch-size', str(batch_size),
                    '--lr', str(lr),
                ]
                if val_dir:
                    cmd.extend(['--val', val_dir])

            elif stage == 2:
                script = os.path.join(PROJECT_ROOT, 'training', 'train_edge.py')
                if not dncnn_ckpt:
                    with _tasks_lock:
                        _tasks[task_id]['status'] = 'error'
                        _tasks[task_id]['message'] = 'Stage 2 requires --dncnn'
                    return
                cmd = [
                    sys.executable, script,
                    '--data', data_dir,
                    '--dncnn', dncnn_ckpt,
                    '--epochs', str(epochs),
                ]
                if val_dir:
                    cmd.extend(['--val', val_dir])
            else:  # stage 1
                script = os.path.join(PROJECT_ROOT, 'training', 'train_dncnn.py')
                cmd = [
                    sys.executable, script,
                    '--data', data_dir,
                    '--epochs', str(epochs),
                    '--batch-size', str(batch_size),
                    '--sigma', str(sigma),
                    '--lr', str(lr),
                ]
                if val_dir:
                    cmd.extend(['--val', val_dir])
                if eval_sigmas:
                    cmd.extend(['--eval-sigmas', eval_sigmas])
                if deep:
                    cmd.append('--deep')

            with _tasks_lock:
                _tasks[task_id]['message'] = f'Running: {" ".join(cmd[-8:])}'

            # run subprocess (unbuffered for real-time output)
            _env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=PROJECT_ROOT, env=_env,
            )
            output_lines = []
            _has_metrics = False
            for line in proc.stdout:
                line = line.rstrip()

                # structured metrics → update status + progress
                if line.startswith('__METRICS__'):
                    try:
                        md = json.loads(line[11:])
                        with _tasks_lock:
                            if 'metrics' not in _tasks[task_id]:
                                _tasks[task_id]['metrics'] = []
                            _tasks[task_id]['metrics'].append(md)
                            _tasks[task_id]['progress'] = int(
                                md.get('epoch', 0) / epochs * 100
                            ) if 'epoch' in md else 0
                            # Human-readable message
                            parts = [f'Epoch {md.get("epoch", "?")}/{epochs}']
                            tl = md.get('train_loss')
                            parts.append(f'train={tl:.4e}' if isinstance(tl, float) else f'train={tl}')
                            vl = md.get('val_loss')
                            parts.append(f'val={vl:.4e}' if isinstance(vl, float) else f'val={vl}')
                            lr = md.get('lr')
                            if isinstance(lr, float): parts.append(f'lr={lr:.2e}')
                            sig = md.get('sigma')
                            if sig is not None: parts.append(f'sigma={sig:.1f}')
                            for k, v in md.items():
                                if k.startswith('psnr_') and isinstance(v, (int, float)):
                                    parts.append(f'{k}={v:.1f}dB')
                            _tasks[task_id]['message'] = '  '.join(parts)
                        _has_metrics = True
                    except Exception:
                        pass
                    continue

                # Raw stdout line — show only if no metrics yet (avoids clutter)
                output_lines.append(line)
                if not _has_metrics:
                    with _tasks_lock:
                        _tasks[task_id]['message'] = line[-200:] if len(line) > 200 else line

            proc.wait()
            exit_code = proc.returncode
            output_text = '\n'.join(output_lines[-50:])

            with _tasks_lock:
                if exit_code == 0:
                    _tasks[task_id]['status'] = 'completed'
                    _tasks[task_id]['progress'] = 100
                    _tasks[task_id]['message'] = f'Stage {stage} training completed successfully'
                    _tasks[task_id]['output'] = output_text
                else:
                    _tasks[task_id]['status'] = 'error'
                    _tasks[task_id]['message'] = f'Training failed (exit code {exit_code})'
                    _tasks[task_id]['output'] = output_text

            # ── save to history ──
            try:
                _append_history('train', {
                    'id': f'train_{task_id}',
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'stage': stage,
                    'data': data_dir,
                    'val': val_dir,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'sigma': sigma,
                    'status': 'completed' if exit_code == 0 else 'error',
                    'message': _tasks[task_id].get('message', ''),
                })
            except Exception as e:
                print(f'Warning: failed to save train history: {e}')

        except Exception as e:
            with _tasks_lock:
                _tasks[task_id]['status'] = 'error'
                _tasks[task_id]['message'] = str(e)
            # ── save error to history ──
            try:
                _append_history('train', {
                    'id': f'train_{task_id}',
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'stage': stage,
                    'data': data_dir,
                    'val': val_dir,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'sigma': sigma,
                    'status': 'error',
                    'message': str(e),
                })
            except Exception:
                pass

    t = threading.Thread(target=_run_train, args=(
        task_id, data_dir, val_dir, stage, epochs, batch_size, lr, sigma,
        dncnn_ckpt, edge_ckpt,
    ), daemon=True)
    t.start()

    return jsonify({'task_id': task_id, 'status': 'running'})


@app.route('/api/task/<task_id>')
def api_task_status(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


# ── train pipeline (3 stages sequential) ──

@app.route('/api/train_pipeline', methods=['POST'])
def api_train_pipeline():
    data_dir = request.json.get('data', '')
    val_dir = request.json.get('val', '') or None
    batch_size = int(request.json.get('batch_size', 8))

    s1 = request.json.get('stage1', {})
    s2 = request.json.get('stage2', {})
    s3 = request.json.get('stage3', {})

    if not data_dir or not os.path.isdir(data_dir):
        return jsonify({'error': f'Data directory not found: {data_dir}'}), 400

    task_id = uuid.uuid4().hex[:12]
    task_info = {
        'id': task_id, 'status': 'running', 'progress': 0,
        'stage': 0, 'message': 'Preparing pipeline...',
        'created': time.time(), 'pipeline': True,
        'metrics': [],
    }
    with _tasks_lock:
        _tasks[task_id] = task_info

    def _run_pipeline():
        import subprocess, sys, re
        from pathlib import Path

        def _update(**kw):
            with _tasks_lock:
                for k, v in kw.items():
                    _tasks[task_id][k] = v

        def _find_latest(pattern: str) -> str | None:
            """Find most recent file matching glob pattern."""
            files = glob.glob(os.path.join(MODEL_DIR, pattern))
            if files:
                return max(files, key=os.path.getmtime)
            return None

        def _run_subprocess(cmd: list[str], stage: int, total_epochs_this_stage: int):
            """Run a subprocess, capture __METRICS__ and stdout, return exit code."""
            _env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=PROJECT_ROOT, env=_env,
            )
            output_lines = []
            _has_metrics = False
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith('__METRICS__'):
                    try:
                        md = json.loads(line[11:])
                        with _tasks_lock:
                            _tasks[task_id].setdefault('metrics', []).append(md)
                            epoch = md.get('epoch', 0)
                            train_loss = md.get('train_loss', '?')
                            val_loss = md.get('val_loss', '?')
                            lr = md.get('lr', '?')
                            # Build human-readable message
                            parts = [f'[Stage {stage}] Epoch {epoch}/{total_epochs_this_stage}']
                            parts.append(f'train={train_loss:.4e}' if isinstance(train_loss, float) else f'train={train_loss}')
                            parts.append(f'val={val_loss:.4e}' if isinstance(val_loss, float) else f'val={val_loss}')
                            if isinstance(lr, float):
                                parts.append(f'lr={lr:.2e}')
                            sig = md.get('sigma')
                            if sig is not None:
                                parts.append(f'sigma={sig:.1f}')
                            for k, v in md.items():
                                if k.startswith('psnr_') and isinstance(v, (int, float)):
                                    parts.append(f'{k}={v:.1f}dB')
                            _tasks[task_id]['message'] = '  '.join(parts)
                            progress_in_stage = int(epoch / total_epochs_this_stage * 100) if total_epochs_this_stage > 0 else 0
                            _tasks[task_id]['progress'] = (stage - 1) * 33 + int(progress_in_stage * 0.33)
                        _has_metrics = True
                    except Exception:
                        pass
                    continue  # skip from visible log
                output_lines.append(line)
                if not _has_metrics:
                    _update(message=line[-200:])
            proc.wait()
            return proc.returncode, '\n'.join(output_lines[-50:])

        try:
            # ── Stage 1: DnCNN ──
            _update(stage=1, message='Stage 1: Training DnCNN...')
            out_dir = Path(MODEL_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)

            epochs1 = int(s1.get('epochs', 100))
            cmd1 = [
                sys.executable,
                os.path.join(PROJECT_ROOT, 'training', 'train_dncnn.py'),
                '--data', data_dir,
                '--epochs', str(epochs1),
                '--batch-size', str(batch_size),
                '--sigma', str(s1.get('sigma', 25)),
                '--lr', str(s1.get('lr', 1e-3)),
                '--out', str(out_dir),
            ]
            if val_dir:
                cmd1.extend(['--val', val_dir])
            if s1.get('eval_sigmas'):
                cmd1.extend(['--eval-sigmas', s1['eval_sigmas']])
            if s1.get('deep'):
                cmd1.append('--deep')

            rc1, out1 = _run_subprocess(cmd1, stage=1, total_epochs_this_stage=epochs1)
            if rc1 != 0:
                _update(status='error', message=f'Stage 1 failed (exit {rc1})', output=out1)
                return

            # Find stage 1 checkpoint
            dncnn_ckpt = _find_latest('dncnn_best_*.pth')
            if not dncnn_ckpt:
                _update(status='error', message='Stage 1 completed but no checkpoint found!', output=out1)
                return
            _update(message=f'Stage 1 done: {os.path.basename(dncnn_ckpt)}')

            # ── Stage 2: EdgeEnhance ──
            _update(stage=2, message='Stage 2: Training EdgeEnhance...')
            epochs2 = int(s2.get('epochs', 50))
            cmd2 = [
                sys.executable,
                os.path.join(PROJECT_ROOT, 'training', 'train_edge.py'),
                '--data', data_dir,
                '--dncnn', dncnn_ckpt,
                '--epochs', str(epochs2),
                '--batch-size', str(batch_size),
                '--lr', str(s2.get('lr', 3e-4)),
                '--out', str(out_dir),
            ]
            if val_dir:
                cmd2.extend(['--val', val_dir])

            rc2, out2 = _run_subprocess(cmd2, stage=2, total_epochs_this_stage=epochs2)
            if rc2 != 0:
                _update(status='error', message=f'Stage 2 failed (exit {rc2})', output=out2)
                return

            edge_ckpt = _find_latest('edge_stage2_best_*.pth')
            if not edge_ckpt:
                _update(status='error', message='Stage 2 completed but no checkpoint found!', output=out2)
                return
            _update(message=f'Stage 2 done: {os.path.basename(edge_ckpt)}')

            # ── Stage 3: Joint fine-tune ──
            _update(stage=3, message='Stage 3: Joint fine-tuning...')
            epochs3 = int(s3.get('epochs', 20))
            cmd3 = [
                sys.executable,
                os.path.join(PROJECT_ROOT, 'train.py'),
                '--data', data_dir,
                '--dncnn', dncnn_ckpt,
                '--edge', edge_ckpt,
                '--epochs', str(epochs3),
                '--batch-size', str(batch_size),
                '--lr', str(s3.get('lr', 5e-5)),
                '--out', str(out_dir),
            ]
            if val_dir:
                cmd3.extend(['--val', val_dir])

            rc3, out3 = _run_subprocess(cmd3, stage=3, total_epochs_this_stage=epochs3)
            if rc3 != 0:
                _update(status='error', message=f'Stage 3 failed (exit {rc3})', output=out3)
                return

            joint_ckpt = _find_latest('joint_best_*.pth')
            ckpt_msg = f'joint model: {os.path.basename(joint_ckpt)}' if joint_ckpt else ''

            # ── success ──
            _update(
                status='completed', progress=100,
                message=f'Pipeline complete! {ckpt_msg}',
                output=f'Stage 1: {os.path.basename(dncnn_ckpt)}\nStage 2: {os.path.basename(edge_ckpt)}\nStage 3: {ckpt_msg}',
            )

            # Save to history
            try:
                _append_history('train', {
                    'id': f'pipeline_{task_id}',
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'stage': 'pipeline',
                    'data': data_dir, 'val': val_dir,
                    'batch_size': batch_size,
                    'stage1_epochs': epochs1, 'stage2_epochs': epochs2, 'stage3_epochs': epochs3,
                    'dncnn_ckpt': dncnn_ckpt, 'edge_ckpt': edge_ckpt, 'joint_ckpt': joint_ckpt,
                    'status': 'completed',
                })
            except Exception as e:
                print(f'Warning: failed to save pipeline history: {e}')

        except Exception as e:
            _update(status='error', message=str(e))

    t = threading.Thread(target=_run_pipeline, daemon=True)
    t.start()
    return jsonify({'task_id': task_id, 'status': 'running', 'pipeline': True})


# ── history ──

@app.route('/api/history')
def api_history():
    kind = request.args.get('kind', '')  # 'denoise'|'benchmark'|'train'|'' (all)
    limit = min(int(request.args.get('limit', 50)), 200)

    result = {}
    if not kind or kind == 'denoise':
        result['denoise'] = _load_history('denoise')[-limit:]
    if not kind or kind == 'benchmark':
        result['benchmark'] = _load_history('benchmark')[-limit:]
    if not kind or kind == 'train':
        result['train'] = _load_history('train')[-limit:]

    return jsonify(result)


@app.route('/api/history/images/<filename>')
def api_history_image(filename: str):
    safe = secure_filename(filename)
    fpath = os.path.join(HISTORY_IMAGES_DIR, safe)
    if not os.path.exists(fpath):
        return jsonify({'error': 'Image not found'}), 404
    return send_file(fpath)


# ── static files (model files from trained_models/) ──

@app.route('/model-file/<path:filename>')
def serve_model_file(filename: str):
    safe = secure_filename(filename)
    return send_file(os.path.join(MODEL_DIR, safe), as_attachment=True)


# ═══════════════════════════════════════════════
#  entry
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='DnCNN Web Server')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address')
    parser.add_argument('--debug', action='store_true', help='Flask debug mode')
    args = parser.parse_args()

    print(f"  DnCNN Web Server")
    print(f"  ─────────────────────")
    print(f"  Device : {config.DEVICE}")
    print(f"  URL    : http://{args.host}:{args.port}")
    print(f"  Models : {MODEL_DIR}")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
