# DnCNN Denoising Tool

基于改进 DnCNN 的图像去噪工具，支持训练、单图去噪、数据集基准测试三大功能。

---

## 目录结构

```
code_v4/
├── config.py             # 配置文件（模型超参、训练参数）
├── run.py                # 单张图像去噪（支持 strength/resize/tile/uncertainty）
├── train.py              # 训练：Stage 3 — DnCNN+Edge 端到端联合微调
├── benchmark.py          # 全方法对比基准测试（含传统算法 + TTA）
├── training/             # 分阶段训练脚本
│   ├── train_dncnn.py    # Stage 1: DnCNN 去噪训练
│   └── train_edge.py     # Stage 2: EdgeEnhance 边缘增强训练
├── models/
│   ├── dncnn.py          # DnCNN 网络定义（可选 SE attention）
│   ├── edge_enhancer.py  # 边缘增强网络
│   ├── losses.py         # CombinedLoss（MSE + 感知 + 频率损失）
│   ├── traditional_denoiser.py  # 传统去噪方法
│   └── image_sharpener.py       # 图像锐化
├── results/              # 基准测试 CSV 结果
├── trained_models/       # 训练好的模型存放目录
├── documents/            # 文档（如 .raw 格式说明）
├── legacy/               # 旧版脚本（已归档）
├── samples/              # 样本图像
└── requirements.txt
```

---

## 训练流程（4 阶段）

完整训练分 3 个阶段，由 `training/run_all.cmd` 一键执行：

### Stage 1 — DnCNN 去噪训练

```bash
python training/train_dncnn.py --data ../Div2k/DIV2K_train_HR \
    --val ../Div2k/DIV2K_valid_HR \
    --epochs 100 --batch-size 16 --sigma 25
```

- CosineAnnealingLR、色彩/模糊/缩放增强、课程学习（sigma 从 0 线性 ramp 到目标值）
- 输出：`dncnn_best_{timestamp}.pth`

### Stage 2 — EdgeEnhance 边缘增强训练

```bash
python training/train_edge.py --data ../Div2k/DIV2K_train_HR \
    --val ../Div2k/DIV2K_valid_HR \
    --dncnn trained_models/dncnn_best_*.pth
```

- 以 DnCNN 去噪输出为输入，学习边缘增强
- 输出：`edge_stage2_best_{timestamp}.pth`

### Stage 3 — 联合微调（root `train.py`）

```bash
python train.py --data ../Div2k/DIV2K_train_HR \
    --dncnn trained_models/dncnn_best_*.pth \
    --edge trained_models/edge_stage2_best_*.pth \
    --val ../Div2k/DIV2K_valid_HR
```

- DnCNN + EdgeEnhance 端到端联合微调（小学习率）
- Sobel 梯度 L1 损失增强边缘保留
- 输出：`joint_best_{timestamp}.pth`

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | (必填) | 训练图像文件夹路径 |
| `--dncnn` | (必填) | DnCNN 模型检查点 |
| `--edge` | (必填) | EdgeEnhance 模型检查点 |
| `--val` | None | 验证集路径 |
| `--epochs` | config | 训练轮数 |
| `--batch-size` | config | 批次大小 |
| `--lr` | config.joint_ft_lr | DnCNN 微调学习率 |
| `--seed` | config | 随机种子 |
| `--out` | trained_models | 模型保存目录 |

---

## 2. 单图去噪 `run.py`

支持 DnCNN / DnCNN+Edge / Joint 三种模式，提供强度控制、大图分块、不确定性估计。

```bash
# 基本：DnCNN 去噪
python run.py --model trained_models/dncnn_best.pth --input noisy.png

# DnCNN + Edge 增强
python run.py --model dncnn_best.pth --edge-model edge_best.pth --input photo.png

# Joint 模型（DnCNN+Edge 合一）
python run.py --joint-model joint_best.pth --input photo.png

# 去噪强度控制（0.0=无去噪，1.0=完整，>1.0=过度）
python run.py --model dncnn_best.pth --input noise.png --strength 0.7

# 大图处理：先缩放再推理
python run.py --model dncnn_best.pth --input large.png --resize 1024

# 大图处理：分块推理（256x256 瓦片，32px 重叠）
python run.py --model dncnn_best.pth --input huge.png --tile "256,32"

# 不确定性估计（4-way TTA 方差）
python run.py --joint-model joint_best.pth --input photo.png --uncertainty

# 先加噪声再测试（模拟评估）
python run.py --model dncnn_best.pth --input clean.png --sigma 25 --noise-seed 42
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model`, `-m` | None | DnCNN 模型检查点 |
| `--edge-model` | None | EdgeEnhance 模型检查点 |
| `--joint-model` | None | Joint 模型（包含两个网络） |
| `--input`, `-i` | (必填) | 输入图像路径 |
| `--output`, `-o` | 自动生成 | 输出路径 |
| `--sigma`, `-s` | 0 | 添加高斯噪声标准差 |
| `--noise-seed` | None | 噪声随机种子 |
| `--strength` | 1.0 | 去噪强度（0.0–1.0+） |
| `--resize` | 0 | 最长边缩放到 N 像素 |
| `--tile` | None | 分块推理 "tile_size,overlap" 如 "256,32" |
| `--tta` | False | 启用测试时增强（4-way flip） |
| `--uncertainty` | False | 输出不确定性热图 + .raw 流 |
| `--output`, `-o` | 自动生成 | 输出去噪图像路径 |

---

## 3. 基准测试 `benchmark.py`

对比 DnCNN、DnCNN+EdgeEnhance、双边滤波、小波去噪、NLM 等多种方法。

```bash
# 基本用法
python benchmark.py \
    --model trained_models/dncnn_best.pth \
    --data ../Div2k/DIV2K_valid_HR \
    --sigma 25

# 启用 TTA（4-way flip ensemble）
python benchmark.py --model dncnn_best.pth --data ../Div2k/DIV2K_valid_HR \
    --edge-model edge_best.pth --sigma 25 --tta

# Joint 模型 + TTA
python benchmark.py --joint-model joint_best.pth \
    --data ../Div2k/DIV2K_valid_HR --sigma 25 --tta
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data`, `-d` | (必填) | 测试图像文件夹 |
| `--model`, `-m` | (必填) | DnCNN 模型检查点 |
| `--edge-model` | None | 边缘增强模型 |
| `--joint-model` | None | Joint 模型（优先于单独模型） |
| `--sigma` | 25 | 高斯噪声标准差 |
| `--max-size` | 1024 | 图像最大边长限制 |
| `--max-images` | 0 | 测试图像数量限制（0=全部） |
| `--output`, `-o` | 自动生成 | CSV 输出路径 |
| `--noise-seed` | None | 噪声随机种子 |
| `--noise-type` | gaussian | 噪声分布类型 |
| `--save-samples` | 0 | 保存 N 张对比图 |
| `--tta` | False | 启用测试时增强 |

---

## 当前基准测试结果

在 **DIV2K 验证集 (100 张)** 上，高斯噪声 σ=25 的测试结果：

| 方法 | 平均 PSNR | 平均 SSIM |
|------|-----------|-----------|
| Bilateral | 27.87 dB | 0.8106 |
| DnCNN+Edge | 26.77 dB | 0.7190 |
| **DnCNN** | **26.29 dB** | **0.7063** |
| GaussianBlur | 25.79 dB | 0.7587 |
| Median | 24.51 dB | 0.6879 |
| NLM | 24.23 dB | 0.7200 |
| EdgeOnly | 23.09 dB | 0.5687 |
| Wavelet | 20.91 dB | 0.5479 |

> **注意：** 当前 DnCNN 模型尚未在 DIV2K 上充分训练。完整的 3 阶段训练后 DnCNN 预期可达到 **31~33 dB**。

---

## 训练建议

1. **数据准备**：DIV2K 包含 800 张 2K 高清训练图，直接用 `--data ../Div2k/DIV2K_train_HR`
2. **批量大小**：RTX 3050 Ti (4GB) 建议 batch_size=8~16，patch_size=128
3. **训练时长**：Stage 1 约 2~4 小时，Stage 2+3 各约 30~60 分钟
4. **一键运行**：`training/run_all.cmd` 自动执行全部 4 阶段
5. **验证集**：可以用 DIV2K 的 100 张验证图 `--val ../Div2k/DIV2K_valid_HR`

---

## 参考

- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE TIP.
