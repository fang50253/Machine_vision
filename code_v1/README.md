# 机器视觉期末作业

```markdown
# DeNoise_GPU - 高级图像去噪系统

一个基于深度学习和传统算法的混合图像去噪系统，支持 GPU 加速和多种去噪方法。

## 📋 项目概述

本项目提供了一个完整的图像去噪解决方案，结合了：
- **深度学习**：改进的 DnCNN 网络架构
- **传统算法**：小波变换、双边滤波、中值滤波等
- **混合方法**：多种算法组合的优化方案
- **锐化增强**：自适应锐化处理

## ✨ 主要特性

### 🎯 核心功能
- ✅ 支持单张图像和批量处理
- ✅ 多种噪声类型模拟（高斯、椒盐、泊松、散斑等）
- ✅ 5种去噪方法对比
- ✅ PSNR/SSIM 质量评估
- ✅ 自适应图像锐化
- ✅ GPU 加速训练和推理

### 🚀 技术亮点
- **混合噪声训练**：模拟真实世界的复杂噪声
- **早停机制**：防止过拟合，优化训练过程
- **数据增强**：丰富的图像增强策略
- **多GPU支持**：自动检测并使用可用GPU
- **可视化分析**：训练曲线和效果对比

## 📁 项目结构

```
DeNoise_GPU/
├── main.py                    # 主程序 - 图像去噪测试
├── train.py                   # 模型训练程序
├── improved_models/           # 训练好的模型保存目录
├── trained_models/            # 训练过程中的模型保存
├── batch_results_*/           # 批量处理结果
├── Show.py                    # 显示和工具函数
└── README.md                  # 项目说明
```

## 🛠 安装要求

### 基础依赖
```bash
pip install torch torchvision opencv-python
pip install numpy matplotlib scikit-image
pip install pandas tqdm pywavelets
```

### GPU 支持（可选）
```bash
# 对于 CUDA 用户
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

# 验证安装
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 🎮 快速开始

### 1. 单张图像去噪测试
```bash
python Show_v1.py # v1 v2具体参考世纪版本信息
```
然后选择：
- 模式 1：单张图像处理
- 选择图像文件
- 选择噪声类型和强度
- 查看5种方法的去噪效果对比

### 2. 批量图像处理
```bash
python Show_v1.py # v1 v2具体参考世纪版本信息
```
选择模式 2，指定包含多张图像的文件夹，系统会自动处理所有图像并生成详细报告。

### 3. 训练新模型
```bash
python Train_GPU_v1.py # v1 v2具体参考世纪版本信息
```
按照提示：
- 指定训练图像文件夹
- 设置训练参数（轮数、批量大小等）
- 等待训练完成，模型将保存在 `improved_models/` 目录

## 📊 去噪方法对比

系统提供5种去噪方法：

| 方法 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| Wavelet | 传统 | 频域去噪，保留细节 | 纹理丰富的图像 |
| Bilateral | 传统 | 边缘保持滤波 | 需要保留边缘的图像 |
| DnCNN | 深度学习 | 深度网络，强去噪能力 | 高噪声水平图像 |
| Hybrid V1 | 混合 | 传统+深度学习+锐化 | 需要增强细节的图像 |
| Hybrid V2 | 混合 | 多种传统方法加权融合 | 平衡去噪和细节保持 |

## ⚙️ 配置参数

### 模型参数
```python
NUM_Laters = 17        # 网络层数
MAX_Pixel = 1024       # 最大处理图像尺寸
num_features = 64      # 网络特征图数量
```

### 训练参数
```python
epochs = 50            # 训练轮数
batch_size = 8         # 批量大小
learning_rate = 0.001  # 学习率
patience = 10          # 早停耐心值
```

### 噪声参数
- **高斯噪声**：强度 1-100
- **椒盐噪声**：密度 0.1%-20%
- **泊松噪声**：基于光子计数模型
- **散斑噪声**：乘性噪声模型

## 🎯 高级功能

### 自适应锐化
```python
# 在混合去噪方法中使用
denoiser.hybrid_denoise_v2(image, sharpen_amount=10, use_adaptive_sharpen=True)
```

### 混合噪声训练
```python
# 支持多种噪声组合
noise_types = ['gaussian', 'salt_pepper', 'poisson', 'speckle']
```

### 早停机制
自动监控验证损失，在性能不再提升时停止训练，保存最佳模型。

## 📈 性能评估

系统自动计算以下指标：
- **PSNR** (峰值信噪比)：衡量去噪效果
- **SSIM** (结构相似性)：评估结构保持能力
- **标准化PSNR**：便于方法间对比

## 🗂 输出文件

### 训练输出
```
improved_models/
├── dncnn_YYYYMMDD_HHMMSS_best.pth    # 最佳模型
├── dncnn_YYYYMMDD_HHMMSS_final.pth   # 最终模型
└── dncnn_YYYYMMDD_HHMMSS_loss_curve.png  # 损失曲线
```

### 批量处理输出
```
batch_results_YYYYMMDD_HHMMSS/
├── denoising_results.csv          # 详细结果表格
└── images/
    ├── image1/
    │   ├── original.jpg
    │   ├── noisy.jpg
    │   ├── Wavelet.jpg
    │   └── ...
    └── image2/
        └── ...
```

## 🔧 故障排除

### 常见问题

1. **GPU 不可用**
   ```bash
   # 检查CUDA安装
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **内存不足**
   - 减小 `batch_size`
   - 降低 `image_size`
   - 使用 `max_samples` 限制训练数据

3. **模型加载失败**
   - 检查模型文件路径
   - 确认模型架构匹配
   - 重新训练模型

### 性能优化建议

- 使用 SSD 存储加速数据加载
- 调整 `num_workers` 优化数据加载
- 使用混合精度训练（需要 GPU 支持）
- 定期清理 GPU 缓存

## 📝 版本历史

### v2025-10-27-17-07
- ✅ 添加 Hybrid_v2 锐化功能
- ✅ 改进自适应锐化算法
- ✅ 优化噪声生成逻辑
- ✅ 修复泊松噪声生成错误
- ✅ 增强 GPU 检测机制

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！
1. Fork 本项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 基于 DnCNN 网络的改进实现
- 使用 OpenCV、PyTorch 等开源库
- 感谢所有贡献者和用户反馈

---

**开始使用**: 运行 `python main.py` 体验图像去噪效果！
```

这个 README.md 文件包含了：
1. **完整的项目介绍** - 让用户快速了解项目
2. **详细的安装和使用指南** - 从安装到使用的完整流程
3. **技术特性说明** - 突出项目的技术亮点
4. **故障排除** - 常见问题的解决方案
5. **版本信息** - 包含你提到的版本更新说明

你可以根据实际需要调整内容，比如添加具体的示例图片、性能基准测试结果等。