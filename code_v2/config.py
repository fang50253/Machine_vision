# 配置文件
import torch
import os

# 模型配置
MAX_PIXEL = 1024
NUM_LAYERS = 17

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 噪声配置
NOISE_INTENSITIES = {
    'gaussian': 25,
    'salt_pepper': 25,
    'poisson': 25,
    'speckle': 25
}

# 随机噪声配置
RANDOM_NOISE_CONFIG = {
    'enabled': True,  # 是否启用随机噪声
    'gaussian_range': (10, 50),    # 高斯噪声范围 [min, max]
    'salt_pepper_range': (5, 30),  # 椒盐噪声范围
    'poisson_range': (10, 40),     # 泊松噪声范围
    'speckle_range': (10, 40),     # 散斑噪声范围
    'mixed_noise_prob': 0.3,       # 使用混合噪声的概率
}

# 文件路径配置
MODEL_PATHS = ["improved_models", "trained_models", "models"]
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']