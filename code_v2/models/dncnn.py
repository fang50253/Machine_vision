# models/dncnn.py
import torch
import torch.nn as nn
from config import NUM_LAYERS

class DnCNN(nn.Module):
    """
    DnCNN 去噪模型
    参考: https://arxiv.org/abs/1608.03981
    """
    
    def __init__(self, channels=3, num_layers=NUM_LAYERS, num_features=64):
        super(DnCNN, self).__init__()
        
        layers = []
        
        # 第一层: 输入 -> 特征
        layers.append(nn.Conv2d(channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层: 特征 -> 特征
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层: 特征 -> 输出
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        # 残差学习: 输出噪声，然后从输入中减去
        noise = self.dncnn(x)
        return noise

class ImprovedDnCNN(nn.Module):
    """
    改进的 DnCNN 模型
    添加了更多的正则化和优化
    """
    
    def __init__(self, channels=3, num_layers=20, num_features=64, use_residual=True):
        super(ImprovedDnCNN, self).__init__()
        self.use_residual = use_residual
        
        layers = []
        
        # 第一层
        layers.append(nn.Conv2d(channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
            
            # 添加 dropout 防止过拟合
            if i % 4 == 0:
                layers.append(nn.Dropout2d(0.1))
        
        # 最后一层
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            # 残差学习
            noise = self.network(x)
            return x - noise
        else:
            # 直接学习干净图像
            return self.network(x)

class SimpleDnCNN(nn.Module):
    """
    简化的 DnCNN 模型，用于快速测试
    """
    
    def __init__(self, channels=3, num_layers=10, num_features=32):
        super(SimpleDnCNN, self).__init__()
        
        layers = []
        
        # 第一层
        layers.append(nn.Conv2d(channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # 残差学习
        noise = self.network(x)
        return x - noise

def create_dncnn_model(model_type='standard', **kwargs):
    """
    创建 DnCNN 模型的工厂函数
    
    参数:
        model_type: 'standard', 'improved', 'simple'
        **kwargs: 模型参数
    """
    if model_type == 'standard':
        return DnCNN(**kwargs)
    elif model_type == 'improved':
        return ImprovedDnCNN(**kwargs)
    elif model_type == 'simple':
        return SimpleDnCNN(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

# 测试代码
if __name__ == "__main__":
    # 测试模型创建和前向传播
    model = DnCNN(channels=3, num_layers=NUM_LAYERS)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")