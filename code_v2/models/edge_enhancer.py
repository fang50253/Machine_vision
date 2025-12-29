"""
edge_enhancer_simple.py
简化但有效的边缘增强网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeEnhancementNetwork(nn.Module):
    """简化边缘增强网络 - 保证能运行"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        print(f"初始化网络: in_channels={in_channels}, base_channels={base_channels}")
        
        # ========== 边缘提取模块 ==========
        self.edge_extractor = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 第二层
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 第三层
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 输出层 - 输出边缘特征
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh()  # 输出在[-1, 1]之间，表示边缘增强量
        )
        
        # ========== 细节恢复模块 ==========
        self.detail_restorer = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 第二层
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 输出层 - 输出细节恢复量
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
        
        # ========== 自适应融合模块 ==========
        # 输入: [原始图像(3) + 边缘特征(3) + 细节恢复(3)] = 9通道
        self.fusion_net = nn.Sequential(
            nn.Conv2d(9, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels//2, 1, 3, padding=1),
            nn.Sigmoid()  # 输出在[0, 1]之间
        )
        
    def forward(self, x):
        # 打印输入信息
        batch_size, channels, height, width = x.shape
        # print(f"输入: batch={batch_size}, channels={channels}, size={height}x{width}")
        
        # 1. 提取边缘特征
        edge_features = self.edge_extractor(x)
        # print(f"边缘特征: {edge_features.shape}")
        
        # 2. 恢复细节
        detail_features = self.detail_restorer(x)
        # print(f"细节特征: {detail_features.shape}")
        
        # 3. 计算融合权重
        fusion_input = torch.cat([x, edge_features, detail_features], dim=1)
        # print(f"融合输入: {fusion_input.shape}")
        
        weight = self.fusion_net(fusion_input)
        # print(f"融合权重: {weight.shape}")
        
        # 4. 自适应融合
        # 公式: output = input + edge_strength * edge_features + detail_strength * detail_features
        enhanced = x + edge_features * weight + detail_features * (1 - weight)
        # print(f"增强输出: {enhanced.shape}")
        
        return enhanced, edge_features

class EdgeLoss(nn.Module):
    """简化的边缘损失函数"""
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # 重建损失权重
        self.beta = beta    # 边缘保持损失权重
        self.gamma = gamma  # 边缘图监督权重
        
    def forward(self, output, target, edge_map=None):
        # 重建损失 (L1损失，保留细节更好)
        recon_loss = F.l1_loss(output, target)
        
        # 边缘保持损失 - 使用Sobel算子
        # 计算目标图像的边缘
        target_gray = torch.mean(target, dim=1, keepdim=True)
        output_gray = torch.mean(output, dim=1, keepdim=True)
        
        # 手动计算Sobel梯度
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2], 
                                  [-1, 0, 1]]]], dtype=torch.float32, device=target.device)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]]], dtype=torch.float32, device=target.device)
        
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        output_grad_x = F.conv2d(output_gray, sobel_x, padding=1)
        output_grad_y = F.conv2d(output_gray, sobel_y, padding=1)
        output_edges = torch.sqrt(output_grad_x**2 + output_grad_y**2 + 1e-6)
        
        edge_loss = F.l1_loss(output_edges, target_edges)
        
        # 总损失
        total_loss = self.alpha * recon_loss + self.beta * edge_loss
        
        # 如果有边缘图，添加边缘监督
        if edge_map is not None:
            edge_map_gray = torch.mean(edge_map, dim=1, keepdim=True)
            edge_supervision_loss = F.l1_loss(edge_map_gray, target_edges)
            total_loss += self.gamma * edge_supervision_loss
        
        return total_loss, recon_loss, edge_loss


# ============================================================================
# 测试函数
# ============================================================================

def test_model():
    """测试模型是否能正常运行"""
    print("=" * 60)
    print("测试边缘增强网络")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = EdgeEnhancerSimple(in_channels=3, base_channels=64).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    test_input = torch.randn(4, 3, 128, 128).to(device)  # batch=4, channels=3, 128x128
    print(f"输入形状: {test_input.shape}")
    
    try:
        with torch.no_grad():
            enhanced, edges = model(test_input)
        print(f"✓ 前向传播成功!")
        print(f"  增强输出形状: {enhanced.shape}")
        print(f"  边缘图形状: {edges.shape}")
        
        # 测试损失函数
        print("\n测试损失函数...")
        criterion = EdgeLossSimple()
        loss, recon_loss, edge_loss = criterion(enhanced, test_input, edges)
        print(f"✓ 损失计算成功!")
        print(f"  总损失: {loss.item():.4f}")
        print(f"  重建损失: {recon_loss.item():.4f}")
        print(f"  边缘损失: {edge_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n" + "=" * 60)
        print("✅ 模型测试通过，可以开始训练!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 模型测试失败，请检查代码!")
        print("=" * 60)