"""
train_edge_enhancer.py
修复版本 - 支持MPS，兼容所有PyTorch版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from models.edge_enhancer import EdgeEnhancementNetwork, EdgeLoss

class EdgeEnhancementDataset(Dataset):
    """边缘增强数据集"""
    def __init__(self, image_dir, transform=None, patch_size=128, augment=True):
        self.image_dir = image_dir
        self.transform = transform
        self.patch_size = patch_size
        self.augment = augment
        
        # 获取所有图像文件
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.image_files.extend(
                [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                 if f.lower().endswith(ext)]
            )
        
        print(f"找到 {len(self.image_files)} 张图像")
    
    def __len__(self):
        return len(self.image_files) * 20  # 每张图像提取多个patch
    
    def __getitem__(self, idx):
        # 随机选择一张图像
        img_idx = idx % len(self.image_files)
        img_path = self.image_files[img_idx]
        
        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                # 如果读取失败，返回随机图像
                return self.__getitem__((idx + 1) % len(self))
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 随机裁剪
            h, w = image.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                top = np.random.randint(0, h - self.patch_size)
                left = np.random.randint(0, w - self.patch_size)
                image = image[top:top+self.patch_size, left:left+self.patch_size]
            else:
                # 调整大小
                image = cv2.resize(image, (self.patch_size, self.patch_size))
            
            # 数据增强：随机旋转和翻转
            if self.augment:
                # 随机旋转
                angle = np.random.choice([0, 90, 180, 270])
                if angle == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # 随机水平翻转
                if np.random.random() > 0.5:
                    image = cv2.flip(image, 1)
                
                # 随机垂直翻转
                if np.random.random() > 0.5:
                    image = cv2.flip(image, 0)
            
            # 添加不同程度的模糊和噪声来模拟输入
            blur_level = np.random.choice([0, 1, 2, 3])
            if blur_level == 1:
                # 轻度模糊
                image_blurred = cv2.GaussianBlur(image, (3, 3), 0.5)
            elif blur_level == 2:
                # 中度模糊
                image_blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
            elif blur_level == 3:
                # 强模糊
                image_blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
            else:
                image_blurred = image.copy()
            
            # 添加轻微噪声
            noise_level = np.random.uniform(0, 8)
            noise = np.random.randn(*image_blurred.shape) * noise_level
            input_image = np.clip(image_blurred + noise, 0, 255).astype(np.uint8)
            
            # 转换为tensor
            if self.transform:
                input_tensor = self.transform(input_image)
                target_tensor = self.transform(image)
            else:
                # 默认转换：归一化到[0, 1]
                input_tensor = torch.from_numpy(input_image.astype(np.float32) / 255.0).permute(2, 0, 1)
                target_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
            
            return input_tensor, target_tensor
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            # 返回一个随机生成的图像作为备用
            dummy_image = np.random.randint(0, 255, (self.patch_size, self.patch_size, 3), dtype=np.uint8)
            dummy_tensor = torch.from_numpy(dummy_image.astype(np.float32) / 255.0).permute(2, 0, 1)
            return dummy_tensor, dummy_tensor

def setup_device():
    """设置设备，优先使用MPS/GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 使用 CUDA (NVIDIA GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # 检查MPS是否真的可用
        try:
            # 测试MPS
            test_tensor = torch.tensor([1.0], device='mps')
            _ = test_tensor * 2
            device = torch.device('mps')
            print("🚀 使用 MPS (Apple Silicon GPU)")
        except Exception as e:
            print(f"⚠️ MPS测试失败，使用CPU: {e}")
            device = torch.device('cpu')
            print("⚙️ 使用 CPU")
    else:
        device = torch.device('cpu')
        print("⚙️ 使用 CPU")
    
    return device

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_edge_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs, edge_maps = model(inputs)
        
        # 计算损失
        loss, recon_loss, edge_loss = criterion(outputs, targets, edge_maps)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_edge_loss += edge_loss.item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Recon': f'{recon_loss.item():.4f}',
            'Edge': f'{edge_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_edge = total_edge_loss / len(dataloader)
    
    return avg_loss, avg_recon, avg_edge

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_edge_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, edge_maps = model(inputs)
            loss, recon_loss, edge_loss = criterion(outputs, targets, edge_maps)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_edge_loss += edge_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_edge = total_edge_loss / len(dataloader)
    
    return avg_loss, avg_recon, avg_edge

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # 如果使用MPS，将模型移到CPU保存
    if next(model.parameters()).device.type == 'mps':
        checkpoint = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                     for k, v in checkpoint.items()}
    
    torch.save(checkpoint, save_path)
    print(f"✅ 检查点保存到: {save_path}")

def visualize_results(model, dataloader, device, num_samples=3, save_dir='visualizations'):
    """可视化结果"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, edge_maps = model(inputs)
            
            # 转换回numpy
            input_np = inputs[0].cpu().numpy().transpose(1, 2, 0)
            target_np = targets[0].cpu().numpy().transpose(1, 2, 0)
            output_np = outputs[0].cpu().numpy().transpose(1, 2, 0)
            edge_np = edge_maps[0].cpu().numpy().transpose(1, 2, 0)
            
            # 裁剪到[0, 1]
            input_np = np.clip(input_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            output_np = np.clip(output_np, 0, 1)
            edge_np = np.clip(edge_np, 0, 1)
            
            # 创建对比图
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(input_np)
            axes[0, 0].set_title('Input (Blurred+Noisy)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(target_np)
            axes[0, 1].set_title('Target (Original)')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(output_np)
            axes[1, 0].set_title('Enhanced Output')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(edge_np.mean(axis=2), cmap='gray')
            axes[1, 1].set_title('Edge Map')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✅ 可视化结果保存到: {save_dir}")

def plot_training_curves(train_history, val_history, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.plot(val_history['loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_history['recon'], label='Train Recon')
    plt.plot(val_history['recon'], label='Val Recon')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(train_history['edge'], label='Train Edge')
    plt.plot(val_history['edge'], label='Val Edge')
    plt.title('Edge Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    """主训练函数"""
    # ========== 配置参数 ==========
    config = {
        'image_dir': 'datasets/train_images',  # 训练图像目录
        'val_dir': 'datasets/val_images',      # 验证图像目录
        'batch_size': 16,
        'num_epochs': 50,                      # 减少epochs以便快速测试
        'learning_rate': 1e-4,
        'patch_size': 128,
        'save_dir': 'checkpoints/edge_enhancer',
        'visualization_dir': 'training_visualizations'
    }
    
    print("=" * 60)
    print("         边缘增强网络训练")
    print("=" * 60)
    
    # ========== 创建保存目录 ==========
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['visualization_dir'], exist_ok=True)
    
    # ========== 设置设备 ==========
    device = setup_device()
    
    # ========== 数据转换 ==========
    # 简单的数据增强
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),  # 直接使用numpy数组，跳过PIL转换
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # ========== 创建数据集 ==========
    print("\n📊 加载数据集...")
    train_dataset = EdgeEnhancementDataset(
        config['image_dir'], 
        transform=train_transform,
        patch_size=config['patch_size'],
        augment=True
    )
    
    val_dataset = EdgeEnhancementDataset(
        config['val_dir'],
        transform=val_transform,
        patch_size=config['patch_size'],
        augment=False
    )
    
    # ========== 创建数据加载器 ==========
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # macOS上设为0避免问题
        pin_memory=device.type != 'mps'  # MPS设备不支持pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=device.type != 'mps'
    )
    
    print(f"训练集: {len(train_dataset)} patches")
    print(f"验证集: {len(val_dataset)} patches")
    print(f"批量大小: {config['batch_size']}")
    
    # ========== 创建模型 ==========
    print("\n🤖 创建模型...")
    model = EdgeEnhancementNetwork().to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # ========== 损失函数和优化器 ==========
    criterion = EdgeLoss(alpha=1.0, beta=0.5, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 修复：使用兼容的scheduler参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # ========== 训练历史 ==========
    train_history = {'loss': [], 'recon': [], 'edge': []}
    val_history = {'loss': [], 'recon': [], 'edge': []}
    
    # ========== 训练循环 ==========
    print("\n🚀 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_recon, train_edge = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_history['loss'].append(train_loss)
        train_history['recon'].append(train_recon)
        train_history['edge'].append(train_edge)
        
        # 验证
        val_loss, val_recon, val_edge = validate(model, val_loader, criterion, device)
        val_history['loss'].append(val_loss)
        val_history['recon'].append(val_recon)
        val_history['edge'].append(val_edge)
        
        print(f"\n📈 训练损失: {train_loss:.4f} (Recon: {train_recon:.4f}, Edge: {train_edge:.4f})")
        print(f"📊 验证损失: {val_loss:.4f} (Recon: {val_recon:.4f}, Edge: {val_edge:.4f})")
        print(f"📉 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config['save_dir']}/best_model.pth"
            )
            print(f"✨ 新的最佳模型保存 (损失: {val_loss:.4f})")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config['save_dir']}/checkpoint_epoch_{epoch}.pth"
            )
            print(f"💾 定期检查点保存")
        
        # 定期可视化
        if epoch % 20 == 0 or epoch == 1:
            visualize_results(
                model, val_loader, device, 
                save_dir=f"{config['visualization_dir']}/epoch_{epoch}"
            )
        
        # 定期绘制训练曲线
        if epoch % 5 == 0:
            plot_training_curves(
                train_history, val_history,
                f"{config['save_dir']}/training_curves.png"
            )
    
    # ========== 训练完成 ==========
    print("\n" + "="*60)
    print("🎉 训练完成!")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}")
    print(f"💾 模型保存到: {config['save_dir']}")
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, config['num_epochs'], val_loss,
        f"{config['save_dir']}/final_model.pth"
    )
    
    # 绘制最终训练曲线
    plot_training_curves(
        train_history, val_history,
        f"{config['save_dir']}/final_training_curves.png"
    )
    
    print("="*60)

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()