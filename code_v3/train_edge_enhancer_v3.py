"""
train_dncnn_to_original.py
训练模型从DnCNN结果恢复到原始图像
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 导入边缘增强网络
from models.edge_enhancer import EdgeEnhancementNetwork, EdgeLoss

class DnCNNtoOriginalDataset(Dataset):
    """
    数据集：DnCNN结果 -> 原始图像
    目录结构：
    batch_results/
        images/
            0801x4d/
                DnCNN.jpg      # 输入
                original.jpg   # 目标
            0802x4d/
                DnCNN.jpg
                original.jpg
            ...
    """
    def __init__(self, results_dir, patch_size=128, transform=None, augment=True):
        self.results_dir = results_dir
        self.patch_size = patch_size
        self.transform = transform
        self.augment = augment
        
        # 收集所有图像对
        self.image_pairs = []
        
        # 遍历所有子目录
        for subdir in os.listdir(results_dir):
            subdir_path = os.path.join(results_dir, subdir)
            
            # 确保是目录
            if not os.path.isdir(subdir_path):
                continue
            
            # 检查是否有DnCNN.jpg和original.jpg
            dncnn_path = os.path.join(subdir_path, 'DnCNN.jpg')
            original_path = os.path.join(subdir_path, 'original.jpg')
            
            if os.path.exists(dncnn_path) and os.path.exists(original_path):
                self.image_pairs.append((dncnn_path, original_path))
        
        print(f"找到 {len(self.image_pairs)} 个图像对")
        
        # 验证数据质量
        if len(self.image_pairs) > 0:
            self._validate_data()
    
    def _validate_data(self):
        """验证数据质量"""
        print("\n验证数据质量...")
        psnr_values = []
        
        for i in range(min(5, len(self.image_pairs))):
            dncnn_path, original_path = self.image_pairs[i]
            
            dncnn_img = cv2.imread(dncnn_path)
            original_img = cv2.imread(original_path)
            
            if dncnn_img is not None and original_img is not None:
                # 确保尺寸相同
                if dncnn_img.shape != original_img.shape:
                    dncnn_img = cv2.resize(dncnn_img, (original_img.shape[1], original_img.shape[0]))
                
                # 计算PSNR
                mse = np.mean((dncnn_img.astype(float) - original_img.astype(float)) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                psnr_values.append(psnr)
                
                print(f"  样本 {i+1}: PSNR = {psnr:.2f} dB")
        
        if psnr_values:
            avg_psnr = np.mean(psnr_values)
            print(f"  平均PSNR: {avg_psnr:.2f} dB")
            if avg_psnr > 35:
                print(f"  ⚠️ PSNR较高，DnCNN结果可能已经很好")
            elif avg_psnr < 25:
                print(f"  ⚠️ PSNR较低，任务较困难")
    
    def __len__(self):
        return len(self.image_pairs) * 20  # 每个图像对生成多个patch
    
    def __getitem__(self, idx):
        # 选择图像对
        pair_idx = idx % len(self.image_pairs)
        dncnn_path, original_path = self.image_pairs[pair_idx]
        
        try:
            # 读取图像
            dncnn_img = cv2.imread(dncnn_path)
            original_img = cv2.imread(original_path)
            
            if dncnn_img is None or original_img is None:
                return self.__getitem__((idx + 1) % len(self))
            
            # 转换为RGB
            dncnn_img = cv2.cvtColor(dncnn_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # 确保尺寸相同
            if dncnn_img.shape != original_img.shape:
                dncnn_img = cv2.resize(dncnn_img, (original_img.shape[1], original_img.shape[0]))
            
            # 随机裁剪
            h, w = dncnn_img.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                # 相同的随机位置
                top = np.random.randint(0, h - self.patch_size)
                left = np.random.randint(0, w - self.patch_size)
                
                dncnn_patch = dncnn_img[top:top+self.patch_size, left:left+self.patch_size]
                original_patch = original_img[top:top+self.patch_size, left:left+self.patch_size]
            else:
                # 调整大小
                dncnn_patch = cv2.resize(dncnn_img, (self.patch_size, self.patch_size))
                original_patch = cv2.resize(original_img, (self.patch_size, self.patch_size))
            
            # 数据增强（只在训练时）
            if self.augment:
                # 随机旋转
                angle = np.random.choice([0, 90, 180, 270])
                if angle == 90:
                    dncnn_patch = cv2.rotate(dncnn_patch, cv2.ROTATE_90_CLOCKWISE)
                    original_patch = cv2.rotate(original_patch, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    dncnn_patch = cv2.rotate(dncnn_patch, cv2.ROTATE_180)
                    original_patch = cv2.rotate(original_patch, cv2.ROTATE_180)
                elif angle == 270:
                    dncnn_patch = cv2.rotate(dncnn_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    original_patch = cv2.rotate(original_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # 随机翻转
                if np.random.random() > 0.5:
                    dncnn_patch = cv2.flip(dncnn_patch, 1)
                    original_patch = cv2.flip(original_patch, 1)
                
                if np.random.random() > 0.5:
                    dncnn_patch = cv2.flip(dncnn_patch, 0)
                    original_patch = cv2.flip(original_patch, 0)
                
                # 轻微的颜色扰动
                if np.random.random() > 0.7:
                    # 亮度调整
                    brightness = np.random.uniform(0.9, 1.1)
                    dncnn_patch = np.clip(dncnn_patch * brightness, 0, 255).astype(np.uint8)
                    original_patch = np.clip(original_patch * brightness, 0, 255).astype(np.uint8)
            
            # 转换为tensor
            if self.transform:
                dncnn_tensor = self.transform(dncnn_patch)
                original_tensor = self.transform(original_patch)
            else:
                # 默认转换：归一化到[0, 1]
                dncnn_tensor = torch.from_numpy(dncnn_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
                original_tensor = torch.from_numpy(original_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
            
            return dncnn_tensor, original_tensor
            
        except Exception as e:
            print(f"处理图像对时出错: {e}")
            # 返回随机生成的图像作为备用
            dummy_image = np.random.randint(0, 255, (self.patch_size, self.patch_size, 3), dtype=np.uint8)
            dummy_tensor = torch.from_numpy(dummy_image.astype(np.float32) / 255.0).permute(2, 0, 1)
            return dummy_tensor, dummy_tensor

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 使用 CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            _ = test_tensor * 2
            device = torch.device('mps')
            print("🚀 使用 MPS")
        except:
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
    total_recon = 0
    total_edge = 0
    
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
        total_recon += recon_loss.item()
        total_edge += edge_loss.item()
        
        # 更新进度条
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'Edge': f'{edge_loss.item():.4f}'
            })
    
    return (total_loss / len(dataloader),
            total_recon / len(dataloader),
            total_edge / len(dataloader))

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_edge = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, edge_maps = model(inputs)
            loss, recon_loss, edge_loss = criterion(outputs, targets, edge_maps)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_edge += edge_loss.item()
    
    return (total_loss / len(dataloader),
            total_recon / len(dataloader),
            total_edge / len(dataloader))

def save_checkpoint(model, optimizer, epoch, loss, save_path, config=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    if next(model.parameters()).device.type == 'mps':
        checkpoint = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                     for k, v in checkpoint.items()}
    
    torch.save(checkpoint, save_path)
    print(f"✅ 保存检查点: {save_path}")

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
            
            # 计算PSNR
            from utils.image_utils import calculate_psnr
            psnr_input = calculate_psnr(target_np * 255, input_np * 255)
            psnr_output = calculate_psnr(target_np * 255, output_np * 255)
            
            # 创建对比图
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            axes[0, 0].imshow(input_np)
            axes[0, 0].set_title(f'DnCNN Input\nPSNR: {psnr_input:.2f} dB')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(target_np)
            axes[0, 1].set_title('Original Target')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(output_np)
            axes[0, 2].set_title(f'Enhanced Output\nPSNR: {psnr_output:.2f} dB')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(edge_np.mean(axis=2), cmap='gray')
            axes[0, 3].set_title('Edge Map')
            axes[0, 3].axis('off')
            
            # 显示差异
            diff_input = np.abs(target_np - input_np)
            diff_output = np.abs(target_np - output_np)
            
            axes[1, 0].imshow(diff_input, cmap='hot')
            axes[1, 0].set_title('Input Difference')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(diff_output, cmap='hot')
            axes[1, 1].set_title('Output Difference')
            axes[1, 1].axis('off')
            
            # 显示局部放大
            h, w = input_np.shape[:2]
            crop_size = min(100, h//4, w//4)
            y, x = h//2, w//2
            
            axes[1, 2].imshow(input_np[y:y+crop_size, x:x+crop_size])
            axes[1, 2].set_title('DnCNN (Zoom)')
            axes[1, 2].axis('off')
            
            axes[1, 3].imshow(output_np[y:y+crop_size, x:x+crop_size])
            axes[1, 3].set_title('Enhanced (Zoom)')
            axes[1, 3].axis('off')
            
            plt.suptitle(f'DnCNN → Original Enhancement (Sample {i+1})', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✅ 可视化结果保存到: {save_dir}")

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 改善的最小变化量
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            # 保存最佳模型状态
            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
            return True  # 表示有改善
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # 表示没有改善
    
    def restore_best_model(self, model):
        """恢复最佳模型权重"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"✅ 恢复第{self.best_epoch}轮的最佳模型权重")

def save_training_history(train_history, val_history, config, save_dir='trained_models'):
    """保存训练历史数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = os.path.join(save_dir, f"training_history_{timestamp}")
    os.makedirs(history_dir, exist_ok=True)
    
    # 保存为JSON格式
    history_data = {
        'train_history': train_history,
        'val_history': val_history,
        'config': config,
        'timestamp': timestamp
    }
    
    json_path = os.path.join(history_dir, 'training_history.json')
    with open(json_path, 'w') as f:
        json.dump(history_data, f, indent=2, default=str)
    
    # 保存为CSV格式
    import csv
    csv_path = os.path.join(history_dir, 'loss_history.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_recon', 'train_edge', 
                        'val_loss', 'val_recon', 'val_edge'])
        
        for i in range(len(train_history['loss'])):
            writer.writerow([
                i+1,
                train_history['loss'][i],
                train_history['recon'][i],
                train_history['edge'][i],
                val_history['loss'][i] if i < len(val_history['loss']) else '',
                val_history['recon'][i] if i < len(val_history['recon']) else '',
                val_history['edge'][i] if i < len(val_history['edge']) else ''
            ])
    
    # 绘制并保存训练曲线
    plt.figure(figsize=(15, 5))
    
    # 总损失
    plt.subplot(1, 3, 1)
    plt.plot(train_history['loss'], label='Train', linewidth=2)
    plt.plot(val_history['loss'], label='Val', linewidth=2)
    plt.title('Total Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 重建损失
    plt.subplot(1, 3, 2)
    plt.plot(train_history['recon'], label='Train', linewidth=2)
    plt.plot(val_history['recon'], label='Val', linewidth=2)
    plt.title('Reconstruction Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 边缘损失
    plt.subplot(1, 3, 3)
    plt.plot(train_history['edge'], label='Train', linewidth=2)
    plt.plot(val_history['edge'], label='Val', linewidth=2)
    plt.title('Edge Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.suptitle('Training Progress', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(history_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 训练历史数据保存到: {history_dir}")
    return history_dir

def main():
    """主训练函数"""
    print("=" * 70)
    print("         训练模型：从DnCNN结果恢复到原始图像")
    print("=" * 70)
    
    # 配置参数
    config = {
        'results_dir': 'batch_results_20260120_142356/images',  # 你的结果目录
        'batch_size': 48,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'patch_size': 128,
        'val_ratio': 0.1,  # 10%作为验证集
        'save_dir': 'trained_models/dncnn_to_original',  # 修改为trained_models
        'visualization_dir': 'trained_models/dncnn_to_original/viz',
        'early_stopping_patience': 15,  # 早停耐心值
        'early_stopping_min_delta': 1e-4,  # 最小改善阈值
        'lr_scheduler_patience': 5,  # 学习率调度器耐心值
        'lr_scheduler_factor': 0.5,  # 学习率衰减因子
        'warmup_epochs': 5,  # 学习率预热轮数
    }
    
    # 检查目录
    if not os.path.exists(config['results_dir']):
        print(f"❌ 结果目录不存在: {config['results_dir']}")
        return
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['visualization_dir'], exist_ok=True)
    
    # 设置设备
    device = setup_device()
    
    # 准备数据集
    print("\n📊 准备数据集...")
    
    # 获取所有图像对路径
    image_pairs = []
    for subdir in os.listdir(config['results_dir']):
        subdir_path = os.path.join(config['results_dir'], subdir)
        if os.path.isdir(subdir_path):
            dncnn_path = os.path.join(subdir_path, 'DnCNN.jpg')
            original_path = os.path.join(subdir_path, 'original.jpg')
            if os.path.exists(dncnn_path) and os.path.exists(original_path):
                image_pairs.append((dncnn_path, original_path))
    
    print(f"总共找到 {len(image_pairs)} 个图像对")
    
    if len(image_pairs) < 10:
        print("❌ 图像对数量不足，至少需要10对")
        return
    
    # 划分训练集和验证集
    np.random.shuffle(image_pairs)
    split_idx = int(len(image_pairs) * (1 - config['val_ratio']))
    train_pairs = image_pairs[:split_idx]
    val_pairs = image_pairs[split_idx:]
    
    print(f"训练集: {len(train_pairs)} 对")
    print(f"验证集: {len(val_pairs)} 对")
    
    # 创建自定义数据集类
    class SplitDataset(Dataset):
        def __init__(self, pairs, patch_size=128, augment=True):
            self.pairs = pairs
            self.patch_size = patch_size
            self.augment = augment
        
        def __len__(self):
            return len(self.pairs) * 20
        
        def __getitem__(self, idx):
            pair_idx = idx % len(self.pairs)
            dncnn_path, original_path = self.pairs[pair_idx]
            
            try:
                dncnn_img = cv2.imread(dncnn_path)
                original_img = cv2.imread(original_path)
                
                if dncnn_img is None or original_img is None:
                    return self.__getitem__((idx + 1) % len(self))
                
                dncnn_img = cv2.cvtColor(dncnn_img, cv2.COLOR_BGR2RGB)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                if dncnn_img.shape != original_img.shape:
                    dncnn_img = cv2.resize(dncnn_img, (original_img.shape[1], original_img.shape[0]))
                
                h, w = dncnn_img.shape[:2]
                if h > self.patch_size and w > self.patch_size:
                    top = np.random.randint(0, h - self.patch_size)
                    left = np.random.randint(0, w - self.patch_size)
                    dncnn_patch = dncnn_img[top:top+self.patch_size, left:left+self.patch_size]
                    original_patch = original_img[top:top+self.patch_size, left:left+self.patch_size]
                else:
                    dncnn_patch = cv2.resize(dncnn_img, (self.patch_size, self.patch_size))
                    original_patch = cv2.resize(original_img, (self.patch_size, self.patch_size))
                
                # 数据增强
                if self.augment:
                    angle = np.random.choice([0, 90, 180, 270])
                    if angle == 90:
                        dncnn_patch = cv2.rotate(dncnn_patch, cv2.ROTATE_90_CLOCKWISE)
                        original_patch = cv2.rotate(original_patch, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        dncnn_patch = cv2.rotate(dncnn_patch, cv2.ROTATE_180)
                        original_patch = cv2.rotate(original_patch, cv2.ROTATE_180)
                    elif angle == 270:
                        dncnn_patch = cv2.rotate(dncnn_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        original_patch = cv2.rotate(original_patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    if np.random.random() > 0.5:
                        dncnn_patch = cv2.flip(dncnn_patch, 1)
                        original_patch = cv2.flip(original_patch, 1)
                    if np.random.random() > 0.5:
                        dncnn_patch = cv2.flip(dncnn_patch, 0)
                        original_patch = cv2.flip(original_patch, 0)
                
                # 转换为tensor
                dncnn_tensor = torch.from_numpy(dncnn_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
                original_tensor = torch.from_numpy(original_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
                
                return dncnn_tensor, original_tensor
                
            except Exception as e:
                dummy = np.random.randint(0, 255, (self.patch_size, self.patch_size, 3), dtype=np.uint8)
                dummy_tensor = torch.from_numpy(dummy.astype(np.float32) / 255.0).permute(2, 0, 1)
                return dummy_tensor, dummy_tensor
    
    # 创建数据集
    train_dataset = SplitDataset(train_pairs, patch_size=config['patch_size'], augment=True)
    val_dataset = SplitDataset(val_pairs, patch_size=config['patch_size'], augment=False)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type != 'mps'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type != 'mps'
    )
    
    # 创建模型
    print("\n🤖 创建模型...")
    model = EdgeEnhancementNetwork().to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")
    
    # 损失函数和优化器
    criterion = EdgeLoss(alpha=1.0, beta=0.5, gamma=0.2)  # 增加边缘损失的权重
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 动态学习率调度器：余弦退火 + 热重启
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 初始周期
        T_mult=2,  # 周期倍增因子
        eta_min=1e-6,  # 最小学习率
        last_epoch=-1
    )
    
    # 添加ReduceLROnPlateau作为备选调度器
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'],
        verbose=True
    )
    
    # 初始化早停机制
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        restore_best_weights=True
    )
    
    # 训练历史
    train_history = {'loss': [], 'recon': [], 'edge': []}
    val_history = {'loss': [], 'recon': [], 'edge': []}
    lr_history = []
    
    # 训练循环
    print("\n🚀 开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # 学习率预热
        if epoch <= config['warmup_epochs']:
            warmup_factor = epoch / config['warmup_epochs']
            current_lr = config['learning_rate'] * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
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
        
        # 记录学习率
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        print(f"\n训练损失: {train_loss:.4f} (重建: {train_recon:.4f}, 边缘: {train_edge:.4f})")
        print(f"验证损失: {val_loss:.4f} (重建: {val_recon:.4f}, 边缘: {val_edge:.4f})")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新学习率调度器
        if epoch > config['warmup_epochs']:
            # 使用验证损失更新ReduceLROnPlateau
            lr_scheduler.step(val_loss)
            
            # 每10个epoch使用余弦退火
            if epoch % 10 == 0:
                scheduler.step()
        
        # 检查早停
        improved = early_stopping(val_loss, model, epoch)
        
        # 保存最佳模型
        if improved:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config['save_dir']}/best_model.pth",
                config
            )
            print(f"✨ 新的最佳模型! 验证损失: {val_loss:.4f}")
        
        # 定期保存
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config['save_dir']}/epoch_{epoch}.pth",
                config
            )
        
        # 定期可视化
        if epoch % 20 == 0 or epoch == 1:
            visualize_results(
                model, val_loader, device,
                save_dir=f"{config['visualization_dir']}/epoch_{epoch}"
            )
        
        # 绘制训练曲线
        if epoch % 5 == 0:
            # 保存训练历史
            history_dir = save_training_history(train_history, val_history, config)
            
            # 绘制学习率变化
            plt.figure(figsize=(10, 4))
            plt.plot(lr_history, 'b-', linewidth=2, marker='o', markersize=4)
            plt.title('Learning Rate Schedule', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(history_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 检查是否早停
        if early_stopping.early_stop:
            print(f"\n⚠️ 早停触发! 连续 {config['early_stopping_patience']} 个epoch验证损失没有改善")
            early_stopping.restore_best_model(model)
            break
    
    # 训练完成
    print("\n" + "="*70)
    print("🎉 训练完成!")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f} (第{early_stopping.best_epoch}轮)")
    print(f"💾 模型保存到: {config['save_dir']}")
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, epoch, val_loss,
        f"{config['save_dir']}/final_model.pth",
        config
    )
    
    # 保存最终训练历史
    history_dir = save_training_history(train_history, val_history, config)
    
    # 生成训练报告
    report = f"""
    ===========================================
               训练报告
    ===========================================
    训练参数:
    - 总轮数: {epoch}
    - 批次大小: {config['batch_size']}
    - 初始学习率: {config['learning_rate']}
    - 训练样本: {len(train_pairs)} 对
    - 验证样本: {len(val_pairs)} 对
    
    训练结果:
    - 最佳验证损失: {best_val_loss:.4f}
    - 最终训练损失: {train_loss:.4f}
    - 最终验证损失: {val_loss:.4f}
    - 训练是否完成: {"是" if epoch == config['num_epochs'] else "早停触发"}
    
    模型信息:
    - 总参数量: {total_params:,}
    - 保存目录: {config['save_dir']}
    - 历史数据: {history_dir}
    ===========================================
    """
    print(report)
    
    # 保存报告
    with open(os.path.join(history_dir, 'training_report.txt'), 'w') as f:
        f.write(report)
    
    print("="*70)

if __name__ == "__main__":
    # 设置随机种子
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