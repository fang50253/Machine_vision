import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
import glob
from datetime import datetime
from tqdm import tqdm
import math

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class ImprovedDnCNN(nn.Module):
    """改进的DnCNN模型"""
    
    def __init__(self, channels=3, num_layers=20, num_features=128):
        super(ImprovedDnCNN, self).__init__()
        
        layers = []
        # 第一层 - 增加特征数
        layers.append(nn.Conv2d(channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层 - 更深的网络
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dncnn(x)
        return x - out

class ResidualDnCNN(nn.Module):
    """带残差连接的改进DnCNN"""
    
    def __init__(self, channels=3, num_blocks=8, num_features=128):
        super(ResidualDnCNN, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features)
            ))
        
        self.final = nn.Conv2d(num_features, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        initial = self.initial(x)
        
        out = initial
        for block in self.blocks:
            residual = out
            out = block(out)
            out += residual  # 残差连接
            out = torch.relu(out)
        
        out = self.final(out)
        return x - out

class AdvancedDenoisingDataset(Dataset):
    """改进的训练数据集"""
    
    def __init__(self, image_folder, target_size=(256, 256), max_samples=None):
        self.target_size = target_size
        self.clean_images = []
        self.noisy_images = []
        
        self.image_paths = self._collect_image_paths(image_folder)
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        print(f"找到 {len(self.image_paths)} 张图像，开始预处理...")
        self._preprocess_images()
        
    def _collect_image_paths(self, image_folder):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        
        for extension in image_extensions:
            pattern = os.path.join(image_folder, '**', extension)
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        return list(set(image_paths))
    
    def _preprocess_images(self):
        """改进的预处理：更多数据增强和噪声类型"""
        for image_path in tqdm(self.image_paths, desc="预处理图像"):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 调整尺寸
                if image.shape[0] != self.target_size[0] or image.shape[1] != self.target_size[1]:
                    image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
                
                # 更多数据增强
                augmented_images = self._advanced_augmentation(image)
                
                for aug_image in augmented_images:
                    # 多种噪声强度和类型
                    for intensity in [15, 25, 35, 50]:
                        for noise_type in ['gaussian', 'salt_pepper', 'mixed']:
                            noisy_image = self._add_advanced_noise(aug_image, noise_type, intensity)
                            
                            self.clean_images.append(aug_image)
                            self.noisy_images.append(noisy_image)
                            
            except Exception as e:
                continue
        
        print(f"预处理完成！生成 {len(self.clean_images)} 个训练样本")
    
    def _advanced_augmentation(self, image):
        """高级数据增强"""
        augmented = [image]
        
        # 基础增强
        augmented.append(cv2.flip(image, 1))  # 水平翻转
        augmented.append(cv2.flip(image, 0))  # 垂直翻转
        
        # 旋转
        for angle in [90, 180, 270]:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        # 亮度调整
        for alpha in [0.7, 0.8, 1.2, 1.3]:
            brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented.append(brightened)
        
        # 添加高斯模糊模拟真实噪声
        for sigma in [0.5, 1.0]:
            blurred = cv2.GaussianBlur(image, (5, 5), sigma)
            augmented.append(blurred)
        
        return augmented
    
    def _add_advanced_noise(self, image, noise_type, intensity):
        """高级噪声添加"""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        
        elif noise_type == 'salt_pepper':
            noisy = image.copy()
            amount = intensity / 400.0
            # 盐噪声
            salt_mask = np.random.random(image.shape[:2]) < amount
            noisy[salt_mask] = 255
            # 椒噪声
            pepper_mask = np.random.random(image.shape[:2]) < amount
            noisy[pepper_mask] = 0
        
        elif noise_type == 'mixed':
            # 混合噪声
            noisy = image.copy().astype(np.float32)
            # 高斯噪声
            gaussian_noise = np.random.normal(0, intensity, image.shape)
            noisy += gaussian_noise
            # 椒盐噪声
            amount = intensity / 800.0
            salt_mask = np.random.random(image.shape[:2]) < amount
            pepper_mask = np.random.random(image.shape[:2]) < amount
            noisy[salt_mask] = 255
            noisy[pepper_mask] = 0
            
            noisy = np.clip(noisy, 0, 255)
        
        return noisy.astype(np.uint8)
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean = self.clean_images[idx]
        noisy = self.noisy_images[idx]
        
        clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1)).float() / 255.0
        noisy_tensor = torch.from_numpy(noisy.transpose(2, 0, 1)).float() / 255.0
        
        return noisy_tensor, clean_tensor

class ImprovedModelTrainer:
    """改进的模型训练器"""
    
    def __init__(self, model, device, model_save_dir="improved_models"):
        self.model = model
        self.device = device
        self.model_save_dir = model_save_dir
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        self.model.to(device)
        
    def train(self, train_loader, val_loader, epochs=200, lr=0.001):
        """改进的训练策略"""
        # 使用AdamW优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 使用L1损失 + MSE损失的组合
        criterion1 = nn.L1Loss()
        criterion2 = nn.MSELoss()
        
        # 余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_base_name = f"improved_dncnn_{timestamp}"
        
        print("开始改进训练...")
        print(f"使用模型: {self.model.__class__.__name__}")
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (noisy, clean) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(noisy)
                
                # 组合损失函数
                loss1 = criterion1(outputs, clean)  # L1损失保持边缘
                loss2 = criterion2(outputs, clean)  # MSE损失平滑区域
                loss = 0.7 * loss1 + 0.3 * loss2  # 加权组合
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches = batch_idx + 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(self.device), clean.to(self.device)
                    outputs = self.model(noisy)
                    loss = criterion1(outputs, clean)  # 验证时只用L1损失
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # 更新学习率
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}')
            
            # 早停机制
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_best.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"新的最佳模型已保存: {best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 如果连续20个epoch没有改进，提前停止
            if patience_counter >= 20:
                print(f"早停触发，在epoch {epoch+1}停止训练")
                break
            
            # 每25个epoch保存检查点
            if (epoch + 1) % 25 == 0:
                checkpoint_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_time/60:.2f} 分钟")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 绘制详细的训练曲线
        self._plot_detailed_curve(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def _plot_detailed_curve(self, train_losses, val_losses):
        """绘制详细训练曲线"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
        plt.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # 绘制最后50个epoch的细节
        start_idx = max(0, len(train_losses) - 50)
        plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:], 
                label='Training Loss (last 50)', linewidth=2, color='blue')
        plt.plot(range(start_idx, len(val_losses)), val_losses[start_idx:], 
                label='Validation Loss (last 50)', linewidth=2, color='red')
        plt.title('Loss Curve (Last 50 Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        loss_plot_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_detailed_curve.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"详细训练曲线已保存: {loss_plot_path}")
        plt.show()

def improved_train_model():
    """改进的训练主函数"""
    print("\n" + "="*70)
    print("高级图像去噪模型训练程序")
    print("="*70)
    
    # 获取用户输入
    image_folder = input("请输入包含训练图像的文件夹路径: ").strip().strip('"\'')
    
    if not os.path.exists(image_folder):
        print(f"错误：文件夹 '{image_folder}' 不存在！")
        return
    
    # 模型选择
    print("\n请选择模型架构:")
    print("1. 改进DnCNN (更深网络)")
    print("2. 残差DnCNN (推荐)")
    
    model_choice = input("请选择 (1/2, 默认2): ").strip() or "2"
    
    # 训练参数
    try:
        epochs = int(input("训练轮数 (默认200): ").strip() or "200")
        batch_size = int(input("批量大小 (默认16): ").strip() or "16")
        image_size = int(input("图像尺寸 (默认256): ").strip() or "256")
        max_samples = input("最大样本数 (默认全部): ").strip()
        max_samples = int(max_samples) if max_samples else None
    except ValueError:
        print("输入参数错误，使用默认值")
        epochs = 200
        batch_size = 16
        image_size = 256
        max_samples = None
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据集
    print("\n正在准备高级训练数据...")
    dataset = AdvancedDenoisingDataset(
        image_folder=image_folder,
        target_size=(image_size, image_size),
        max_samples=max_samples
    )
    
    if len(dataset) == 0:
        print("错误：没有找到可用的训练图像！")
        return
    
    # 数据集分割
    train_size = int(0.85 * len(dataset))  # 增加训练集比例
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    if model_choice == "1":
        model = ImprovedDnCNN(channels=3, num_layers=20, num_features=128)
        print("使用: ImprovedDnCNN (20层, 128特征)")
    else:
        model = ResidualDnCNN(channels=3, num_blocks=8, num_features=128)
        print("使用: ResidualDnCNN (8残差块, 128特征)")
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 创建训练器并开始训练
    trainer = ImprovedModelTrainer(model, device, "improved_models")
    
    print("\n开始高级训练...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=0.001
    )
    
    print("\n训练完成！")
    print(f"模型文件保存在: {trainer.model_save_dir}")
    print(f"使用命令测试模型: python denoise_program.py --model {trainer.model_base_name}_best.pth")

if __name__ == "__main__":
    improved_train_model()