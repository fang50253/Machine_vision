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

# 设置随机种子以保证结果可重现
torch.manual_seed(42)
np.random.seed(42)

class RealImageDenoisingDataset(Dataset):
    """真实图像去噪数据集"""
    
    def __init__(self, image_folder, target_size=(256, 256), noise_intensity=25, max_samples=None):
        """
        参数:
            image_folder: 包含训练图像的文件夹路径
            target_size: 目标图像尺寸 (高度, 宽度)
            noise_intensity: 噪声强度
            max_samples: 最大样本数量
        """
        self.target_size = target_size
        self.noise_intensity = noise_intensity
        self.clean_images = []
        self.noisy_images = []
        
        # 收集图像文件
        self.image_paths = self._collect_image_paths(image_folder)
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        print(f"找到 {len(self.image_paths)} 张图像，开始预处理...")
        self._preprocess_images()
        
    def _collect_image_paths(self, image_folder):
        """收集文件夹中的所有图像文件路径"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        
        for extension in image_extensions:
            pattern = os.path.join(image_folder, '**', extension)  # 递归搜索
            image_paths.extend(glob.glob(pattern, recursive=True))
            image_paths.extend(glob.glob(pattern.upper(), recursive=True))
        
        # 去重并排序
        image_paths = list(set(image_paths))
        image_paths.sort()
        
        return image_paths
    
    def _preprocess_images(self):
        """预处理所有图像"""
        for image_path in tqdm(self.image_paths, desc="预处理图像"):
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # 转换为RGB（OpenCV读取的是BGR）
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 调整尺寸
                if image.shape[0] != self.target_size[0] or image.shape[1] != self.target_size[1]:
                    image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
                
                # 数据增强：随机裁剪、翻转等
                augmented_images = self._data_augmentation(image)
                
                for aug_image in augmented_images:
                    # 添加多种噪声类型
                    noise_types = ['gaussian', 'salt_pepper']
                    for noise_type in noise_types:
                        noisy_image = self._add_noise(aug_image, noise_type)
                        
                        self.clean_images.append(aug_image)
                        self.noisy_images.append(noisy_image)
                        
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                continue
        
        print(f"预处理完成！生成 {len(self.clean_images)} 个训练样本")
    
    def _data_augmentation(self, image):
        """数据增强"""
        augmented = [image]  # 原始图像
        
        # 水平翻转
        augmented.append(cv2.flip(image, 1))
        
        # 垂直翻转
        augmented.append(cv2.flip(image, 0))
        
        # 随机旋转90度
        k = np.random.randint(0, 4)
        augmented.append(np.rot90(image, k).copy())
        
        return augmented
    
    def _add_noise(self, image, noise_type='gaussian'):
        """添加噪声"""
        noisy_image = image.copy()
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_intensity, image.shape).astype(np.float32)
            noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        elif noise_type == 'salt_pepper':
            amount = self.noise_intensity / 500.0  # 调整比例
            
            # 盐噪声
            salt_mask = np.random.random(image.shape[:2]) < amount
            noisy_image[salt_mask] = 255
            
            # 椒噪声
            pepper_mask = np.random.random(image.shape[:2]) < amount
            noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean = self.clean_images[idx]
        noisy = self.noisy_images[idx]
        
        # 转换为Tensor并归一化
        clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1)).float() / 255.0
        noisy_tensor = torch.from_numpy(noisy.transpose(2, 0, 1)).float() / 255.0
        
        return noisy_tensor, clean_tensor

class DnCNN(nn.Module):
    """DnCNN去噪模型"""
    
    def __init__(self, channels=3, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        
        layers = []
        # 第一层
        layers.append(nn.Conv2d(channels, num_features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # 学习残差

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device, model_save_dir="trained_models"):
        self.model = model
        self.device = device
        self.model_save_dir = model_save_dir
        
        # 创建保存目录
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        self.model.to(device)
        
    def train(self, train_loader, val_loader=None, epochs=100, lr=0.001):
        """训练模型"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # 修复：移除verbose参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        # 生成模型名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_base_name = f"dncnn_{timestamp}"
        
        print("开始训练模型...")
        print(f"模型将保存为: {self.model_base_name}_*.pth")
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
                loss = criterion(outputs, clean)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches = batch_idx + 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy, clean = noisy.to(self.device), clean.to(self.device)
                        outputs = self.model(noisy)
                        loss = criterion(outputs, clean)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                # 更新学习率
                scheduler.step(avg_val_loss)
                
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_best.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"新的最佳模型已保存: {best_model_path}")
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}')
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss if val_loader else None
                }, checkpoint_path)
                print(f"检查点已保存: {checkpoint_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_time/60:.2f} 分钟")
        print(f"最终模型已保存: {final_model_path}")
        if val_loader:
            print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 绘制损失曲线
        self._plot_training_curve(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def _plot_training_curve(self, train_losses, val_losses):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
        
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存损失曲线图
        loss_plot_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_loss_curve.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线图已保存: {loss_plot_path}")
        plt.show()

def train_model_from_folder():
    """从文件夹训练模型的主函数"""
    print("\n" + "="*60)
    print("真实图像去噪模型训练程序")
    print("="*60)
    
    # 获取用户输入
    image_folder = input("请输入包含训练图像的文件夹路径: ").strip()
    image_folder = image_folder.strip('"\'')
    
    if not os.path.exists(image_folder):
        print(f"错误：文件夹 '{image_folder}' 不存在！")
        return
    
    # 训练参数设置
    try:
        epochs = int(input("训练轮数 (默认100): ").strip() or "100")
        batch_size = int(input("批量大小 (默认8): ").strip() or "8")
        image_size = int(input("图像尺寸 (默认256): ").strip() or "256")
        max_samples = input("最大样本数 (默认全部，直接回车): ").strip()
        max_samples = int(max_samples) if max_samples else None
    except ValueError:
        print("输入参数错误，使用默认值")
        epochs = 100
        batch_size = 8
        image_size = 256
        max_samples = None
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 创建数据集
    print("\n正在准备训练数据...")
    dataset = RealImageDenoisingDataset(
        image_folder=image_folder,
        target_size=(image_size, image_size),
        noise_intensity=25,
        max_samples=max_samples
    )
    
    if len(dataset) == 0:
        print("错误：没有找到可用的训练图像！")
        return
    
    # 分割训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 修复：num_workers设为0避免多进程问题
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    model = DnCNN(channels=3, num_layers=17, num_features=64)
    
    # 创建训练器
    trainer = ModelTrainer(model, device)
    
    # 开始训练
    print("\n开始训练...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=0.001
    )
    
    print("\n训练完成！")
    print(f"所有模型文件保存在: {trainer.model_save_dir}")
    print(f"模型名称前缀: {trainer.model_base_name}")

if __name__ == "__main__":
    train_model_from_folder()