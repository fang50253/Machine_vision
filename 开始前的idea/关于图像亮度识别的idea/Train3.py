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

NUM_Layers=30

class ImprovedDnCNN(nn.Module):
    """改进的DnCNN模型"""
    
    def __init__(self, channels=3, num_layers=NUM_Layers, num_features=64):
        super(ImprovedDnCNN, self).__init__()
        
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
        return x - out

class AdvancedDenoisingDataset(Dataset):
    """改进的训练数据集"""
    
    def __init__(self, image_folder, target_size=(256, 256), max_samples=None):
        self.target_size = target_size
        self.clean_images = []
        self.noisy_images = []
        
        self.image_paths = self._collect_image_paths(image_folder)
        if max_samples and len(self.image_paths) > max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        print(f"找到 {len(self.image_paths)} 张图像，开始预处理...")
        
        if len(self.image_paths) == 0:
            print("警告：没有找到图像文件，创建示例数据...")
            self._create_sample_data()
        else:
            self._preprocess_images()
        
        print(f"预处理完成！生成 {len(self.clean_images)} 个训练样本")
    
    def _collect_image_paths(self, image_folder):
        """改进的文件搜索：支持更多格式和递归搜索"""
        # 支持的图像格式
        extensions = [
            'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif',
            'JPG', 'JPEG', 'PNG', 'BMP', 'TIFF', 'TIF'
        ]
        
        image_paths = []
        
        # 方法1：使用glob递归搜索
        for ext in extensions:
            pattern = os.path.join(image_folder, '**', f'*.{ext}')
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        # 方法2：直接遍历文件夹（如果glob失败）
        if len(image_paths) == 0:
            print("使用直接文件夹遍历...")
            for root, dirs, files in os.walk(image_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in ['jpg', 'jpeg', 'png', 'bmp']):
                        image_paths.append(os.path.join(root, file))
        
        # 去重
        image_paths = list(set(image_paths))
        image_paths.sort()
        
        # 打印找到的文件
        if image_paths:
            print(f"找到的图像文件示例:")
            for i, path in enumerate(image_paths[:5]):  # 显示前5个文件
                print(f"  {i+1}. {os.path.basename(path)}")
            if len(image_paths) > 5:
                print(f"  ... 还有 {len(image_paths) - 5} 个文件")
        
        return image_paths
    
    def _create_sample_data(self):
        """创建示例训练数据"""
        print("创建示例训练数据...")
        num_samples = 1000
        
        for i in range(num_samples):
            # 创建多样化的示例图像
            if i % 4 == 0:
                # 随机纹理
                img = np.random.randint(0, 256, (self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            elif i % 4 == 1:
                # 渐变图像
                img = self._create_gradient_image()
            elif i % 4 == 2:
                # 几何形状
                img = self._create_shape_image()
            else:
                # 混合纹理
                img = self._create_mixed_texture()
            
            # 数据增强
            augmented_images = self._advanced_augmentation(img)
            
            for aug_image in augmented_images:
                # 添加噪声
                for intensity in [15, 25, 35]:
                    for noise_type in ['gaussian', 'salt_pepper']:
                        noisy_image = self._add_advanced_noise(aug_image, noise_type, intensity)
                        self.clean_images.append(aug_image)
                        self.noisy_images.append(noisy_image)
    
    def _create_gradient_image(self):
        """创建渐变图像"""
        img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        for i in range(self.target_size[0]):
            for j in range(self.target_size[1]):
                img[i, j] = [
                    int(128 + 127 * math.sin(i * 0.05)),
                    int(128 + 127 * math.cos(j * 0.05)),
                    int(128 + 127 * math.sin((i + j) * 0.02))
                ]
        return img
    
    def _create_shape_image(self):
        """创建几何形状图像"""
        img = np.random.randint(50, 200, (self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        # 添加一些形状
        center_x, center_y = self.target_size[1] // 2, self.target_size[0] // 2
        cv2.circle(img, (center_x, center_y), 30, (255, 0, 0), -1)
        cv2.rectangle(img, (center_x-40, center_y-20), (center_x+40, center_y+20), (0, 255, 0), -1)
        
        return img
    
    def _create_mixed_texture(self):
        """创建混合纹理图像"""
        img = np.random.randint(0, 256, (self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        # 添加一些纹理模式
        for i in range(0, self.target_size[0], 20):
            cv2.line(img, (0, i), (self.target_size[1], i), (255, 255, 255), 1)
        for j in range(0, self.target_size[1], 20):
            cv2.line(img, (j, 0), (j, self.target_size[0]), (255, 255, 255), 1)
        
        return img
    
    def _preprocess_images(self):
        """预处理真实图像"""
        for image_path in tqdm(self.image_paths, desc="预处理图像"):
            try:
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"无法读取图像: {image_path}")
                    continue
                
                # 转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 调整尺寸
                if image.shape[0] != self.target_size[0] or image.shape[1] != self.target_size[1]:
                    image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
                
                # 数据增强
                augmented_images = self._advanced_augmentation(image)
                
                for aug_image in augmented_images:
                    # 添加多种噪声
                    for intensity in [15, 25, 35]:
                        for noise_type in ['gaussian', 'salt_pepper']:
                            noisy_image = self._add_advanced_noise(aug_image, noise_type, intensity)
                            self.clean_images.append(aug_image)
                            self.noisy_images.append(noisy_image)
                            
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                continue
    
    def _advanced_augmentation(self, image):
        """数据增强"""
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
        
        return augmented
    
    def _add_advanced_noise(self, image, noise_type, intensity):
        """添加噪声"""
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
        
    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        """训练模型"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_base_name = f"dncnn_{timestamp}"
        
        print("开始训练模型...")
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
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_best.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"新的最佳模型已保存: {best_model_path}")
            
            # 每20个epoch保存检查点
            if (epoch + 1) % 20 == 0:
                checkpoint_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_time/60:.2f} 分钟")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 绘制训练曲线
        self._plot_training_curve(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def _plot_training_curve(self, train_losses, val_losses):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        loss_plot_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_loss_curve.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线图已保存: {loss_plot_path}")
        plt.show()

def improved_train_model():
    """改进的训练主函数"""
    print("\n" + "="*60)
    print("图像去噪模型训练程序")
    print("="*60)
    
    # 获取用户输入
    image_folder = input("请输入包含训练图像的文件夹路径: ").strip().strip('"\'')
    
    if not os.path.exists(image_folder):
        print(f"错误：文件夹 '{image_folder}' 不存在！")
        print("请检查路径是否正确")
        return
    
    # 显示文件夹内容
    print(f"\n检查文件夹内容...")
    try:
        items = os.listdir(image_folder)
        print(f"文件夹包含 {len(items)} 个项")
        image_files = [f for f in items if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"其中 {len(image_files)} 个是图像文件")
        
        if image_files:
            print("图像文件示例:")
            for i, file in enumerate(image_files[:5]):
                print(f"  {i+1}. {file}")
    except Exception as e:
        print(f"无法读取文件夹内容: {e}")
    
    # 训练参数
    try:
        epochs = int(input("\n训练轮数 (默认50): ").strip() or "50")
        batch_size = int(input("批量大小 (默认8): ").strip() or "8")
        image_size = int(input("图像尺寸 (默认128): ").strip() or "128")
        max_samples = input("最大图像文件数 (默认全部): ").strip()
        max_samples = int(max_samples) if max_samples else None
    except ValueError:
        print("输入参数错误，使用默认值")
        epochs = 50
        batch_size = 8
        image_size = 128
        max_samples = None
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据集
    print("\n正在准备训练数据...")
    dataset = AdvancedDenoisingDataset(
        image_folder=image_folder,
        target_size=(image_size, image_size),
        max_samples=max_samples
    )
    
    if len(dataset) == 0:
        print("错误：无法创建训练数据！")
        return
    
    # 数据集分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    model = ImprovedDnCNN(channels=3, num_layers=NUM_Layers, num_features=64)
    print(f"使用模型: ImprovedDnCNN (NUM_Layers层)")
    
    # 创建训练器并开始训练
    trainer = ImprovedModelTrainer(model, device, "trained_models")
    
    print("\n开始训练...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=0.001
    )
    
    print("\n训练完成！")
    print(f"模型文件保存在: {trainer.model_save_dir}")
    print(f"最佳模型: {trainer.model_base_name}_best.pth")

if __name__ == "__main__":
    improved_train_model()