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
import cv2
import sys
import numpy as np
import platform
from config import NUM_LAYERS

class DeviceManager:
    """设备管理器，自动选择最佳设备"""
    
    @staticmethod
    def get_device():
        """获取最佳可用设备"""
        system = platform.system()
        if system == "Darwin":  # macOS
            if torch.backends.mps.is_available():
                print("🚀 使用 Apple Silicon GPU (MPS) 进行加速")
                return torch.device("mps")
            else:
                print("⚠️  MPS 不可用，使用 CPU")
                return torch.device("cpu")
                
        elif system == "Windows":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 使用 NVIDIA GPU: {gpu_name}")
                return torch.device("cuda")
            else:
                print("⚠️  CUDA 不可用，使用 CPU")
                return torch.device("cpu")
                
        else:  # Linux 或其他系统
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 使用 NVIDIA GPU: {gpu_name}")
                return torch.device("cuda")
            else:
                print("⚠️  无 GPU 可用，使用 CPU")
                return torch.device("cpu")

class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('触发早停机制！')
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

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
        """收集图像文件路径"""
        extensions = [
            'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif',
            'JPG', 'JPEG', 'PNG', 'BMP', 'TIFF', 'TIF'
        ]
        
        image_paths = []
        
        for ext in extensions:
            pattern = os.path.join(image_folder, '**', f'*.{ext}')
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        if len(image_paths) == 0:
            print("使用直接文件夹遍历...")
            for root, dirs, files in os.walk(image_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in ['jpg', 'jpeg', 'png', 'bmp']):
                        image_paths.append(os.path.join(root, file))
        
        image_paths = list(set(image_paths))
        image_paths.sort()
        
        if image_paths:
            print(f"找到的图像文件示例:")
            for i, path in enumerate(image_paths[:5]):
                print(f"  {i+1}. {os.path.basename(path)}")
            if len(image_paths) > 5:
                print(f"  ... 还有 {len(image_paths) - 5} 个文件")
        
        return image_paths
    
    def _create_sample_data(self):
        """创建示例训练数据"""
        print("创建示例训练数据...")
        num_samples = 1000
        
        for i in range(num_samples):
            if i % 4 == 0:
                img = np.random.randint(0, 256, (self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            elif i % 4 == 1:
                img = self._create_gradient_image()
            elif i % 4 == 2:
                img = self._create_shape_image()
            else:
                img = self._create_mixed_texture()
            
            augmented_images = self._advanced_augmentation(img)
            
            for aug_image in augmented_images:
                for intensity in [15, 25, 35, 50]:
                    noisy_image = self._add_mixed_noise(aug_image, intensity)
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
        
        center_x, center_y = self.target_size[1] // 2, self.target_size[0] // 2
        cv2.circle(img, (center_x, center_y), 30, (255, 0, 0), -1)
        cv2.rectangle(img, (center_x-40, center_y-20), (center_x+40, center_y+20), (0, 255, 0), -1)
        
        return img
    
    def _create_mixed_texture(self):
        """创建混合纹理图像"""
        img = np.random.randint(0, 256, (self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        for i in range(0, self.target_size[0], 20):
            cv2.line(img, (0, i), (self.target_size[1], i), (255, 255, 255), 1)
        for j in range(0, self.target_size[1], 20):
            cv2.line(img, (j, 0), (j, self.target_size[0]), (255, 255, 255), 1)
        
        return img
    
    def _preprocess_images(self):
        """预处理真实图像"""
        for image_path in tqdm(self.image_paths, desc="预处理图像"):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"无法读取图像: {image_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if image.shape[0] != self.target_size[0] or image.shape[1] != self.target_size[1]:
                    image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
                
                augmented_images = self._advanced_augmentation(image)
                
                for aug_image in augmented_images:
                    for intensity in [15, 25, 35, 50]:
                        noisy_image = self._add_mixed_noise(aug_image, intensity)
                        self.clean_images.append(aug_image)
                        self.noisy_images.append(noisy_image)
                            
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                continue
    
    def _advanced_augmentation(self, image):
        """数据增强"""
        augmented = [image]
        
        augmented.append(cv2.flip(image, 1))
        augmented.append(cv2.flip(image, 0))
        
        for angle in [90, 180, 270]:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        return augmented
    
    def _add_mixed_noise(self, image, base_intensity):
        """混合噪声生成函数"""
        noisy = image.astype(np.float32)
        
        num_noise_types = np.random.randint(2, 4)
        noise_types = np.random.choice([
            'gaussian', 'salt_pepper', 'poisson', 'speckle', 'quantization'
        ], num_noise_types, replace=False)
        
        for noise_type in noise_types:
            intensity = base_intensity * np.random.uniform(0.5, 1.5)
            
            if noise_type == 'gaussian':
                noise = np.random.normal(0, intensity, image.shape)
                noisy += noise
            
            elif noise_type == 'salt_pepper':
                amount = intensity / 400.0
                salt_mask = np.random.random(image.shape[:2]) < amount
                pepper_mask = np.random.random(image.shape[:2]) < amount
                for c in range(3):
                    noisy_channel = noisy[:, :, c]
                    noisy_channel[salt_mask] = 255
                    noisy_channel[pepper_mask] = 0
                    noisy[:, :, c] = noisy_channel
            
            elif noise_type == 'poisson':
                scale_factor = max(1, intensity / 10.0)
                normalized = np.clip(noisy / 255.0, 0, 1)
                poisson_noise = np.random.poisson(normalized * scale_factor * 255)
                noisy = poisson_noise / scale_factor
            
            elif noise_type == 'speckle':
                speckle = np.random.randn(*image.shape) * intensity * 0.01
                noisy += noisy * speckle
            
            elif noise_type == 'quantization':
                quant_level = max(2, 256 - int(intensity * 2))
                noisy = (noisy // (256 // quant_level)) * (256 // quant_level)
        
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

class ModelTrainer:
    """模型训练器 - 跨平台兼容"""
    
    def __init__(self, model, model_save_dir="trained_models"):
        self.device = DeviceManager.get_device()
        self.model = model
        self.model_save_dir = model_save_dir
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        self.model = self._setup_model_device(model)
        
    def _setup_model_device(self, model):
        """设置模型设备 - 跨平台兼容"""
        model = model.to(self.device)
        
        # 只在 CUDA 上使用多 GPU，MPS 不支持 DataParallel
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            model = nn.DataParallel(model)
        elif self.device.type == "mps":
            print("使用 Apple Silicon GPU (MPS) 进行训练")
        elif self.device.type == "cuda":
            print(f"使用单个 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用 CPU 进行训练")
        
        return model
    
    def _memory_cleanup(self):
        """内存清理 - 跨平台兼容"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            # MPS 目前没有显式的内存清理函数，但可以尝试其他方法
            import gc
            gc.collect()
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
        """训练模型 - 跨平台兼容"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
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
            
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}')
            
            # 内存清理
            self._memory_cleanup()
            
            early_stopping(avg_val_loss, self.model)
            
            if early_stopping.early_stop:
                print("早停触发，恢复最佳模型...")
                self.model.load_state_dict(early_stopping.best_model_state)
                break
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_best.pth')
                
                # 保存模型状态 - 兼容多GPU训练
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save(model_to_save.state_dict(), best_model_path)
                    
                print(f"新的最佳模型已保存: {best_model_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_final.pth')
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), final_model_path)
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {training_time/60:.2f} 分钟")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        return train_losses, val_losses, best_val_loss

# 使用示例
if __name__ == "__main__":
    # 示例模型定义
    class DnCNN(nn.Module,max_layers=NUM_LAYERS):
        def __init__(self, channels=3, num_layers=NUM_LAYERS):
            super(DnCNN, self).__init__()
            layers = []
            layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            
            for _ in range(num_layers-2):
                layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(64))
                layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
            self.dncnn = nn.Sequential(*layers)
        
        def forward(self, x):
            out = self.dncnn(x)
            return x - out  # 残差学习
    
    # 创建模型和训练器
    model = DnCNN()
    trainer = ModelTrainer(model)
    
    print(f"当前系统: {platform.system()}")
    print(f"使用设备: {trainer.device}")
    
    # 这里可以添加数据加载和训练代码
    # dataset = AdvancedDenoisingDataset("your_image_folder")
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # trainer.train(train_loader, train_loader)  # 示例用法