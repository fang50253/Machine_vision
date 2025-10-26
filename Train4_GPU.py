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
import sys

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)

NUM_Layers = 17

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
    """改进的训练数据集 - 修复了噪声生成问题"""
    
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
            # 创建多样化的示例图像
            if i % 4 == 0:
                img = np.random.randint(0, 256, (self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            elif i % 4 == 1:
                img = self._create_gradient_image()
            elif i % 4 == 2:
                img = self._create_shape_image()
            else:
                img = self._create_mixed_texture()
            
            # 数据增强
            augmented_images = self._advanced_augmentation(img)
            
            for aug_image in augmented_images:
                # 添加混合噪声
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
                
                # 数据增强
                augmented_images = self._advanced_augmentation(image)
                
                for aug_image in augmented_images:
                    # 添加混合噪声
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
        
        # 基础增强
        augmented.append(cv2.flip(image, 1))
        augmented.append(cv2.flip(image, 0))
        
        # 旋转
        for angle in [90, 180, 270]:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        return augmented
    
    def _add_mixed_noise(self, image, base_intensity):
        """修复的混合噪声生成函数"""
        noisy = image.astype(np.float32)
        
        # 随机选择2-3种噪声进行混合
        num_noise_types = np.random.randint(2, 4)
        noise_types = np.random.choice([
            'gaussian', 'salt_pepper', 'poisson', 'speckle', 'quantization'
        ], num_noise_types, replace=False)
        
        for noise_type in noise_types:
            intensity = base_intensity * np.random.uniform(0.5, 1.5)
            
            if noise_type == 'gaussian':
                # 高斯噪声
                noise = np.random.normal(0, intensity, image.shape)
                noisy += noise
            
            elif noise_type == 'salt_pepper':
                # 椒盐噪声
                amount = intensity / 400.0
                salt_mask = np.random.random(image.shape[:2]) < amount
                pepper_mask = np.random.random(image.shape[:2]) < amount
                # 为每个通道应用相同的mask
                for c in range(3):
                    noisy_channel = noisy[:, :, c]
                    noisy_channel[salt_mask] = 255
                    noisy_channel[pepper_mask] = 0
                    noisy[:, :, c] = noisy_channel
            
            elif noise_type == 'poisson':
                # 修复的泊松噪声 - 确保lambda参数有效
                # 将图像归一化到[0, 1]范围，然后乘以一个缩放因子
                scale_factor = max(1, intensity / 10.0)  # 确保是正数
                normalized = np.clip(noisy / 255.0, 0, 1)
                poisson_noise = np.random.poisson(normalized * scale_factor * 255)
                noisy = poisson_noise / scale_factor
            
            elif noise_type == 'speckle':
                # 散斑噪声 (乘性噪声)
                speckle = np.random.randn(*image.shape) * intensity * 0.01
                noisy += noisy * speckle
            
            elif noise_type == 'quantization':
                # 量化噪声 (模拟压缩伪影)
                quant_level = max(2, 256 - int(intensity * 2))
                noisy = (noisy // (256 // quant_level)) * (256 // quant_level)
        
        # 确保像素值在有效范围内
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
        
        # 改进的GPU设置
        self.model = self._setup_model_device(model, device)
        
    def _setup_model_device(self, model, device):
        """设置模型设备，支持多GPU"""
        model = model.to(device)
        
        # 如果有多GPU，使用数据并行
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            model = nn.DataParallel(model)
        
        return model
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
        """训练模型 - 包含早停机制"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        # 初始化早停
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
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}')
            
            # 早停检查
            early_stopping(avg_val_loss, self.model)
            
            if early_stopping.early_stop:
                print("早停触发，恢复最佳模型...")
                self.model.load_state_dict(early_stopping.best_model_state)
                break
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_best.pth')
                
                # 如果使用多GPU，保存原始模型状态
                if isinstance(self.model, nn.DataParallel):
                    torch.save(self.model.module.state_dict(), best_model_path)
                else:
                    torch.save(self.model.state_dict(), best_model_path)
                    
                print(f"新的最佳模型已保存: {best_model_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(self.model_save_dir, f'{self.model_base_name}_final.pth')
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), final_model_path)
        else:
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

def setup_device():
    """设置训练设备，优化GPU使用"""
    # 强制检查CUDA可用性
    print("正在检测GPU设备...")
    
    # 方法1: 直接检查CUDA
    if torch.cuda.is_available():
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        device = torch.device('cuda')
        
        # 显示详细的GPU信息
        gpu_count = torch.cuda.device_count()
        print(f"发现 {gpu_count} 个GPU设备:")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"    显存: {gpu_memory:.1f} GB")
        
        # 设置当前GPU
        torch.cuda.set_device(0)
        
        # 设置CUDA优化标志
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # 验证GPU是否真的可用
        test_tensor = torch.tensor([1.0]).cuda()
        if test_tensor.is_cuda:
            print("✓ GPU测试通过，正在使用GPU进行训练")
        else:
            print("✗ GPU测试失败，回退到CPU")
            device = torch.device('cpu')
        
    else:
        device = torch.device('cpu')
        print("✗ 未检测到可用的CUDA设备")
        print("可能的原因:")
        print("  1. 未安装NVIDIA显卡驱动")
        print("  2. 未安装CUDA工具包")
        print("  3. PyTorch版本不支持当前CUDA")
        print("  4. 显卡太老不支持CUDA")
    
    return device

def check_pytorch_cuda_support():
    """检查PyTorch的CUDA支持"""
    print("\n" + "="*50)
    print("PyTorch CUDA支持诊断")
    print("="*50)
    
    # 基本CUDA检查
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        # 显示CUDA版本信息
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
        
        # 测试GPU计算
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print("✓ GPU计算测试通过")
        except Exception as e:
            print(f"✗ GPU计算测试失败: {e}")
    else:
        print("✗ CUDA不可用")
        
        # 检查PyTorch构建信息
        print(f"PyTorch版本: {torch.__version__}")
        print(f"PyTorch构建配置:")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        print(f"  MPS可用: {getattr(torch, 'has_mps', False)}")  # Apple Silicon
    
    print("="*50)

def improved_train_model():
    """改进的训练主函数"""
    print("\n" + "="*60)
    print("修复版图像去噪模型训练程序")
    print("已修复GPU检测和泊松噪声生成错误")
    print("="*60)
    
    # 首先运行CUDA诊断
    check_pytorch_cuda_support()
    
    # 获取用户输入
    image_folder = input("\n请输入包含训练图像的文件夹路径: ").strip().strip('"\'')
    
    if not os.path.exists(image_folder):
        print(f"错误：文件夹 '{image_folder}' 不存在！")
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
        patience = int(input("早停耐心值 (默认10): ").strip() or "10")
        max_samples = input("最大图像文件数 (默认全部): ").strip()
        max_samples = int(max_samples) if max_samples else None
    except ValueError:
        print("输入参数错误，使用默认值")
        epochs = 50
        batch_size = 8
        image_size = 128
        patience = 10
        max_samples = None
    
    # 设备设置
    device = setup_device()
    
    # 创建数据集
    print("\n正在准备训练数据...")
    print("噪声类型: 高斯噪声 + 椒盐噪声 + 泊松噪声 + 散斑噪声 + 量化噪声")
    print("训练策略: 混合噪声训练")
    
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
    
    # 根据GPU内存调整num_workers
    num_workers = 4 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    model = ImprovedDnCNN(channels=3, num_layers=NUM_Layers, num_features=64)
    print(f"使用模型: ImprovedDnCNN ({NUM_Layers}层)")
    
    # 创建训练器并开始训练
    trainer = ImprovedModelTrainer(model, device, "trained_models")
    
    print("\n开始训练...")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=0.001,
        patience=patience
    )
    
    print("\n训练完成！")
    print(f"模型文件保存在: {trainer.model_save_dir}")
    print(f"最佳模型: {trainer.model_base_name}_best.pth")

# 如果直接运行此脚本，先检查环境
if __name__ == "__main__":
    # 额外的环境检查
    print("系统环境信息:")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"OpenCV版本: {cv2.__version__}")
    
    improved_train_model()