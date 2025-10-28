import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from tqdm import tqdm

# 设置随机种子以保证结果可重现
np.random.seed(42)
torch.manual_seed(42)

MAX_Pixel=256

class DenoisingDataset(Dataset):
    """去噪数据集"""
    def __init__(self, clean_images, noisy_images):
        self.clean_images = clean_images
        self.noisy_images = noisy_images
        
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean = self.clean_images[idx]
        noisy = self.noisy_images[idx]
        
        # 转换为Tensor并归一化
        clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1)).float() / 255.0
        noisy_tensor = torch.from_numpy(noisy.transpose(2, 0, 1)).float() / 255.0
        
        return noisy_tensor, clean_tensor

class TraditionalDenoiser:
    """传统图像去噪方法"""
    
    @staticmethod
    def wavelet_denoise(image, wavelet='db4', level=2, threshold=0.1):
        """小波变换去噪"""
        try:
            import pywt
            # 将图像转换为浮点数
            image_float = image.astype(np.float32) / 255.0
            
            # 对每个通道进行小波变换去噪
            if len(image.shape) == 3:
                denoised = np.zeros_like(image_float)
                for i in range(3):
                    coeffs = pywt.wavedec2(image_float[:,:,i], wavelet, level=level)
                    # 修复：正确的阈值处理方式
                    coeffs_thresh = []
                    coeffs_thresh.append(coeffs[0])  # 近似系数
                    for level_coeff in coeffs[1:]:
                        # 对每个细节系数进行阈值处理
                        thresh_coeff = []
                        for detail in level_coeff:
                            thresh_detail = pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft')
                            thresh_coeff.append(thresh_detail)
                        coeffs_thresh.append(tuple(thresh_coeff))
                    
                    denoised[:,:,i] = pywt.waverec2(coeffs_thresh, wavelet)
            else:
                coeffs = pywt.wavedec2(image_float, wavelet, level=level)
                coeffs_thresh = []
                coeffs_thresh.append(coeffs[0])  # 近似系数
                for level_coeff in coeffs[1:]:
                    thresh_coeff = []
                    for detail in level_coeff:
                        thresh_detail = pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft')
                        thresh_coeff.append(thresh_detail)
                    coeffs_thresh.append(tuple(thresh_coeff))
                
                denoised = pywt.waverec2(coeffs_thresh, wavelet)
            
            # 裁剪到有效范围并转换回uint8
            denoised = np.clip(denoised, 0, 1)
            return (denoised * 255).astype(np.uint8)
            
        except ImportError:
            print("PyWavelets未安装，使用中值滤波代替")
            return cv2.medianBlur(image, 5)
        except Exception as e:
            print(f"小波去噪出错: {e}，使用中值滤波代替")
            return cv2.medianBlur(image, 5)
    
    @staticmethod
    def median_denoise(image, kernel_size=5):
        """中值滤波去噪"""
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_denoise(image, d=9, sigma_color=75, sigma_space=75):
        """双边滤波去噪"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def gaussian_denoise(image, kernel_size=5, sigma=1.0):
        """高斯滤波去噪"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

class DnCNN(nn.Module):
    """简单的DnCNN深度学习去噪模型"""
    
    def __init__(self, channels=3, num_layers=7, num_features=64):
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
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train(self, train_loader, val_loader=None, epochs=50, lr=0.001, save_path='trained_models'):
        """训练模型"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
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
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}')
            
            # 每10个epoch保存一次模型
            if (epoch + 1) % 10 == 0:
                model_path = os.path.join(save_path, f'dncnn_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"模型已保存: {model_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(save_path, 'dncnn_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        training_time = time.time() - start_time
        print(f"训练完成！总耗时: {training_time:.2f}秒")
        print(f"最终模型已保存: {final_model_path}")
        
        return train_losses, val_losses

class HybridDenoiser:
    """混合去噪器：结合传统方法和深度学习"""
    
    def __init__(self, model_path=None):
        self.traditional_denoiser = TraditionalDenoiser()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化时就加载模型
        self.load_pretrained_model(model_path)
    
    def load_pretrained_model(self, model_path=None):
        """加载预训练模型"""
        self.model = DnCNN(channels=3)
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"预训练模型加载成功: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"加载模型失败: {e}，使用随机初始化的模型")
        else:
            print("未提供模型路径或文件不存在，使用随机初始化的模型")
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为Tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image_tensor.unsqueeze(0).to(self.device)
    
    def postprocess_image(self, tensor):
        """图像后处理"""
        image = tensor.squeeze(0).cpu().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image
    
    def deep_learning_denoise(self, image):
        """深度学习去噪"""
        if self.model is None:
            print("错误：模型未初始化")
            return image
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output_tensor = self.model(input_tensor)
            denoised_image = self.postprocess_image(output_tensor)
        
        return denoised_image
    
    def hybrid_denoise_v1(self, image):
        """混合方法1：传统方法预处理 + 深度学习精处理"""
        # 第一步：使用传统方法进行初步去噪
        traditional_denoised = self.traditional_denoiser.bilateral_denoise(image)
        # 第二步：使用深度学习进行精细去噪
        final_denoised = self.deep_learning_denoise(traditional_denoised)
        return final_denoised
    
    def hybrid_denoise_v2(self, image):
        """混合方法2：多方法融合"""
        # 获取多种传统方法的结果
        wavelet_denoised = self.traditional_denoiser.wavelet_denoise(image)
        bilateral_denoised = self.traditional_denoiser.bilateral_denoise(image)
        # 获取深度学习结果
        dl_denoised = self.deep_learning_denoise(image)
        
        # 简单加权融合
        alpha = 0.3  # 传统方法权重
        beta = 0.7   # 深度学习方法权重
        
        # 将传统方法结果进行平均
        traditional_avg = cv2.addWeighted(wavelet_denoised, 0.5, bilateral_denoised, 0.5, 0)
        # 与传统方法和深度学习结果融合
        hybrid_result = cv2.addWeighted(traditional_avg, alpha, dl_denoised, beta, 0)
        
        return hybrid_result

def add_noise(image, noise_type='gaussian', intensity=25):
    """添加噪声到图像"""
    noisy_image = image.copy()
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
    
    elif noise_type == 'salt_pepper':
        salt_pepper_ratio = 0.05
        amount = intensity / 255.0 * salt_pepper_ratio
        
        # 盐噪声
        salt_mask = np.random.random(image.shape[:2]) < amount
        noisy_image[salt_mask] = 255
        # 椒噪声
        pepper_mask = np.random.random(image.shape[:2]) < amount
        noisy_image[pepper_mask] = 0
    
    return noisy_image

def calculate_psnr(original, denoised):
    """计算PSNR（峰值信噪比）"""
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def visualize_results(original, noisy, results_dict, save_path=None):
    """可视化结果"""
    num_methods = len(results_dict)
    fig, axes = plt.subplots(2, num_methods + 2, figsize=(20, 8))
    
    # 第一行：显示图像
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    for i, (method_name, result) in enumerate(results_dict.items(), 2):
        if i < axes.shape[1]:
            axes[0, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'{method_name}')
            axes[0, i].axis('off')
    
    # 第二行：显示残差（噪声）
    axes[1, 0].axis('off')
    
    residual_noisy = cv2.absdiff(original, noisy)
    axes[1, 1].imshow(residual_noisy, cmap='hot')
    axes[1, 1].set_title('Noise Pattern')
    axes[1, 1].axis('off')
    
    for i, (method_name, result) in enumerate(results_dict.items(), 2):
        if i < axes.shape[1]:
            residual = cv2.absdiff(original, result)
            axes[1, i].imshow(residual, cmap='hot')
            psnr = calculate_psnr(original, result)
            axes[1, i].set_title(f'Residual\nPSNR: {psnr:.2f}dB')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_image_path():
    """从用户输入获取图像路径"""
    while True:
        print("\n" + "="*50)
        print("图像去噪程序")
        print("="*50)
        print("请选择输入方式：")
        print("1. 输入图像文件路径")
        print("2. 使用示例图像")
        print("3. 退出程序")
        
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == '1':
            image_path = input("\n请输入图像文件路径: ").strip()
            image_path = image_path.strip('"\'')
            
            if os.path.exists(image_path):
                return image_path
            else:
                print(f"错误：文件 '{image_path}' 不存在，请重新输入。")
                
        elif choice == '2':
            print("\n使用示例图像...")
            return None
            
        elif choice == '3':
            print("程序退出。")
            exit()
            
        else:
            print("无效选择，请重新输入。")

def get_model_path():
    """从用户输入获取模型路径"""
    models_dir = "trained_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if model_files:
        print("\n发现以下模型文件：")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")
        print(f"{len(model_files) + 1}. 不使用模型（随机初始化）")
        print(f"{len(model_files) + 2}. 手动输入模型路径")
        
        try:
            choice = int(input("\n请选择模型文件: ").strip())
            if 1 <= choice <= len(model_files):
                return os.path.join(models_dir, model_files[choice-1])
            elif choice == len(model_files) + 1:
                return None
            elif choice == len(model_files) + 2:
                manual_path = input("请输入模型文件路径: ").strip()
                manual_path = manual_path.strip('"\'')
                return manual_path if os.path.exists(manual_path) else None
        except ValueError:
            print("无效选择，将不使用预训练模型。")
    
    else:
        print("\n未找到模型文件，将使用随机初始化的模型。")
        print("你可以先运行训练程序来训练模型。")
    
    return None

def get_image_size_choice():
    """获取图像尺寸选择"""
    print("\n请选择图像处理尺寸：")
    print("1. 缩图至横向MAX_Pixel像素（推荐，处理速度快）")
    print("2. 使用原图尺寸（处理速度慢，适合小图像）")
    
    choice = input("请选择 (1/2): ").strip() or '1'
    return choice

def load_and_process_image(image_path, size_choice):
    """加载并处理图像"""
    if image_path:
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError("无法读取图像文件")
            
            print(f"成功读取图像: {os.path.basename(image_path)}")
            print(f"原图尺寸: {original_image.shape[1]} x {original_image.shape[0]}")
            
            # 根据选择调整图像尺寸
            if size_choice == '1':
                # 缩图至横向MAX_Pixel像素
                h, w = original_image.shape[:2]
                if w > MAX_Pixel:
                    scale = MAX_Pixel / w
                    new_w = MAX_Pixel
                    new_h = int(h * scale)
                    resized_image = cv2.resize(original_image, (new_w, new_h))
                    print(f"已调整尺寸至: {new_w} x {new_h}")
                    return resized_image, original_image
                else:
                    print("图像宽度小于MAX_Pixel像素，使用原图")
                    return original_image, original_image
            else:
                # 使用原图
                h, w = original_image.shape[:2]
                if h > 2000 or w > 2000:
                    print("注意：图像尺寸较大，处理可能需要较长时间...")
                return original_image, original_image
                
        except Exception as e:
            print(f"读取图像时出错: {e}")
            print("使用示例图像代替...")
            sample_image = create_sample_image()
            return sample_image, None
    else:
        sample_image = create_sample_image()
        return sample_image, None

def create_sample_image():
    """创建示例图像"""
    height, width = 400, 600
    sample_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加渐变背景
    for i in range(height):
        for j in range(width):
            sample_image[i, j] = [
                int(128 + 127 * np.sin(i * 0.02)),
                int(128 + 127 * np.cos(j * 0.02)),
                int(128 + 127 * np.sin((i + j) * 0.01))
            ]
    
    # 添加一些形状
    cv2.rectangle(sample_image, (50, 50), (200, 150), (255, 0, 0), -1)
    cv2.circle(sample_image, (450, 120), 60, (0, 255, 0), -1)
    cv2.ellipse(sample_image, (300, 280), (80, 40), 45, 0, 360, (0, 0, 255), -1)
    
    # 添加文字
    cv2.putText(sample_image, "Sample Image", (150, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(sample_image, "for Denoising Test", (140, 390), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return sample_image

def get_noise_settings():
    """获取噪声设置"""
    print("\n请选择噪声类型：")
    print("1. 高斯噪声 (默认)")
    print("2. 椒盐噪声")
    
    noise_choice = input("请选择 (1/2): ").strip() or '1'
    
    if noise_choice == '2':
        noise_type = 'salt_pepper'
    else:
        noise_type = 'gaussian'
    
    try:
        intensity = int(input(f"请输入噪声强度 (1-100, 默认25): ").strip() or '25')
        intensity = max(1, min(100, intensity))
    except ValueError:
        intensity = 25
        print("使用默认噪声强度: 25")
    
    return noise_type, intensity

def train_model_program():
    """训练模型的独立程序"""
    print("\n" + "="*50)
    print("模型训练程序")
    print("="*50)
    
    # 创建训练数据
    print("生成训练数据...")
    num_samples = 1000  # 训练样本数量
    img_size = (128, 128)  # 训练图像尺寸
    
    clean_images = []
    noisy_images = []
    
    for i in range(num_samples):
        # 生成随机纹理图像作为干净图像
        clean_img = np.random.randint(0, 256, (img_size[0], img_size[1], 3), dtype=np.uint8)
        # 添加噪声
        noisy_img = add_noise(clean_img, noise_type='gaussian', intensity=25)
        
        clean_images.append(clean_img)
        noisy_images.append(noisy_img)
    
    # 创建数据集和数据加载器
    dataset = DenoisingDataset(clean_images, noisy_images)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型和训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DnCNN(channels=3)
    trainer = ModelTrainer(model, device)
    
    # 开始训练
    epochs = int(input("请输入训练轮数 (默认50): ").strip() or "50")
    trainer.train(train_loader, epochs=epochs)
    
    print("训练程序完成！")

def denoise_program():
    """去噪主程序"""
    # 获取模型路径
    model_path = get_model_path()
    
    # 获取图像尺寸选择
    size_choice = get_image_size_choice()
    
    # 初始化去噪器（自动加载模型）
    denoiser = HybridDenoiser(model_path)
    
    # 获取图像路径
    image_path = get_image_path()
    
    # 读取并处理图像
    processed_image, original_fullsize = load_and_process_image(image_path, size_choice)
    
    # 获取噪声设置
    noise_type, intensity = get_noise_settings()
    
    # 添加噪声
    print(f"\n添加{intensity}强度的{noise_type}噪声...")
    noisy_image = add_noise(processed_image, noise_type=noise_type, intensity=intensity)
    
    print("开始去噪处理...")
    
    # 应用不同的去噪方法
    results = {}
    
    try:
        # 1. 传统方法：小波去噪
        print("1/5 进行小波去噪...")
        results['Wavelet'] = denoiser.traditional_denoiser.wavelet_denoise(noisy_image)
        
        # 2. 传统方法：双边滤波
        print("2/5 进行双边滤波...")
        results['Bilateral'] = denoiser.traditional_denoiser.bilateral_denoise(noisy_image)
        
        # 3. 深度学习方法
        print("3/5 进行深度学习去噪...")
        results['DnCNN'] = denoiser.deep_learning_denoise(noisy_image)
        
        # 4. 混合方法1
        print("4/5 进行混合去噪方法1...")
        results['Hybrid V1'] = denoiser.hybrid_denoise_v1(noisy_image)
        
        # 5. 混合方法2
        print("5/5 进行混合去噪方法2...")
        results['Hybrid V2'] = denoiser.hybrid_denoise_v2(noisy_image)
        
    except Exception as e:
        print(f"去噪过程中出错: {e}")
        print("继续处理其他方法...")
    
    # 计算并显示PSNR（使用处理后的图像作为基准）
    print("\n" + "="*50)
    print("去噪效果评估 (PSNR):")
    print("="*50)
    noisy_psnr = calculate_psnr(processed_image, noisy_image)
    print(f"噪声图像 PSNR: {noisy_psnr:.2f} dB")
    
    for method, result in results.items():
        psnr = calculate_psnr(processed_image, result)
        improvement = psnr - noisy_psnr
        print(f"{method:12}: {psnr:6.2f} dB (+{improvement:5.2f} dB)")
    
    # 可视化结果
    print("\n生成结果对比图...")
    visualize_results(processed_image, noisy_image, results, 'denoising_comparison.png')
    
    # 保存结果
    output_dir = "denoising_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存处理后的原图（可能是缩略图）
    cv2.imwrite(f'{output_dir}/original_processed.jpg', processed_image)
    cv2.imwrite(f'{output_dir}/noisy_image.jpg', noisy_image)
    
    # 保存去噪结果
    for method, result in results.items():
        filename = f'denoised_{method.lower().replace(" ", "_")}.jpg'
        cv2.imwrite(f'{output_dir}/{filename}', result)
    
    # 如果选择了缩图且存在原图，保存原图信息
    if original_fullsize is not None and size_choice == '1':
        cv2.imwrite(f'{output_dir}/original_fullsize.jpg', original_fullsize)
        print(f"\n原图已保存: {output_dir}/original_fullsize.jpg")
    
    print(f"\n处理完成！所有结果已保存到 '{output_dir}' 文件夹。")
    if size_choice == '1':
        print("注意：去噪处理是基于缩略图进行的，原图已单独保存。")

# 主程序入口
def main():
    while True:
        print("\n" + "="*50)
        print("图像去噪系统")
        print("="*50)
        print("请选择功能：")
        print("1. 训练新模型")
        print("2. 图像去噪")
        print("3. 退出系统")
        
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == '1':
            train_model_program()
        elif choice == '2':
            denoise_program()
        elif choice == '3':
            print("系统退出。")
            break
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    main()