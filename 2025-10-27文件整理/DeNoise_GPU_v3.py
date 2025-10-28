# DeNoise_GPU 2025-10-27版本更新说明：
# 删除了锐化相关代码

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import glob
import pandas as pd
from datetime import datetime
import time

# 设置随机种子以保证结果可重现
np.random.seed(42)
torch.manual_seed(42)

MAX_Pixel = 1024
NUM_Laters = 17

class ImprovedDnCNN(nn.Module):
    """改进的DnCNN模型 - NUM_Laters层深度"""
    
    def __init__(self, channels=3, num_layers=NUM_Laters, num_features=64):
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

class TraditionalDenoiser:
    """传统图像去噪方法"""
    
    @staticmethod
    def wavelet_denoise(image, wavelet='db4', level=2, threshold=0.1):
        """小波变换去噪"""
        try:
            import pywt
            image_float = image.astype(np.float32) / 255.0
            
            if len(image.shape) == 3:
                denoised = np.zeros_like(image_float)
                for i in range(3):
                    coeffs = pywt.wavedec2(image_float[:,:,i], wavelet, level=level)
                    coeffs_thresh = []
                    coeffs_thresh.append(coeffs[0])
                    for level_coeff in coeffs[1:]:
                        thresh_coeff = []
                        for detail in level_coeff:
                            thresh_detail = pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft')
                            thresh_coeff.append(thresh_detail)
                        coeffs_thresh.append(tuple(thresh_coeff))
                    
                    denoised[:,:,i] = pywt.waverec2(coeffs_thresh, wavelet)
            else:
                coeffs = pywt.wavedec2(image_float, wavelet, level=level)
                coeffs_thresh = []
                coeffs_thresh.append(coeffs[0])
                for level_coeff in coeffs[1:]:
                    thresh_coeff = []
                    for detail in level_coeff:
                        thresh_detail = pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft')
                        thresh_coeff.append(thresh_detail)
                    coeffs_thresh.append(tuple(thresh_coeff))
                
                denoised = pywt.waverec2(coeffs_thresh, wavelet)
            
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
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_denoise(image, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def gaussian_denoise(image, kernel_size=5, sigma=1.0):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

class AdvancedDenoiser:
    def __init__(self, model_path=None):
        self.traditional_denoiser = TraditionalDenoiser()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.load_pretrained_model(model_path)
        # 添加模型诊断
        self.diagnose_model()
    
    def load_pretrained_model(self, model_path=None):
        self.model = ImprovedDnCNN(channels=3, num_layers=NUM_Laters, num_features=64)
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                # 检查模型文件
                file_size = os.path.getsize(model_path) / 1024 / 1024  # MB
                print(f"模型文件大小: {file_size:.2f} MB")
                
                # 加载模型
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()  # 确保设置为评估模式
                
                # 检查模型参数
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"模型参数总数: {total_params:,}")
                
                print(f"✅ 预训练模型加载成功: {os.path.basename(model_path)}")
                
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                print("使用随机初始化的模型")
        else:
            print("❌ 未提供模型路径或文件不存在，使用随机初始化的模型")
    
    def diagnose_model(self):
        """诊断模型状态"""
        print(f"\n🔍 模型诊断信息:")
        print(f"   设备: {self.device}")
        
        if self.model is not None:
            # 检查模型是否在eval模式
            print(f"   训练模式: {'训练' if self.model.training else '评估'}")
            
            # 检查参数是否全为零（未训练）
            with torch.no_grad():
                sample_input = torch.randn(1, 3, 64, 64).to(self.device)
                output = self.model(sample_input)
                output_range = output.abs().max().item()
                print(f"   测试输出范围: {output_range:.6f}")
    
    def preprocess_image(self, image):
        """改进的预处理 - 确保与第一个代码一致"""
        # 确保图像是uint8类型
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image_tensor.unsqueeze(0).to(self.device)
    
    def postprocess_image(self, tensor):
        """改进的后处理"""
        image = tensor.squeeze(0).cpu().detach().numpy()
        image = image.transpose(1, 2, 0)
        # 确保数值范围正确
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        return image
    
    def deep_learning_denoise(self, image):
        if self.model is None:
            print("错误：模型未初始化")
            return image
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output_tensor = self.model(input_tensor)
            denoised_image = self.postprocess_image(output_tensor)
        
        return denoised_image
    
    def test_model_effectiveness(self, test_image):
        """测试模型有效性"""
        print("\n🧪 测试模型有效性...")
        
        # 添加强噪声
        strong_noise = np.random.normal(0, 50, test_image.shape).astype(np.float32)
        noisy_test = np.clip(test_image.astype(np.float32) + strong_noise, 0, 255).astype(np.uint8)
        
        # 使用深度学习去噪
        dl_denoised = self.deep_learning_denoise(noisy_test)
        
        # 使用传统方法对比
        median_denoised = self.traditional_denoiser.median_denoise(noisy_test)
        
        # 计算PSNR
        original_psnr = calculate_psnr(test_image, noisy_test)
        dl_psnr = calculate_psnr(test_image, dl_denoised)
        median_psnr = calculate_psnr(test_image, median_denoised)
        
        print(f"📊 噪声图像 PSNR: {original_psnr:.2f} dB")
        print(f"📊 深度学习 PSNR: {dl_psnr:.2f} dB")
        print(f"📊 中值滤波 PSNR: {median_psnr:.2f} dB")
        
        improvement = dl_psnr - original_psnr
        print(f"📈 深度学习改进: {improvement:+.2f} dB")
        
        if improvement < 1.0:
            print("⚠️  警告: 深度学习模型效果不佳！")
            return False
        else:
            print("✅ 深度学习模型工作正常")
            return True

    def hybrid_denoise_v1(self, image):
        traditional_denoised = self.traditional_denoiser.bilateral_denoise(image)
        final_denoised = self.deep_learning_denoise(traditional_denoised)
        return final_denoised
    
    def hybrid_denoise_v2(self, image):
        wavelet_denoised = self.traditional_denoiser.wavelet_denoise(image)
        bilateral_denoised = self.traditional_denoiser.bilateral_denoise(image)
        dl_denoised = self.deep_learning_denoise(image)
        
        alpha = 0.3
        beta = 0.7
        
        traditional_avg = cv2.addWeighted(wavelet_denoised, 0.5, bilateral_denoised, 0.5, 0)
        hybrid_result = cv2.addWeighted(traditional_avg, alpha, dl_denoised, beta, 0)
        
        return hybrid_result

def get_model_path():
    possible_dirs = ["improved_models", "trained_models", "models"]
    model_files = []
    
    for model_dir in possible_dirs:
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and 'best' in f]
            model_files.extend([os.path.join(model_dir, f) for f in files])
    
    if model_files:
        print("\n发现以下模型文件：")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")
        print(f"{len(model_files) + 1}. 不使用模型（随机初始化）")
        print(f"{len(model_files) + 2}. 手动输入模型路径")
        
        try:
            choice = int(input("\n请选择模型文件: ").strip())
            if 1 <= choice <= len(model_files):
                return model_files[choice-1]
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
        print("请先运行训练程序来训练模型。")
    
    return None

def add_mixed_noise(image, noise_types=None, intensities=None):
    """添加混合噪声"""
    if noise_types is None:
        noise_types = ['gaussian', 'salt_pepper']
    if intensities is None:
        intensities = [25, 25]
    
    noisy_image = image.copy().astype(np.float32)
    
    for noise_type, intensity in zip(noise_types, intensities):
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
            noisy_image = noisy_image + noise
            
        elif noise_type == 'salt_pepper':
            amount = intensity / 200.0
            salt_mask = np.random.random(image.shape[:2]) < amount
            pepper_mask = np.random.random(image.shape[:2]) < amount
            noisy_image[salt_mask] = 255
            noisy_image[pepper_mask] = 0
            
        elif noise_type == 'poisson':
            # 泊松噪声
            noise = np.random.poisson(noisy_image * intensity / 255.0)
            noisy_image = noise * (255.0 / intensity)
            
        elif noise_type == 'speckle':
            # 散斑噪声
            speckle = np.random.randn(*image.shape) * intensity * 0.01
            noisy_image = noisy_image + noisy_image * speckle
    
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_noise_debug(image, noise_type='gaussian', intensity=25):
    """调试版的噪声添加函数，确保噪声被正确添加"""
    print(f"添加噪声前 - 图像范围: [{image.min()}, {image.max()}], 形状: {image.shape}")
    
    noisy_image = image.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
        print(f"高斯噪声 - 均值: {noise.mean():.2f}, 标准差: {noise.std():.2f}")
        noisy_image = noisy_image + noise
        
    elif noise_type == 'salt_pepper':
        amount = intensity / 500.0
        print(f"椒盐噪声 - 强度: {intensity}, 比例: {amount:.4f}")
        salt_mask = np.random.random(image.shape[:2]) < amount
        noisy_image[salt_mask] = 255
        pepper_mask = np.random.random(image.shape[:2]) < amount
        noisy_image[pepper_mask] = 0
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    print(f"添加噪声后 - 图像范围: [{noisy_image.min()}, {noisy_image.max()}]")
    
    noise_diff = noisy_image.astype(np.float32) - image.astype(np.float32)
    print(f"噪声差异 - 均值: {noise_diff.mean():.2f}, 标准差: {noise_diff.std():.2f}")
    
    return noisy_image

def calculate_psnr(original, denoised):
    original_float = original.astype(np.float64)
    denoised_float = denoised.astype(np.float64)
    
    mse = np.mean((original_float - denoised_float) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, denoised):
    try:
        from skimage.metrics import structural_similarity as compare_ssim
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            denoised_gray = denoised
        return compare_ssim(original_gray, denoised_gray, data_range=255)
    except ImportError:
        return 0

def normalize_psnr(psnr_values):
    """标准化PSNR值到0-1范围"""
    if not psnr_values:
        return []
    
    min_psnr = min(psnr_values)
    max_psnr = max(psnr_values)
    
    if max_psnr == min_psnr:
        return [1.0] * len(psnr_values)
    
    normalized = [(psnr - min_psnr) / (max_psnr - min_psnr) for psnr in psnr_values]
    return normalized

def get_input_mode():
    """获取输入模式选择"""
    print("\n" + "="*50)
    print("高级图像去噪测试程序")
    print("="*50)
    print("请选择处理模式：")
    print("1. 单张图像处理")
    print("2. 批量文件夹处理")
    print("3. 退出程序")
    
    while True:
        choice = input("\n请选择 (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        else:
            print("无效选择，请重新输入。")

def get_folder_path():
    """获取文件夹路径 - 修复版"""
    while True:
        folder_path = input("\n请输入包含图像的文件夹路径: ").strip().strip('"\'')
        
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 '{folder_path}' 不存在，请重新输入。")
            continue
            
        # 改进的图像文件搜索逻辑
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        
        print(f"正在搜索文件夹: {folder_path}")
        
        # 方法1: 使用os.walk递归搜索
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        # 方法2: 如果方法1没找到，尝试直接搜索当前文件夹
        if not image_files:
            print("尝试直接搜索当前文件夹...")
            for file in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file)):
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
                        full_path = os.path.join(folder_path, file)
                        image_files.append(full_path)
        
        # 显示调试信息
        print(f"找到的文件总数: {len(os.listdir(folder_path))}")
        print(f"识别出的图像文件: {len(image_files)}")
        
        if not image_files:
            print(f"在文件夹 '{folder_path}' 中未找到图像文件。")
            print("支持的格式: JPG, JPEG, PNG, BMP, TIFF, TIF")
            
            # 显示文件夹内容以便调试
            try:
                all_files = os.listdir(folder_path)
                print(f"\n文件夹内容 ({len(all_files)} 个文件/文件夹):")
                for i, item in enumerate(all_files[:10]):  # 只显示前10个
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        print(f"  文件: {item}")
                    else:
                        print(f"  文件夹: {item}/")
                if len(all_files) > 10:
                    print(f"  ... 还有 {len(all_files) - 10} 个项")
            except Exception as e:
                print(f"无法列出文件夹内容: {e}")
            
            continue
            
        # 去重和排序
        image_files = list(set(image_files))
        image_files.sort()
        
        print(f"\n找到 {len(image_files)} 个图像文件:")
        for i, file_path in enumerate(image_files[:5]):
            file_size = os.path.getsize(file_path) // 1024  # KB
            print(f"  {i+1}. {os.path.basename(file_path)} ({file_size} KB)")
        if len(image_files) > 5:
            print(f"  ... 还有 {len(image_files) - 5} 个文件")
            
        confirm = input("\n是否处理这些文件？(y/n): ").strip().lower()
        if confirm in ['y', 'yes', '']:
            return folder_path, image_files
        else:
            print("请重新选择文件夹。")

def get_noise_settings():
    """获取噪声设置"""
    print("\n请选择噪声类型：")
    print("1. 高斯噪声")
    print("2. 椒盐噪声") 
    print("3. 混合噪声 (高斯+椒盐)")
    print("4. 自定义混合噪声")
    
    noise_choice = input("请选择 (1/2/3/4): ").strip() or '1'
    
    if noise_choice == '2':
        noise_type = 'salt_pepper'
        noise_types = ['salt_pepper']
    elif noise_choice == '3':
        noise_type = 'mixed'
        noise_types = ['gaussian', 'salt_pepper']
    elif noise_choice == '4':
        print("\n可选的噪声类型: gaussian, salt_pepper, poisson, speckle")
        custom_types = input("请输入噪声类型(用逗号分隔, 如: gaussian,salt_pepper): ").strip()
        noise_types = [t.strip() for t in custom_types.split(',')]
        noise_type = 'custom_mixed'
    else:
        noise_type = 'gaussian'
        noise_types = ['gaussian']
    
    intensities = []
    if len(noise_types) == 1:
        try:
            intensity = int(input(f"请输入噪声强度 (1-100, 默认25): ").strip() or '25')
            intensity = max(1, min(100, intensity))
            intensities = [intensity]
        except ValueError:
            intensities = [25]
            print("使用默认噪声强度: 25")
    else:
        print("\n请为每种噪声类型设置强度 (1-100):")
        for i, n_type in enumerate(noise_types):
            try:
                intensity = int(input(f"  {n_type} 噪声强度 (默认25): ").strip() or '25')
                intensity = max(1, min(100, intensity))
                intensities.append(intensity)
            except ValueError:
                intensities.append(25)
                print(f"  {n_type} 使用默认强度: 25")
    
    return noise_type, noise_types, intensities

def process_single_image():
    """处理单张图像"""
    from Show import get_image_path, get_model_path, get_image_size_choice, load_and_process_image, denoise_program
    
    model_path = get_model_path()
    size_choice = get_image_size_choice()
    image_path = get_image_path()
    processed_image, original_fullsize = load_and_process_image(image_path, size_choice)
    
    noise_type, noise_types, intensities = get_noise_settings()
    
    print(f"\n添加{noise_type}噪声...")
    if len(noise_types) == 1:
        noisy_image = add_noise_debug(processed_image, noise_type=noise_types[0], intensity=intensities[0])
    else:
        noisy_image = add_mixed_noise(processed_image, noise_types, intensities)
        print(f"混合噪声类型: {noise_types}")
        print(f"噪声强度: {intensities}")
    
    # 继续原有的单张图像处理流程
    denoiser = AdvancedDenoiser(model_path)
    
    print("开始去噪处理...")
    results = {}
    
    try:
        print("1/5 进行小波去噪...")
        results['Wavelet'] = denoiser.traditional_denoiser.wavelet_denoise(noisy_image)
        
        print("2/5 进行双边滤波...")
        results['Bilateral'] = denoiser.traditional_denoiser.bilateral_denoise(noisy_image)
        
        print("3/5 进行深度学习去噪...")
        results['DnCNN'] = denoiser.deep_learning_denoise(noisy_image)
        
        print("4/5 进行混合去噪方法1...")
        results['Hybrid V1'] = denoiser.hybrid_denoise_v1(noisy_image)
        
        print("5/5 进行混合去噪方法2...")
        results['Hybrid V2'] = denoiser.hybrid_denoise_v2(noisy_image)
        
    except Exception as e:
        print(f"去噪过程中出错: {e}")
    
    # 计算评估指标
    print("\n" + "="*50)
    print("去噪效果评估")
    print("="*50)
    
    noisy_psnr = calculate_psnr(processed_image, noisy_image)
    noisy_ssim = calculate_ssim(processed_image, noisy_image)
    
    print(f"噪声图像 - PSNR: {noisy_psnr:.2f} dB")
    if noisy_ssim > 0:
        print(f"噪声图像 - SSIM: {noisy_ssim:.3f}")
    
    print("\n各方法效果对比:")
    print("-" * 50)
    
    psnr_values = []
    for method, result in results.items():
        psnr = calculate_psnr(processed_image, result)
        ssim_val = calculate_ssim(processed_image, result)
        improvement_psnr = psnr - noisy_psnr
        
        psnr_values.append(psnr)
        
        if noisy_ssim > 0 and ssim_val > 0:
            improvement_ssim = ssim_val - noisy_ssim
            print(f"{method:12}: PSNR: {psnr:6.2f} dB (+{improvement_psnr:5.2f}) | SSIM: {ssim_val:.3f} (+{improvement_ssim:.3f})")
        else:
            print(f"{method:12}: PSNR: {psnr:6.2f} dB (+{improvement_psnr:5.2f})")
    
    # 标准化PSNR值
    normalized_psnr = normalize_psnr(psnr_values)
    print("\n标准化PSNR值 (0-1范围):")
    for i, (method, norm_psnr) in enumerate(zip(results.keys(), normalized_psnr)):
        print(f"{method:12}: {norm_psnr:.3f}")
    
    return {
        'noise_type': noise_type,
        'noise_intensities': intensities,
        'noisy_psnr': noisy_psnr,
        'noisy_ssim': noisy_ssim,
        'results': results,
        'psnr_values': psnr_values,
        'normalized_psnr': normalized_psnr
    }

def process_batch_images():
    """批量处理文件夹中的图像"""
    folder_path, image_files = get_folder_path()
    model_path = get_model_path()
    noise_type, noise_types, intensities = get_noise_settings()
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"batch_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    denoiser = AdvancedDenoiser(model_path)
    results_data = []
    
    print(f"\n开始批量处理 {len(image_files)} 张图像...")
    print(f"噪声设置: {noise_type}, 强度: {intensities}")
    print(f"结果将保存到: {output_dir}")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理进度: {i}/{len(image_files)}")
        print(f"当前图像: {os.path.basename(image_path)}")
        
        try:
            # 读取图像
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"  警告: 无法读取图像 {image_path}，跳过")
                continue
                
            # 调整尺寸
            h, w = original_image.shape[:2]
            if w > MAX_Pixel:
                scale = MAX_Pixel / w
                new_w = MAX_Pixel
                new_h = int(h * scale)
                processed_image = cv2.resize(original_image, (new_w, new_h))
            else:
                processed_image = original_image
            
            # 添加噪声
            if len(noise_types) == 1:
                noisy_image = add_mixed_noise(processed_image, [noise_types[0]], [intensities[0]])
            else:
                noisy_image = add_mixed_noise(processed_image, noise_types, intensities)
            
            # 应用去噪方法
            denoising_results = {}
            
            # 小波去噪
            try:
                denoising_results['Wavelet'] = denoiser.traditional_denoiser.wavelet_denoise(noisy_image)
            except Exception as e:
                print(f"  小波去噪失败: {e}")
                denoising_results['Wavelet'] = processed_image
            
            # 双边滤波
            try:
                denoising_results['Bilateral'] = denoiser.traditional_denoiser.bilateral_denoise(noisy_image)
            except Exception as e:
                print(f"  双边滤波失败: {e}")
                denoising_results['Bilateral'] = processed_image
            
            # 深度学习去噪
            try:
                denoising_results['DnCNN'] = denoiser.deep_learning_denoise(noisy_image)
            except Exception as e:
                print(f"  深度学习去噪失败: {e}")
                denoising_results['DnCNN'] = processed_image
            
            # 混合去噪方法1
            try:
                denoising_results['Hybrid_V1'] = denoiser.hybrid_denoise_v1(noisy_image)
            except Exception as e:
                print(f"  混合去噪V1失败: {e}")
                denoising_results['Hybrid_V1'] = processed_image
            
            # 混合去噪方法2
            try:
                denoising_results['Hybrid_V2'] = denoiser.hybrid_denoise_v2(noisy_image)
            except Exception as e:
                print(f"  混合去噪V2失败: {e}")
                denoising_results['Hybrid_V2'] = processed_image
            
            # 计算指标
            noisy_psnr = calculate_psnr(processed_image, noisy_image)
            noisy_ssim = calculate_ssim(processed_image, noisy_image)
            
            method_psnrs = {}
            method_ssims = {}
            
            for method, result in denoising_results.items():
                method_psnrs[method] = calculate_psnr(processed_image, result)
                method_ssims[method] = calculate_ssim(processed_image, result)
            
            # 标准化PSNR
            psnr_values = list(method_psnrs.values())
            normalized_psnr = normalize_psnr(psnr_values)
            norm_psnr_dict = dict(zip(denoising_results.keys(), normalized_psnr))
            
            # 保存结果数据
            image_result = {
                'image_name': os.path.basename(image_path),
                'image_path': image_path,
                'image_size': f"{processed_image.shape[1]}x{processed_image.shape[0]}",
                'noise_type': noise_type,
                'noise_intensities': str(intensities),
                'noisy_psnr': noisy_psnr,
                'noisy_ssim': noisy_ssim
            }
            
            # 添加各方法的指标
            for method in denoising_results.keys():
                image_result[f'{method}_psnr'] = method_psnrs[method]
                image_result[f'{method}_ssim'] = method_ssims[method]
                image_result[f'{method}_psnr_norm'] = norm_psnr_dict[method]
            
            results_data.append(image_result)
            
            # 保存处理后的图像
            img_output_dir = os.path.join(output_dir, 'images', os.path.splitext(os.path.basename(image_path))[0])
            os.makedirs(img_output_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(img_output_dir, 'original.jpg'), processed_image)
            cv2.imwrite(os.path.join(img_output_dir, 'noisy.jpg'), noisy_image)
            
            for method, result in denoising_results.items():
                cv2.imwrite(os.path.join(img_output_dir, f'{method}.jpg'), result)
            
            print(f"  完成: PSNR={noisy_psnr:.2f}dB, 已保存结果")
            
        except Exception as e:
            print(f"  处理图像 {image_path} 时出错: {e}")
            continue
    
    # 保存CSV结果
    if results_data:
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, 'denoising_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 计算平均指标
        print(f"\n{'='*50}")
        print("批量处理完成！")
        print(f"{'='*50}")
        print(f"处理图像数量: {len(results_data)}")
        print(f"结果CSV文件: {csv_path}")
        print(f"图像输出目录: {output_dir}/images/")
        
        # 显示平均PSNR
        print("\n各方法平均PSNR:")
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        for method in methods:
            avg_psnr = df[f'{method}_psnr'].mean()
            avg_norm_psnr = df[f'{method}_psnr_norm'].mean()
            print(f"  {method:12}: {avg_psnr:.2f} dB (标准化: {avg_norm_psnr:.3f})")
    
    return results_data

def main():
    print("PyTorch 版本:", torch.__version__)
    print("是否支持 CUDA:", torch.cuda.is_available())
    print("当前设备:", "cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print("高级图像去噪测试系统 - 批量处理版")
    print("="*60)
    
    while True:
        mode = get_input_mode()
        
        if mode == '1':
            print("\n进入单张图像处理模式...")
            process_single_image()
            break
        elif mode == '2':
            print("\n进入批量处理模式...")
            process_batch_images()
            break
        elif mode == '3':
            print("程序退出。")
            break

if __name__ == "__main__":
    main()