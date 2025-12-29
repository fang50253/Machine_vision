import os
import numpy as np
import cv2
import random
from config import RANDOM_NOISE_CONFIG

def add_mixed_noise(image, noise_types=None, intensities=None):
    """添加混合噪声 - 支持随机强度"""
    # 如果启用了随机噪声且没有指定强度，使用随机强度
    if RANDOM_NOISE_CONFIG['enabled'] and intensities is None:
        intensities = generate_random_intensities(noise_types)
        print(f"使用随机噪声强度: {dict(zip(noise_types, intensities))}")
    
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
            noise = np.random.poisson(noisy_image * intensity / 255.0)
            noisy_image = noise * (255.0 / intensity)
            
        elif noise_type == 'speckle':
            speckle = np.random.randn(*image.shape) * intensity * 0.01
            noisy_image = noisy_image + noisy_image * speckle
    
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def generate_random_intensities(noise_types):
    """生成随机噪声强度"""
    intensities = []
    for noise_type in noise_types:
        if noise_type == 'gaussian':
            min_val, max_val = RANDOM_NOISE_CONFIG['gaussian_range']
        elif noise_type == 'salt_pepper':
            min_val, max_val = RANDOM_NOISE_CONFIG['salt_pepper_range']
        elif noise_type == 'poisson':
            min_val, max_val = RANDOM_NOISE_CONFIG['poisson_range']
        elif noise_type == 'speckle':
            min_val, max_val = RANDOM_NOISE_CONFIG['speckle_range']
        else:
            min_val, max_val = (10, 40)  # 默认范围
        
        intensity = random.randint(min_val, max_val)
        intensities.append(intensity)
    
    return intensities

def generate_random_noise_types():
    """生成随机噪声类型组合"""
    if RANDOM_NOISE_CONFIG['enabled'] and random.random() < RANDOM_NOISE_CONFIG['mixed_noise_prob']:
        # 生成混合噪声
        all_noise_types = ['gaussian', 'salt_pepper', 'poisson', 'speckle']
        num_types = random.randint(2, min(3, len(all_noise_types)))  # 2-3种噪声混合
        noise_types = random.sample(all_noise_types, num_types)
    else:
        # 单一噪声
        noise_types = [random.choice(['gaussian', 'salt_pepper', 'poisson', 'speckle'])]
    
    return noise_types

def get_noise_settings_interactive():
    """交互式获取噪声设置 - 支持随机选项"""
    print("\n请选择噪声设置方式：")
    print("1. 手动设置噪声类型和强度")
    print("2. 使用随机噪声（推荐用于测试）")
    
    choice = input("请选择 (1/2, 默认2): ").strip() or '2'
    
    if choice == '1':
        # 手动设置（原有逻辑）
        return get_manual_noise_settings()
    else:
        # 随机设置
        return get_random_noise_settings()

def get_manual_noise_settings():
    """手动设置噪声（原有逻辑）"""
    print("\n请选择噪声类型：")
    print("1. 高斯噪声")
    print("2. 椒盐噪声") 
    print("3. 混合噪声 (高斯+椒盐)")
    print("4. 自定义混合噪声")
    
    noise_choice = input("请选择 (1/2/3/4): ").strip() or '1'
    
    if noise_choice == '2':
        noise_types = ['salt_pepper']
    elif noise_choice == '3':
        noise_types = ['gaussian', 'salt_pepper']
    elif noise_choice == '4':
        custom_types = input("请输入噪声类型(用逗号分隔, 如: gaussian,salt_pepper): ").strip()
        noise_types = [t.strip() for t in custom_types.split(',')]
    else:
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
        for n_type in noise_types:
            try:
                intensity = int(input(f"  {n_type} 噪声强度 (默认25): ").strip() or '25')
                intensity = max(1, min(100, intensity))
                intensities.append(intensity)
            except ValueError:
                intensities.append(25)
                print(f"  {n_type} 使用默认强度: 25")
    
    return noise_types, intensities

def get_random_noise_settings():
    """获取随机噪声设置"""
    noise_types = generate_random_noise_types()
    intensities = generate_random_intensities(noise_types)
    
    print(f"\n🎲 随机噪声设置:")
    print(f"   噪声类型: {noise_types}")
    print(f"   噪声强度: {intensities}")
    print(f"   配置范围: {RANDOM_NOISE_CONFIG}")
    
    return noise_types, intensities

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
            noise = np.random.poisson(noisy_image * intensity / 255.0)
            noisy_image = noise * (255.0 / intensity)
            
        elif noise_type == 'speckle':
            speckle = np.random.randn(*image.shape) * intensity * 0.01
            noisy_image = noisy_image + noisy_image * speckle
    
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def add_noise_debug(image, noise_type='gaussian', intensity=25):
    """调试版的噪声添加函数"""
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

def get_model_path():
    """获取模型路径"""
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


def calculate_psnr(img1, img2, max_val=255.0):
    """
    计算峰值信噪比 (PSNR)
    
    参数:
        img1: 图像1 (numpy数组)
        img2: 图像2 (numpy数组)
        max_val: 最大像素值
    
    返回:
        PSNR值 (dB)
    """
    try:
        # 确保尺寸相同
        if img1.shape != img2.shape:
            # 调整尺寸
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 确保数据类型
        img1_float = img1.astype(np.float64)
        img2_float = img2.astype(np.float64)
        
        # 计算MSE
        mse = np.mean((img1_float - img2_float) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # 计算PSNR
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr
    
    except Exception as e:
        print(f"PSNR计算错误: {e}")
        return 0.0

def calculate_ssim(img1, img2):
    """
    计算结构相似性指数 (SSIM)
    
    参数:
        img1: 图像1 (numpy数组)
        img2: 图像2 (numpy数组)
    
    返回:
        SSIM值 (0-1之间)
    """
    try:
        from skimage.metrics import structural_similarity as ssim_calc
        
        # 确保尺寸相同
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # 确保数据类型
        if img1.dtype != np.uint8:
            img1 = img1.astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = img2.astype(np.uint8)
        
        # 确保数值范围
        if img1.max() <= 1.0:
            img1 = (img1 * 255).astype(np.uint8)
        if img2.max() <= 1.0:
            img2 = (img2 * 255).astype(np.uint8)
        
        # 计算SSIM
        if len(img1.shape) == 3:
            # 彩色图像
            ssim_value = ssim_calc(img1, img2, 
                                  data_range=255,
                                  channel_axis=2,
                                  win_size=7)
        else:
            # 灰度图像
            ssim_value = ssim_calc(img1, img2,
                                  data_range=255,
                                  win_size=7)
        
        return ssim_value
    
    except ImportError:
        print("警告: skimage库未安装，无法计算SSIM")
        return 0.0
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        return 0.0