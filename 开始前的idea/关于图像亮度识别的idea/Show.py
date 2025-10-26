import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import glob

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
    
    def load_pretrained_model(self, model_path=None):
        self.model = ImprovedDnCNN(channels=3, num_layers=NUM_Laters, num_features=64)
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
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image_tensor.unsqueeze(0).to(self.device)
    
    def postprocess_image(self, tensor):
        image = tensor.squeeze(0).cpu().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
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

def add_noise_debug(image, noise_type='gaussian', intensity=25):
    """调试版的噪声添加函数，确保噪声被正确添加"""
    print(f"添加噪声前 - 图像范围: [{image.min()}, {image.max()}], 形状: {image.shape}")
    
    noisy_image = image.copy().astype(np.float32)  # 使用浮点数计算
    
    if noise_type == 'gaussian':
        # 生成高斯噪声
        noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
        print(f"高斯噪声 - 均值: {noise.mean():.2f}, 标准差: {noise.std():.2f}")
        
        noisy_image = noisy_image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        
    elif noise_type == 'salt_pepper':
        amount = intensity / 500.0  # 调整比例
        print(f"椒盐噪声 - 强度: {intensity}, 比例: {amount:.4f}")
        
        # 盐噪声
        salt_mask = np.random.random(image.shape[:2]) < amount
        noisy_image[salt_mask] = 255
        # 椒噪声
        pepper_mask = np.random.random(image.shape[:2]) < amount
        noisy_image[pepper_mask] = 0
    
    noisy_image = noisy_image.astype(np.uint8)
    print(f"添加噪声后 - 图像范围: [{noisy_image.min()}, {noisy_image.max()}]")
    
    # 计算噪声差异
    noise_diff = noisy_image.astype(np.float32) - image.astype(np.float32)
    print(f"噪声差异 - 均值: {noise_diff.mean():.2f}, 标准差: {noise_diff.std():.2f}")
    
    return noisy_image

def add_noise(image, noise_type='gaussian', intensity=25):
    """改进的噪声添加函数"""
    noisy_image = image.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        # 确保噪声强度足够明显
        noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
        noisy_image = noisy_image + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        
    elif noise_type == 'salt_pepper':
        # 增加椒盐噪声的可见性
        amount = intensity / 200.0  # 增加比例使噪声更明显
        salt_mask = np.random.random(image.shape[:2]) < amount
        pepper_mask = np.random.random(image.shape[:2]) < amount
        noisy_image[salt_mask] = 255
        noisy_image[pepper_mask] = 0
    
    return noisy_image.astype(np.uint8)

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

def visualize_results(original, noisy, results_dict, save_path=None):
    """改进的可视化函数，确保噪声图像正确显示"""
    num_methods = len(results_dict)
    fig, axes = plt.subplots(2, num_methods + 2, figsize=(20, 8))
    
    # 第一行：显示图像
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 确保噪声图像正确显示
    noisy_display = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(noisy_display)
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    for i, (method_name, result) in enumerate(results_dict.items(), 2):
        if i < axes.shape[1]:
            axes[0, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'{method_name}')
            axes[0, i].axis('off')
    
    # 第二行：显示残差
    axes[1, 0].axis('off')
    
    # 计算并显示噪声模式
    residual_noisy = cv2.absdiff(original, noisy)
    axes[1, 1].imshow(residual_noisy, cmap='hot')
    axes[1, 1].set_title('Noise Pattern')
    axes[1, 1].axis('off')
    
    for i, (method_name, result) in enumerate(results_dict.items(), 2):
        if i < axes.shape[1]:
            residual = cv2.absdiff(original, result)
            axes[1, i].imshow(residual, cmap='hot')
            psnr = calculate_psnr(original, result)
            ssim_val = calculate_ssim(original, result)
            if ssim_val > 0:
                axes[1, i].set_title(f'Residual\nPSNR: {psnr:.2f}dB\nSSIM: {ssim_val:.3f}')
            else:
                axes[1, i].set_title(f'Residual\nPSNR: {psnr:.2f}dB')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_image_path():
    while True:
        print("\n" + "="*50)
        print("高级图像去噪测试程序")
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

def get_image_size_choice():
    print("\n请选择图像处理尺寸：")
    print(f"1. 缩图至横向{MAX_Pixel}像素（推荐，处理速度快）")
    print("2. 使用原图尺寸（处理速度慢，适合小图像）")
    
    choice = input("请选择 (1/2): ").strip() or '1'
    return choice

def load_and_process_image(image_path, size_choice):
    if image_path:
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError("无法读取图像文件")
            
            print(f"成功读取图像: {os.path.basename(image_path)}")
            print(f"原图尺寸: {original_image.shape[1]} x {original_image.shape[0]}")
            
            if size_choice == '1':
                h, w = original_image.shape[:2]
                if w > MAX_Pixel:
                    scale = MAX_Pixel / w
                    new_w = MAX_Pixel
                    new_h = int(h * scale)
                    resized_image = cv2.resize(original_image, (new_w, new_h))
                    print(f"已调整尺寸至: {new_w} x {new_h}")
                    return resized_image, original_image
                else:
                    print(f"图像宽度小于{MAX_Pixel}像素，使用原图")
                    return original_image, original_image
            else:
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
    height, width = 512, 768
    sample_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            sample_image[i, j] = [
                int(128 + 127 * np.sin(i * 0.01) * np.cos(j * 0.01)),
                int(128 + 127 * np.sin((i + j) * 0.005)),
                int(128 + 127 * np.cos(i * 0.008) * np.sin(j * 0.008))
            ]
    
    cv2.rectangle(sample_image, (50, 50), (200, 150), (255, 0, 0), -1)
    cv2.circle(sample_image, (450, 120), 60, (0, 255, 0), -1)
    cv2.ellipse(sample_image, (300, 280), (80, 40), 45, 0, 360, (0, 0, 255), -1)
    cv2.line(sample_image, (100, 300), (400, 300), (255, 255, 0), 3)
    
    cv2.putText(sample_image, "Sample Test Image", (150, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(sample_image, "for Advanced Denoising", (120, 390), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return sample_image

def get_noise_settings():
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

def denoise_program():
    # 获取模型路径
    model_path = get_model_path()
    
    # 获取图像尺寸选择
    size_choice = get_image_size_choice()
    
    # 初始化去噪器
    denoiser = AdvancedDenoiser(model_path)
    
    # 获取图像路径
    image_path = get_image_path()
    
    # 读取并处理图像
    processed_image, original_fullsize = load_and_process_image(image_path, size_choice)
    
    # 获取噪声设置
    noise_type, intensity = get_noise_settings()
    
    # 添加噪声 - 使用调试版本确认噪声被正确添加
    print(f"\n添加{intensity}强度的{noise_type}噪声...")
    noisy_image = add_noise_debug(processed_image, noise_type=noise_type, intensity=intensity)
    
    # 验证噪声是否成功添加
    noise_diff = noisy_image.astype(np.float32) - processed_image.astype(np.float32)
    print(f"噪声验证 - 差异均值: {noise_diff.mean():.2f}, 差异标准差: {noise_diff.std():.2f}")
    
    if abs(noise_diff.mean()) < 1.0 and noise_diff.std() < 1.0:
        print("警告：噪声可能未正确添加！")
        print("尝试增加噪声强度或检查噪声类型...")
    
    print("开始去噪处理...")
    
    # 应用不同的去噪方法
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
    
    # 计算并显示评估指标
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
    for method, result in results.items():
        psnr = calculate_psnr(processed_image, result)
        ssim_val = calculate_ssim(processed_image, result)
        improvement_psnr = psnr - noisy_psnr
        
        if noisy_ssim > 0 and ssim_val > 0:
            improvement_ssim = ssim_val - noisy_ssim
            print(f"{method:12}: PSNR: {psnr:6.2f} dB (+{improvement_psnr:5.2f}) | SSIM: {ssim_val:.3f} (+{improvement_ssim:.3f})")
        else:
            print(f"{method:12}: PSNR: {psnr:6.2f} dB (+{improvement_psnr:5.2f})")
    
    # 可视化结果
    print("\n生成结果对比图...")
    visualize_results(processed_image, noisy_image, results, 'denoising_comparison.png')
    
    # 保存结果
    output_dir = "denoising_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(f'{output_dir}/original_processed.jpg', processed_image)
    cv2.imwrite(f'{output_dir}/noisy_image.jpg', noisy_image)
    
    for method, result in results.items():
        filename = f'denoised_{method.lower().replace(" ", "_")}.jpg'
        cv2.imwrite(f'{output_dir}/{filename}', result)
    
    if original_fullsize is not None and size_choice == '1':
        cv2.imwrite(f'{output_dir}/original_fullsize.jpg', original_fullsize)
        print(f"\n原图已保存: {output_dir}/original_fullsize.jpg")
    
    print(f"\n处理完成！所有结果已保存到 '{output_dir}' 文件夹。")

def main():
    print("PyTorch 版本:", torch.__version__)
    print("是否支持 MPS:", torch.backends.mps.is_built())
    print("MPS 是否可用:", torch.backends.mps.is_available())
    print("当前设备:", "mps" if torch.backends.mps.is_available() else "cpu")
    print("\n" + "="*60)
    print("高级图像去噪测试系统 - 调试版")
    print("="*60)
    print("说明：")
    print("- 此版本包含噪声添加的调试信息")
    print("- 会显示噪声添加前后的图像统计信息")
    print("- 帮助确认噪声是否被正确添加")
    
    denoise_program()

if __name__ == "__main__":
    main()