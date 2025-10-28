import os
import cv2
import pandas as pd
from datetime import datetime
from models.traditional_denoiser import AdvancedDenoiser
from utils.image_utils import add_mixed_noise, generate_random_noise_types, generate_random_intensities
from utils.metrics import calculate_psnr, calculate_ssim, normalize_psnr
from views.image_view import ImageView

class BatchController:
    """批量处理控制器"""
    
    def __init__(self):
        self.denoiser = None
        self.image_view = ImageView()
    
    def initialize_denoiser(self, model_path=None):
        """初始化去噪器"""
        self.denoiser = AdvancedDenoiser(model_path)
        return self.denoiser is not None
    
    def process_batch(self, folder_path, noise_types=None, intensities=None):
        """批量处理图像 - 支持随机噪声"""
        # 查找图像文件
        image_files = self.image_view.find_image_files(folder_path)
        if not image_files:
            raise ValueError("在指定文件夹中未找到图像文件")
        
        # 检查去噪器是否已初始化
        if self.denoiser is None:
            raise ValueError("去噪器未初始化，请先调用 initialize_denoiser()")
        
        # 如果没有提供噪声设置，使用随机设置
        if noise_types is None or intensities is None:
            use_random = input("\n是否为每张图像使用不同的随机噪声？(y/n, 默认y): ").strip().lower() in ['y', 'yes', '']
            if use_random:
                noise_types = None  # 将在每张图像处理时生成
                intensities = None
                print("🎲 将为每张图像使用不同的随机噪声")
            else:
                from utils.image_utils import get_noise_settings_interactive
                noise_types, intensities = get_noise_settings_interactive()
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"batch_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        results_data = []
        
        print(f"\n开始批量处理 {len(image_files)} 张图像...")
        if noise_types and intensities:
            print(f"噪声设置: {noise_types}, 强度: {intensities}")
        else:
            print("🎲 使用随机噪声（每张图像不同）")
        print(f"结果将保存到: {output_dir}")
        
        for i, image_path in enumerate(image_files, 1):
            try:
                # 如果使用随机噪声，为每张图像生成不同的噪声
                if noise_types is None or intensities is None:
                    from utils.image_utils import generate_random_noise_types, generate_random_intensities
                    img_noise_types = generate_random_noise_types()
                    img_intensities = generate_random_intensities(img_noise_types)
                else:
                    img_noise_types = noise_types
                    img_intensities = intensities
                
                result = self._process_single_image_batch(
                    image_path, img_noise_types, img_intensities, output_dir, i, len(image_files))
                if result:
                    results_data.append(result)
            except Exception as e:
                print(f"\n处理图像 {image_path} 时出错: {e}")
                continue
        
        # 保存结果
        if results_data:
            self._save_batch_results(results_data, output_dir)
        
        return results_data
    
    def _process_single_image_batch(self, image_path, noise_types, intensities, output_dir, current, total):
        """处理单张图像（批量模式）"""
        print(f"\n处理进度: {current}/{total}")
        print(f"当前图像: {os.path.basename(image_path)}")
        
        # 读取图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  警告: 无法读取图像 {image_path}，跳过")
            return None
        
        # 调整尺寸
        processed_image = self.image_view.resize_image(original_image)
        
        # 添加噪声
        if len(noise_types) == 1:
            noisy_image = add_mixed_noise(processed_image, [noise_types[0]], [intensities[0]])
        else:
            noisy_image = add_mixed_noise(processed_image, noise_types, intensities)
        
        # 应用去噪方法
        denoising_results = self._apply_denoising_methods_batch(noisy_image)
        
        # 计算指标
        metrics = self._calculate_batch_metrics(processed_image, noisy_image, denoising_results)
        
        # 保存图像
        self._save_batch_images(processed_image, noisy_image, denoising_results, image_path, output_dir)
        
        # 准备结果数据
        result = {
            'image_name': os.path.basename(image_path),
            'image_path': image_path,
            'image_size': f"{processed_image.shape[1]}x{processed_image.shape[0]}",
            'noise_types': str(noise_types),
            'noise_intensities': str(intensities),
            'noisy_psnr': metrics['noisy_psnr'],
            'noisy_ssim': metrics['noisy_ssim']
        }
        
        # 添加各方法的指标
        for method in denoising_results.keys():
            result[f'{method}_psnr'] = metrics['methods'][method]['psnr']
            result[f'{method}_ssim'] = metrics['methods'][method]['ssim']
            result[f'{method}_psnr_norm'] = metrics['methods'][method]['normalized_psnr']
        
        print(f"  完成: PSNR={metrics['noisy_psnr']:.2f}dB")
        return result
    
    def _apply_denoising_methods_batch(self, noisy_image):
        """应用去噪方法（批量模式）"""
        methods = {
            'Wavelet': lambda: self.denoiser.traditional_denoiser.wavelet_denoise_robust(noisy_image),
            'Bilateral': lambda: self.denoiser.traditional_denoiser.bilateral_denoise_adaptive(noisy_image),
            'DnCNN': lambda: self.denoiser.deep_learning_denoise(noisy_image),
            'Hybrid_V1': lambda: self.denoiser.hybrid_denoise_v1(noisy_image),
            'Hybrid_V2': lambda: self.denoiser.hybrid_denoise_v2(noisy_image),
            'Traditional_Hybrid': lambda: self.denoiser.wavelet_bilateral_hybrid(noisy_image),  # 新增
            'Hybrid_V3': lambda: self.denoiser.hybrid_denoise_v3(noisy_image),  # 新增
        }
        
        results = {}
        for method_name, method_func in methods.items():
            try:
                results[method_name] = method_func()
            except Exception as e:
                print(f"  {method_name}去噪失败: {e}")
                results[method_name] = noisy_image.copy()
        
        return results
    
    def _calculate_batch_metrics(self, original, noisy, results):
        """计算批量处理指标"""
        noisy_psnr = calculate_psnr(original, noisy)
        noisy_ssim = calculate_ssim(original, noisy)
        
        metrics = {
            'noisy_psnr': noisy_psnr,
            'noisy_ssim': noisy_ssim,
            'methods': {}
        }
        
        psnr_values = []
        for method, result in results.items():
            psnr = calculate_psnr(original, result)
            ssim_val = calculate_ssim(original, result)
            
            metrics['methods'][method] = {
                'psnr': psnr,
                'ssim': ssim_val
            }
            psnr_values.append(psnr)
        
        # 标准化PSNR
        normalized_psnr = normalize_psnr(psnr_values)
        for i, method in enumerate(results.keys()):
            metrics['methods'][method]['normalized_psnr'] = normalized_psnr[i]
        
        return metrics
    
    def _save_batch_images(self, results_data, output_dir):
        """保存批量处理结果"""
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, 'denoising_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 显示统计信息
        print(f"\n{'='*50}")
        print("批量处理完成！")
        print(f"{'='*50}")
        print(f"处理图像数量: {len(results_data)}")
        print(f"结果CSV文件: {csv_path}")
        print(f"图像输出目录: {output_dir}/images/")
        
        # 显示平均PSNR - 更新方法列表
        print("\n各方法平均PSNR:")
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2', 
                  'Traditional_Hybrid', 'Hybrid_V3']  # 新增两个方法
        for method in methods:
            avg_psnr = df[f'{method}_psnr'].mean()
            avg_norm_psnr = df[f'{method}_psnr_norm'].mean()
            print(f"  {method:18}: {avg_psnr:.2f} dB (标准化: {avg_norm_psnr:.3f})")
    
    def _save_batch_images(self, original, noisy, results, image_path, output_dir):
        """保存批量处理图像 - 修正参数"""
        img_output_dir = os.path.join(output_dir, 'images', 
                                    os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(img_output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(img_output_dir, 'original.jpg'), original)
        cv2.imwrite(os.path.join(img_output_dir, 'noisy.jpg'), noisy)
        
        for method, result in results.items():
            cv2.imwrite(os.path.join(img_output_dir, f'{method}.jpg'), result)
    def _save_batch_results(self, results_data, output_dir):
        """保存批量处理结果"""
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(output_dir, 'denoising_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 显示统计信息
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