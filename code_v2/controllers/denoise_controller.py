import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from models.traditional_denoiser import AdvancedDenoiser
from utils.image_utils import add_mixed_noise, add_noise_debug, get_noise_settings_interactive
from utils.metrics import calculate_psnr, calculate_ssim, normalize_psnr

class DenoiseController:
    """去噪控制器"""
    
    def __init__(self):
        self.denoiser = None
    
    def initialize_denoiser(self, model_path=None):
        """初始化去噪器"""
        self.denoiser = AdvancedDenoiser(model_path)
        return self.denoiser is not None
    
    def process_single_image(self, image_path, noise_types=None, intensities=None, size_choice='auto'):
        """处理单张图像 - 支持随机噪声"""
        # 如果没有提供噪声设置，使用交互式获取
        if noise_types is None or intensities is None:
            noise_types, intensities = get_noise_settings_interactive()
        
        # 读取和预处理图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 调整尺寸
        processed_image = self._resize_image(original_image, size_choice)
        
        # 添加噪声
        if len(noise_types) == 1:
            noisy_image = add_noise_debug(processed_image, noise_type=noise_types[0], intensity=intensities[0])
        else:
            noisy_image = add_mixed_noise(processed_image, noise_types, intensities)
        
        # 应用各种去噪方法
        results = self._apply_denoising_methods(noisy_image, noise_types, intensities)
        
        # 计算评估指标
        metrics = self._calculate_metrics(processed_image, noisy_image, results)
        
        return {
            'original': processed_image,
            'noisy': noisy_image,
            'results': results,
            'metrics': metrics,
            'noise_types': noise_types,
            'intensities': intensities
        }
    
    def _resize_image(self, image, size_choice):
        """调整图像尺寸"""
        from config import MAX_PIXEL
        h, w = image.shape[:2]
        
        if size_choice == 'auto' and w > MAX_PIXEL:
            scale = MAX_PIXEL / w
            new_w = MAX_PIXEL
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image
    
    def _apply_denoising_methods(self, noisy_image, noise_types, intensities):
        """应用各种去噪方法 - 使用噪声信息"""
        results = {}
        
        print("1/5 进行小波去噪...")
        results['Wavelet'] = self.denoiser.traditional_denoiser.wavelet_denoise_robust(noisy_image)
        
        print("2/5 进行双边滤波...")
        results['Bilateral'] = self.denoiser.traditional_denoiser.bilateral_denoise_advanced(
            noisy_image, noise_types, intensities)
        
        print("3/5 进行深度学习去噪...")
        results['DnCNN'] = self.denoiser.deep_learning_denoise(noisy_image)
        
        print("4/5 进行混合去噪方法1...")
        results['Hybrid V1'] = self.denoiser.hybrid_denoise_v1(noisy_image)
        
        print("5/7 进行混合去噪方法2 (锐化增强)...")
        results['Hybrid_V2'] = self.denoiser.hybrid_denoise_v2_enhanced(
            noisy_image, noise_types, intensities, sharpen_strength=10
        )
        
        # 新增的混合传统方法
        print("6/5 进行传统方法混合去噪...")
        results['Traditional Hybrid'] = self.denoiser.wavelet_bilateral_hybrid(noisy_image)
        
        print("7/5 进行混合去噪方法3...")
        results['Hybrid V3'] = self.denoiser.hybrid_denoise_v3(noisy_image)
        
        return results
    
    def _calculate_metrics(self, original, noisy, results):
        """计算评估指标"""
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
                'ssim': ssim_val,
                'improvement_psnr': psnr - noisy_psnr,
                'improvement_ssim': ssim_val - noisy_ssim if ssim_val > 0 else 0
            }
            psnr_values.append(psnr)
        
        # 标准化PSNR
        normalized_psnr = normalize_psnr(psnr_values)
        for i, method in enumerate(results.keys()):
            metrics['methods'][method]['normalized_psnr'] = normalized_psnr[i]
        
        return metrics
    
    def display_metrics(self, metrics):
        """显示评估指标"""
        print("\n" + "="*50)
        print("去噪效果评估")
        print("="*50)
        
        print(f"噪声图像 - PSNR: {metrics['noisy_psnr']:.2f} dB")
        if metrics['noisy_ssim'] > 0:
            print(f"噪声图像 - SSIM: {metrics['noisy_ssim']:.3f}")
        
        print("\n各方法效果对比:")
        print("-" * 50)
        
        for method, method_metrics in metrics['methods'].items():
            if metrics['noisy_ssim'] > 0 and method_metrics['ssim'] > 0:
                print(f"{method:12}: PSNR: {method_metrics['psnr']:6.2f} dB "
                      f"(+{method_metrics['improvement_psnr']:5.2f}) | "
                      f"SSIM: {method_metrics['ssim']:.3f} "
                      f"(+{method_metrics['improvement_ssim']:.3f})")
            else:
                print(f"{method:12}: PSNR: {method_metrics['psnr']:6.2f} dB "
                      f"(+{method_metrics['improvement_psnr']:5.2f})")
        
        print("\n标准化PSNR值 (0-1范围):")
        for method, method_metrics in metrics['methods'].items():
            print(f"{method:12}: {method_metrics['normalized_psnr']:.3f}")