import cv2
import numpy as np
import os
from models.image_sharpener import ImageSharpener
from utils.metrics import calculate_psnr, calculate_ssim

class SharpeningController:
    """锐化控制器"""
    
    def __init__(self):
        self.sharpener = ImageSharpener()
    
    def process_single_image(self, image_path, sharpening_method='adaptive', **kwargs):
        """处理单张图像锐化"""
        try:
            # 读取图像
            original_image = cv2.imread(image_path)
            if original_image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            print(f"\n开始图像锐化处理...")
            print(f"方法: {sharpening_method}")
            print(f"参数: {kwargs}")
            
            # 应用锐化
            if sharpening_method == 'compare':
                # 比较所有方法
                results = self.sharpener.compare_sharpening_methods(original_image)
            else:
                # 单个方法
                sharpened = self.sharpener.adaptive_sharpen(original_image, sharpening_method, **kwargs)
                results = {sharpening_method: sharpened}
            
            # 计算评估指标
            metrics = {}
            for method, result in results.items():
                # 锐化没有真实标签，只能计算一些无参考指标
                metrics[method] = self._calculate_sharpness_metrics(original_image, result)
            
            return {
                'original': original_image,
                'results': results,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"锐化处理出错: {e}")
            raise
    
    def _calculate_sharpness_metrics(self, original, sharpened):
        """计算锐化相关指标"""
        metrics = {}
        
        # 计算梯度幅值（锐度指标）
        def calculate_gradient(image):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(gradient_x**2 + gradient_y**2)
        
        original_gradient = calculate_gradient(original)
        sharpened_gradient = calculate_gradient(sharpened)
        
        metrics['original_sharpness'] = np.mean(original_gradient)
        metrics['sharpened_sharpness'] = np.mean(sharpened_gradient)
        metrics['sharpness_improvement'] = metrics['sharpened_sharpness'] - metrics['original_sharpness']
        metrics['improvement_ratio'] = metrics['sharpness_improvement'] / metrics['original_sharpness']
        
        return metrics
    
    def display_sharpening_results(self, results):
        """显示锐化结果"""
        print("\n" + "="*50)
        print("锐化效果评估")
        print("="*50)
        
        for method, metrics in results['metrics'].items():
            print(f"\n{method}:")
            print(f"  原始图像锐度: {metrics['original_sharpness']:.3f}")
            print(f"  锐化后锐度: {metrics['sharpened_sharpness']:.3f}")
            print(f"  锐度提升: {metrics['sharpness_improvement']:+.3f}")
            print(f"  提升比例: {metrics['improvement_ratio']:+.2%}")