import cv2
import numpy as np
import torch
import os
from .denoiser_models import ImprovedDnCNN
from config import DEVICE, NUM_LAYERS
from .image_sharpener import ImageSharpener as IS

class TraditionalDenoiser:
    """传统图像去噪方法"""
    
    @staticmethod
    def wavelet_denoise(image, wavelet='db4', level=2, threshold=0.1):
        """小波变换去噪 - 修复尺寸问题"""
        try:
            import pywt
            
            # 确保图像是float类型
            image_float = image.astype(np.float32) / 255.0
            
            def wavelet_denoise_single_channel(channel):
                """处理单通道图像"""
                # 获取原始尺寸
                original_shape = channel.shape
                
                # 计算适合小波变换的尺寸（2的幂次方）
                def get_wavelet_shape(shape):
                    return tuple(2 ** int(np.ceil(np.log2(dim))) for dim in shape)
                
                target_shape = get_wavelet_shape(original_shape)
                
                # 调整图像尺寸以适应小波变换
                if original_shape != target_shape:
                    channel_resized = cv2.resize(channel, target_shape[::-1])  # OpenCV使用(width, height)
                else:
                    channel_resized = channel
                
                # 进行小波变换
                coeffs = pywt.wavedec2(channel_resized, wavelet, level=level)
                
                # 阈值处理
                coeffs_thresh = [coeffs[0]]  # 近似系数保持不变
                for level_coeff in coeffs[1:]:
                    thresh_coeff = []
                    for detail in level_coeff:
                        # 使用软阈值
                        thresh_detail = pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft')
                        thresh_coeff.append(thresh_detail)
                    coeffs_thresh.append(tuple(thresh_coeff))
                
                # 小波重构
                denoised_resized = pywt.waverec2(coeffs_thresh, wavelet)
                
                # 调整回原始尺寸
                if original_shape != target_shape:
                    denoised_channel = cv2.resize(denoised_resized, original_shape[::-1])
                else:
                    denoised_channel = denoised_resized
                
                return denoised_channel
            
            # 处理多通道或单通道图像
            if len(image.shape) == 3:
                denoised = np.zeros_like(image_float)
                for i in range(3):
                    denoised[:,:,i] = wavelet_denoise_single_channel(image_float[:,:,i])
            else:
                denoised = wavelet_denoise_single_channel(image_float)
            
            # 确保数值范围正确
            denoised = np.clip(denoised, 0, 1)
            return (denoised * 255).astype(np.uint8)
            
        except ImportError:
            print("PyWavelets未安装，使用中值滤波代替")
            return cv2.medianBlur(image, 5)
        except Exception as e:
            print(f"小波去噪出错: {e}，使用中值滤波代替")
            return cv2.medianBlur(image, 5)
    
    @staticmethod
    def wavelet_denoise_simple(image, wavelet='db4', level=2, threshold=0.1):
        """简化版小波去噪 - 自动调整level避免尺寸问题"""
        try:
            import pywt
            
            image_float = image.astype(np.float32) / 255.0
            
            def get_max_level(shape, wavelet):
                """计算最大可用的小波分解层数"""
                max_level = pywt.dwt_max_level(min(shape), pywt.Wavelet(wavelet).dec_len)
                return min(level, max_level) if max_level else 1
            
            if len(image.shape) == 3:
                denoised = np.zeros_like(image_float)
                for i in range(3):
                    # 为每个通道计算合适的level
                    channel_level = get_max_level(image_float[:,:,i].shape, wavelet)
                    coeffs = pywt.wavedec2(image_float[:,:,i], wavelet, level=channel_level)
                    
                    coeffs_thresh = [coeffs[0]]
                    for level_coeff in coeffs[1:]:
                        thresh_coeff = [pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft') 
                                      for detail in level_coeff]
                        coeffs_thresh.append(tuple(thresh_coeff))
                    
                    denoised[:,:,i] = pywt.waverec2(coeffs_thresh, wavelet)
            else:
                # 为单通道图像计算合适的level
                actual_level = get_max_level(image_float.shape, wavelet)
                coeffs = pywt.wavedec2(image_float, wavelet, level=actual_level)
                
                coeffs_thresh = [coeffs[0]]
                for level_coeff in coeffs[1:]:
                    thresh_coeff = [pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft') 
                                  for detail in level_coeff]
                    coeffs_thresh.append(tuple(thresh_coeff))
                
                denoised = pywt.waverec2(coeffs_thresh, wavelet)
            
            denoised = np.clip(denoised, 0, 1)
            return (denoised * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"小波去噪出错: {e}，使用中值滤波代替")
            return cv2.medianBlur(image, 5)
        
    @staticmethod
    def wavelet_denoise_robust(image, wavelet='db4', level=2, threshold=0.1):
        """鲁棒版小波去噪 - 处理各种尺寸问题"""
        try:
            import pywt
            
            # 转换为float
            image_float = image.astype(np.float32) / 255.0
            
            def wavelet_denoise_channel(channel):
                """处理单通道的小波去噪"""
                try:
                    # 尝试原始尺寸
                    coeffs = pywt.wavedec2(channel, wavelet, level=level)
                except ValueError:
                    try:
                        # 如果失败，减少level
                        reduced_level = level - 1
                        while reduced_level > 0:
                            try:
                                coeffs = pywt.wavedec2(channel, wavelet, level=reduced_level)
                                break
                            except ValueError:
                                reduced_level -= 1
                        if reduced_level == 0:
                            # 如果所有level都失败，返回原图
                            return channel
                    except:
                        return channel
                
                # 阈值处理
                coeffs_thresh = [coeffs[0]]
                for level_coeff in coeffs[1:]:
                    thresh_coeff = [pywt.threshold(detail, threshold * np.max(np.abs(detail)), 'soft') 
                                  for detail in level_coeff]
                    coeffs_thresh.append(tuple(thresh_coeff))
                
                # 重构
                return pywt.waverec2(coeffs_thresh, wavelet)
            
            # 处理图像
            if len(image.shape) == 3:
                denoised = np.zeros_like(image_float)
                for i in range(3):
                    denoised_channel = wavelet_denoise_channel(image_float[:,:,i])
                    # 确保尺寸匹配
                    if denoised_channel.shape == image_float[:,:,i].shape:
                        denoised[:,:,i] = denoised_channel
                    else:
                        # 如果不匹配，调整尺寸
                        denoised[:,:,i] = cv2.resize(denoised_channel, image_float[:,:,i].shape[::-1])
            else:
                denoised = wavelet_denoise_channel(image_float)
                if denoised.shape != image_float.shape:
                    denoised = cv2.resize(denoised, image_float.shape[::-1])
            
            denoised = np.clip(denoised, 0, 1)
            return (denoised * 255).astype(np.uint8)
            
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
    

    @staticmethod
    def bilateral_denoise(image, d=None, sigma_color=None, sigma_space=None):
        """改进的双边滤波 - 自适应参数"""
        try:
            # 调试信息
            print(f"双边滤波输入: 形状={image.shape}, 类型={image.dtype}, 范围=[{image.min()}, {image.max()}]")
            
            # 确保图像类型正确
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # 自适应参数设置
            if d is None:
                # 根据图像尺寸设置d
                h, w = image.shape[:2]
                d = min(9, min(h, w) // 10)  # 动态调整邻域直径
            
            if sigma_color is None:
                # 根据噪声强度设置sigma_color
                sigma_color = 50  # 降低颜色空间sigma
            
            if sigma_space is None:
                # 根据图像尺寸设置sigma_space
                sigma_space = 50  # 降低坐标空间sigma
            
            print(f"双边滤波参数: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
            
            # 处理单通道图像
            if len(image.shape) == 2:
                denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            else:
                # 处理多通道图像 - 分别处理每个通道
                denoised_channels = []
                for i in range(image.shape[2]):
                    channel_denoised = cv2.bilateralFilter(image[:,:,i], d, sigma_color, sigma_space)
                    denoised_channels.append(channel_denoised)
                denoised = np.stack(denoised_channels, axis=2)
            
            print(f"双边滤波输出: 形状={denoised.shape}, 范围=[{denoised.min()}, {denoised.max()}]")
            
            return denoised
            
        except Exception as e:
            print(f"双边滤波出错: {e}，使用高斯滤波代替")
            return cv2.GaussianBlur(image, (5, 5), 1.0)
    
    @staticmethod
    def bilateral_denoise_adaptive(image, noise_intensity=25):
        """自适应双边滤波 - 根据噪声强度调整参数"""
        try:
            # 根据噪声强度调整参数
            if noise_intensity < 20:
                d = 5
                sigma_color = 25
                sigma_space = 25
            elif noise_intensity < 40:
                d = 7
                sigma_color = 35
                sigma_space = 35
            elif noise_intensity < 60:
                d = 9
                sigma_color = 50
                sigma_space = 50
            else:
                d = 11
                sigma_color = 75
                sigma_space = 75
            
            print(f"自适应双边滤波参数: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
            
            if len(image.shape) == 2:
                return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            else:
                denoised_channels = []
                for i in range(image.shape[2]):
                    channel_denoised = cv2.bilateralFilter(image[:,:,i], d, sigma_color, sigma_space)
                    denoised_channels.append(channel_denoised)
                return np.stack(denoised_channels, axis=2)
                
        except Exception as e:
            print(f"自适应双边滤波出错: {e}")
            return TraditionalDenoiser.bilateral_denoise(image)

    @staticmethod
    def hybrid_traditional_denoise(image, method1='wavelet', method2='bilateral', 
                                 method1_params=None, method2_params=None, 
                                 weight1=0.5, weight2=0.5):
        """
        混合两种传统去噪方法
        Args:
            image: 输入图像
            method1: 第一种方法 ('wavelet', 'bilateral', 'median', 'gaussian')
            method2: 第二种方法
            method1_params: 第一种方法的参数
            method2_params: 第二种方法的参数
            weight1: 第一种方法的权重
            weight2: 第二种方法的权重
        """
        try:
            print(f"混合传统去噪: {method1}(权重{weight1}) + {method2}(权重{weight2})")
            
            # 默认参数
            if method1_params is None:
                method1_params = {}
            if method2_params is None:
                method2_params = {}
            
            # 应用第一种方法
            if method1 == 'wavelet':
                denoised1 = TraditionalDenoiser.wavelet_denoise_robust(image, **method1_params)
            elif method1 == 'bilateral':
                denoised1 = TraditionalDenoiser.bilateral_denoise_adaptive(image, **method1_params)
            elif method1 == 'median':
                kernel_size = method1_params.get('kernel_size', 5)
                denoised1 = TraditionalDenoiser.median_denoise(image, kernel_size)
            elif method1 == 'gaussian':
                kernel_size = method1_params.get('kernel_size', 5)
                sigma = method1_params.get('sigma', 1.0)
                denoised1 = TraditionalDenoiser.gaussian_denoise(image, kernel_size, sigma)
            else:
                denoised1 = image
            
            # 应用第二种方法
            if method2 == 'wavelet':
                denoised2 = TraditionalDenoiser.wavelet_denoise_robust(image, **method2_params)
            elif method2 == 'bilateral':
                denoised2 = TraditionalDenoiser.bilateral_denoise_adaptive(image, **method2_params)
            elif method2 == 'median':
                kernel_size = method2_params.get('kernel_size', 5)
                denoised2 = TraditionalDenoiser.median_denoise(image, kernel_size)
            elif method2 == 'gaussian':
                kernel_size = method2_params.get('kernel_size', 5)
                sigma = method2_params.get('sigma', 1.0)
                denoised2 = TraditionalDenoiser.gaussian_denoise(image, kernel_size, sigma)
            else:
                denoised2 = image
            
            # 加权融合
            result = cv2.addWeighted(denoised1, weight1, denoised2, weight2, 0)
            
            print(f"混合传统去噪完成")
            return result
            
        except Exception as e:
            print(f"混合传统去噪出错: {e}，使用第一种方法代替")
            # 出错时返回第一种方法的结果
            if method1 == 'wavelet':
                return TraditionalDenoiser.wavelet_denoise_robust(image)
            else:
                return TraditionalDenoiser.bilateral_denoise_adaptive(image)
    
    @staticmethod
    def wavelet_bilateral_hybrid(image, wavelet_strength=0.7, bilateral_strength=0.3):
        """小波+双边滤波混合去噪"""
        wavelet_denoised = TraditionalDenoiser.wavelet_denoise_robust(image)
        bilateral_denoised = TraditionalDenoiser.bilateral_denoise_adaptive(image)
        return cv2.addWeighted(wavelet_denoised, wavelet_strength, 
                             bilateral_denoised, bilateral_strength, 0)
    
    @staticmethod
    def median_wavelet_hybrid(image, median_kernel=3, wavelet_strength=0.6):
        """中值滤波+小波混合去噪 - 特别适合椒盐噪声"""
        median_denoised = TraditionalDenoiser.median_denoise(image, median_kernel)
        wavelet_denoised = TraditionalDenoiser.wavelet_denoise_robust(image)
        return cv2.addWeighted(median_denoised, 0.4, 
                             wavelet_denoised, wavelet_strength, 0)
        

class AdvancedDenoiser:
    def __init__(self, model_path=None):
        self.traditional_denoiser = TraditionalDenoiser()
        self.model = None
        self.device = DEVICE
        self.load_pretrained_model(model_path)
    
    def load_pretrained_model(self, model_path=None):
        # 使用从config导入的NUM_LAYERS
        self.model = ImprovedDnCNN(channels=3, num_layers=NUM_LAYERS, num_features=64)
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✅ 预训练模型加载成功: {os.path.basename(model_path)}")
            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                print("使用随机初始化的模型")
        else:
            print("❌ 未提供模型路径或文件不存在，使用随机初始化的模型")
    
    def preprocess_image(self, image):
        """预处理图像"""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image_tensor.unsqueeze(0).to(self.device)
    
    def postprocess_image(self, tensor):
        """后处理图像"""
        image = tensor.squeeze(0).cpu().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)
    
    def deep_learning_denoise(self, image):
        if self.model is None:
            return image
        
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output_tensor = self.model(input_tensor)
            return self.postprocess_image(output_tensor)
    
    def hybrid_denoise_v1(self, image):
        # traditional_denoised = self.traditional_denoiser.bilateral_denoise(image)
        sharpen_image=IS.adaptive_sharpen(image)
        return self.deep_learning_denoise(sharpen_image)
    
    def hybrid_denoise_v2(self, image):
        # 使用鲁棒版小波去噪
        wavelet_denoised = self.traditional_denoiser.wavelet_denoise_robust(image)
        bilateral_denoised = self.traditional_denoiser.bilateral_denoise(image)
        dl_denoised = self.deep_learning_denoise(image)
        
        alpha = 0.3
        beta = 0.7
        traditional_avg = cv2.addWeighted(wavelet_denoised, 0.5, bilateral_denoised, 0.5, 0)
        return cv2.addWeighted(traditional_avg, alpha, dl_denoised, beta, 0)
    def hybrid_traditional_denoise(self, image, method1='wavelet', method2='bilateral', 
                                 method1_params=None, method2_params=None, 
                                 weight1=0.5, weight2=0.5):
        """高级混合传统去噪"""
        return self.traditional_denoiser.hybrid_traditional_denoise(
            image, method1, method2, method1_params, method2_params, weight1, weight2)
    
    def wavelet_bilateral_hybrid(self, image):
        """小波+双边滤波混合"""
        return self.traditional_denoiser.wavelet_bilateral_hybrid(image)
    
    def median_wavelet_hybrid(self, image):
        """中值滤波+小波混合"""
        return self.traditional_denoiser.median_wavelet_hybrid(image)
    
    def hybrid_denoise_v3(self, image):
        """混合去噪方法3 - 传统方法混合 + 深度学习"""
        # 先用传统方法混合去噪
        traditional_hybrid = self.traditional_denoiser.wavelet_bilateral_hybrid(image)
        # 再用深度学习进一步去噪
        return self.deep_learning_denoise(traditional_hybrid)