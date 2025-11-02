import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pywt
from typing import Dict, List, Tuple
import torch
import os
import platform
from config import NUM_LAYERS
from models.image_sharpener import ImageSharpener

class TraditionalDenoiser:
    """传统去噪方法 - 完整修复版本"""
    
    def wavelet_denoise_robust(self, image):
        """小波去噪 - 修复形状问题"""
        try:
            if len(image.shape) == 3:
                # 彩色图像 - 分别处理每个通道
                denoised = np.zeros_like(image, dtype=np.float32)
                for i in range(3):
                    channel_denoised = self._wavelet_denoise_channel(image[:,:,i])
                    # 确保形状匹配
                    if channel_denoised.shape == image[:,:,i].shape:
                        denoised[:,:,i] = channel_denoised
                    else:
                        # 如果不匹配，使用中值滤波
                        denoised[:,:,i] = cv2.medianBlur(image[:,:,i], 3)
                return np.clip(denoised, 0, 255).astype(np.uint8)
            else:
                # 灰度图像
                denoised = self._wavelet_denoise_channel(image)
                if denoised.shape == image.shape:
                    return np.clip(denoised, 0, 255).astype(np.uint8)
                else:
                    return cv2.medianBlur(image, 3)
        except Exception as e:
            print(f"小波去噪失败: {e}, 使用中值滤波替代")
            return cv2.medianBlur(image, 5)
    
    def _wavelet_denoise_channel(self, channel):
        """单通道小波去噪"""
        try:
            # 确保输入是2D
            if len(channel.shape) > 2:
                channel = channel.squeeze()
            
            # 使用小波变换
            coeffs = pywt.wavedec2(channel.astype(np.float32), 'db8', level=2)
            
            # 计算阈值
            detail_coeffs = coeffs[1:]
            if detail_coeffs:
                std_dev = np.std([np.std(c) for c in detail_coeffs if hasattr(c, '__iter__')])
                threshold = std_dev * 0.1
            else:
                threshold = 10.0
            
            # 应用软阈值
            new_coeffs = [coeffs[0]]  # 保留近似系数
            for coeff in coeffs[1:]:
                if isinstance(coeff, tuple):
                    # 细节系数
                    coeff_thresh = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeff)
                    new_coeffs.append(coeff_thresh)
                else:
                    new_coeffs.append(coeff)
            
            # 小波重构
            denoised = pywt.waverec2(new_coeffs, 'db8')
            
            # 确保输出形状与输入一致
            if denoised.shape != channel.shape:
                denoised = cv2.resize(denoised, (channel.shape[1], channel.shape[0]))
            
            return denoised
            
        except Exception as e:
            raise Exception(f"单通道小波去噪失败: {e}")
    
    def bilateral_denoise_advanced(self, image, noise_types=None, intensities=None):
        """双边滤波去噪 - 修复方法名"""
        return self.bilateral_denoise_adaptive(image, noise_types, intensities)
    
    def bilateral_denoise_adaptive(self, image, noise_types=None, intensities=None):
        """双边滤波去噪 - 自适应参数"""
        # 根据噪声类型调整参数
        if noise_types and intensities:
            if 'salt_pepper' in noise_types:
                d = 9
                sigma_color = 50
                sigma_space = 50
            else:
                d = 7
                sigma_color = 35
                sigma_space = 35
        else:
            d = 7
            sigma_color = 35
            sigma_space = 35
        
        print(f"自适应双边滤波参数: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
        
        try:
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        except Exception as e:
            print(f"双边滤波失败: {e}, 使用高斯滤波替代")
            return cv2.GaussianBlur(image, (5, 5), 0)
    
    def bilateral_denoise_basic(self, image):
        """基础双边滤波"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def wavelet_bilateral_hybrid(self, image):
        """小波双边混合去噪"""
        try:
            # 先小波去噪
            wavelet_result = self.wavelet_denoise_robust(image)
            # 再双边滤波
            bilateral_result = self.bilateral_denoise_basic(wavelet_result)
            return bilateral_result
        except Exception as e:
            print(f"混合去噪失败: {e}, 使用中值滤波替代")
            return cv2.medianBlur(image, 5)

def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """计算 PSNR (峰值信噪比) - 修复版本"""
    try:
        # 确保图像数据类型和形状一致
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        if original.dtype != processed.dtype:
            processed = processed.astype(original.dtype)
        
        # 转换为 float 进行计算
        original_float = original.astype(np.float64)
        processed_float = processed.astype(np.float64)
        
        # 计算 MSE
        mse = np.mean((original_float - processed_float) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # 计算 PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return max(0.0, psnr)  # 确保非负
    
    except Exception as e:
        print(f"PSNR 计算错误: {e}")
        return 0.0

def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """计算 SSIM (结构相似性指数) - 修复版本"""
    try:
        # 确保图像数据类型和形状一致
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        if original.dtype != processed.dtype:
            processed = processed.astype(original.dtype)
        
        # 确保数据范围正确
        if original.max() > 1.0:
            original = original.astype(np.float64) / 255.0
        if processed.max() > 1.0:
            processed = processed.astype(np.float64) / 255.0
        
        # 对于彩色图像
        if len(original.shape) == 3 and original.shape[2] == 3:
            ssim_values = []
            for i in range(3):
                try:
                    channel_ssim = ssim(
                        original[:, :, i],
                        processed[:, :, i],
                        data_range=1.0,
                        win_size=7,  # 使用固定窗口大小
                        channel_axis=None
                    )
                    ssim_values.append(channel_ssim)
                except:
                    ssim_values.append(0.0)
            
            if ssim_values:
                return float(np.mean(ssim_values))
            else:
                return 0.0
        else:
            # 灰度图像
            try:
                return ssim(
                    original,
                    processed,
                    data_range=1.0,
                    win_size=7,
                    channel_axis=None
                )
            except:
                return 0.0
    
    except Exception as e:
        print(f"SSIM 计算错误: {e}")
        return 0.0

class AdvancedDenoiser:
    """高级去噪器 - 修复版本"""
    

    def __init__(self, model_path: str = None, device: str = "auto"):
        self.device = self._setup_device(device)
        self.model = None
        self.traditional_denoiser = TraditionalDenoiser()
        self.image_sharpener = ImageSharpener()  # 添加锐化器
        
        self._print_device_info()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("ℹ️ 使用传统去噪方法")
    
    def _setup_device(self, device_preference):
        """设置设备"""
        system = platform.system()
        
        if device_preference == "mps" and system == "Darwin":
            return self._get_mps_device()
        elif device_preference == "cuda":
            return self._get_cuda_device()
        elif device_preference == "cpu":
            return torch.device("cpu")
        else:  # auto
            if system == "Darwin":
                return self._get_mps_device()
            else:
                return self._get_cuda_device()
    
    def _get_mps_device(self):
        """获取 MPS 设备"""
        if (hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()):
            try:
                test_tensor = torch.tensor([1.0], device='mps')
                _ = test_tensor * 2
                print("🚀 使用 Apple Silicon GPU (MPS)")
                return torch.device("mps")
            except Exception as e:
                print(f"⚠️ MPS 测试失败: {e}")
        print("⚠️ 使用 CPU")
        return torch.device("cpu")
    
    def _get_cuda_device(self):
        """获取 CUDA 设备"""
        if torch.cuda.is_available():
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                if test_tensor.is_cuda:
                    print("🚀 使用 NVIDIA GPU")
                    return torch.device("cuda")
            except Exception as e:
                print(f"⚠️ CUDA 测试失败: {e}")
        print("⚠️ 使用 CPU")
        return torch.device("cpu")
    
    def _print_device_info(self):
        """打印设备信息"""
        print(f"🎯 当前设备: {self.device}")
    
    def load_model(self, model_path):
        """加载模型"""
        try:
            from models.dncnn import DnCNN
            self.model = DnCNN(channels=3, num_layers=NUM_LAYERS)
            
            print(f"📦 加载模型: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            
            # 清理状态字典
            state_dict = self._clean_state_dict(state_dict)
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model = None
            return False
    
    def _clean_state_dict(self, state_dict):
        """清理状态字典"""
        from collections import OrderedDict
        
        if all(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            return new_state_dict
        
        return state_dict
    
    def deep_learning_denoise(self, image):
        """深度学习去噪"""
        if self.model is None:
            return self.traditional_denoiser.bilateral_denoise_basic(image)
        
        try:
            import time
            start_time = time.time()
            
            # 预处理
            image_tensor = self._preprocess_image(image)
            
            # 推理
            with torch.no_grad():
                noise_pred = self.model(image_tensor)
                output_tensor = image_tensor - noise_pred
            
            # 后处理
            output_image = self._postprocess_output(output_tensor, image.shape)
            
            inference_time = time.time() - start_time
            print(f"⚡ 深度学习推理: {inference_time:.3f}s")
            
            return output_image
            
        except Exception as e:
            print(f"❌ 深度学习失败: {e}")
            return self.traditional_denoiser.bilateral_denoise_basic(image)
    
    def _preprocess_image(self, image):
        """预处理图像"""
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        if len(image.shape) == 3:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0)
        
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor
    
    def _postprocess_output(self, output_tensor, original_shape):
        """后处理输出"""
        output = output_tensor.squeeze(0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        if len(output.shape) == 3:
            output = output.transpose(1, 2, 0)
        
        # 确保输出形状与输入一致
        if output.shape != original_shape:
            output = cv2.resize(output, (original_shape[1], original_shape[0]))
        
        return output
    
    def hybrid_denoise_v1(self, image):
        """混合去噪方法1"""
        dl_result = self.traditional_denoiser.deep_learning_denoise(image)
        return self.wavelet_bilateral_hybrid(dl_result)
    
    
    
    def _initialize_sharpener(self):
        """初始化图像锐化器"""
        try:
            # 从 models 包导入 ImageSharpener
            from models.image_sharpener import ImageSharpener
            print("✅ 图像锐化器初始化成功")
            return ImageSharpener()
        except ImportError as e:
            print(f"⚠️  无法导入 ImageSharpener: {e}")
            print("⚠️  锐化功能将不可用")
            return None
    
    def hybrid_denoise_v2_enhanced(self, image: np.ndarray, noise_types: List[str] = None, 
                                 intensities: List[float] = None, sharpen_strength: int = 10) -> np.ndarray:
        """
        增强的混合去噪方法 V2 - 在 V1 基础上加入锐化处理
        
        参数:
            image: 输入噪声图像
            noise_types: 噪声类型列表
            intensities: 噪声强度列表  
            sharpen_strength: 锐化强度 (1-20)，默认+10
        
        返回:
            去噪并锐化后的图像
        """
        print(f"🔧 开始 V2 混合去噪 (锐化强度: +{sharpen_strength})")
        
    def hybrid_denoise_v2_enhanced(self, image: np.ndarray, noise_types: List[str] = None, 
                             intensities: List[float] = None, sharpen_strength: int = 10) -> np.ndarray:
        """
        增强的混合去噪方法 V2 - 先锐化再DnCNN
        
        参数:
            image: 输入噪声图像
            noise_types: 噪声类型列表
            intensities: 噪声强度列表  
            sharpen_strength: 锐化强度 (1-20)，默认+10
        
        返回:
            去噪并锐化后的图像
        """
        print(f"🔧 开始 V2 混合去噪 (先锐化再DnCNN, 锐化强度: +{sharpen_strength})")
    
        try:
            # 步骤1: 先对噪声图像进行锐化预处理
            sharpened_input = image
            if self.image_sharpener is not None:
                print("1/4 输入图像锐化预处理...")
                sharpened_input = self._apply_sharpening(image, sharpen_strength)
            else:
                print("1/4 跳过输入锐化 (锐化器不可用)")
            
            # 步骤2: 深度学习去噪（对锐化后的图像）
            print("2/4 深度学习去噪...")
            dl_denoised = self.deep_learning_denoise(sharpened_input)
            
            # 步骤3: 传统方法优化细节
            print("3/4 传统方法优化...")
            traditional_refined = self.traditional_denoiser.wavelet_bilateral_hybrid(dl_denoised)
            
            # 步骤4: 最终质量优化
            print("4/4 最终质量优化...")
            final_result = self._post_processing_optimization(traditional_refined, image)
            
            print("✅ V2 混合去噪完成 (先锐化策略)")
            return final_result
            
        except Exception as e:
            print(f"❌ V2 混合去噪失败: {e}, 使用基础方法")
            return self.hybrid_denoise_v1(image)
    
    def _apply_sharpening(self, image: np.ndarray, strength: int) -> np.ndarray:
        """
        应用锐化处理
        
        参数:
            image: 输入图像
            strength: 锐化强度 (1-20)
        
        返回:
            锐化后的图像
        """
        try:
            # 标准化强度值
            strength = max(1, min(20, strength))
            
            # 根据强度选择锐化策略
            if strength <= 5:
                # 轻度锐化 - 保持自然感
                return self.image_sharpener.adaptive_sharpen(
                    image, 
                    method='unsharp',
                    strength=0.8 + strength * 0.1,  # 0.9 - 1.3
                    sigma=1.2,
                    threshold=8
                )
            elif strength <= 10:
                # 中度锐化 - 平衡增强
                result = self.image_sharpener.adaptive_sharpen(
                    image,
                    method='unsharp', 
                    strength=1.3 + (strength - 5) * 0.14,  # 1.3 - 2.0
                    sigma=1.0,
                    threshold=5
                )
                return result
            elif strength <= 15:
                # 强度锐化 - 显著增强
                # 第一轮: 非锐化掩蔽
                sharpened = self.image_sharpener.unsharp_mask(
                    image,
                    strength=2.0 + (strength - 10) * 0.2,  # 2.0 - 3.0
                    sigma=0.8,
                    threshold=3
                )
                # 第二轮: 拉普拉斯增强边缘
                return self.image_sharpener.laplacian_sharpen(sharpened, strength=0.15)
            else:
                # 超强锐化 - 多重处理
                # 第一轮: 强非锐化掩蔽
                sharpened = self.image_sharpener.unsharp_mask(
                    image,
                    strength=3.0,
                    sigma=0.6, 
                    threshold=1
                )
                # 第二轮: 拉普拉斯锐化
                sharpened = self.image_sharpener.laplacian_sharpen(sharpened, strength=0.25)
                # 第三轮: 引导滤波锐化
                return self.image_sharpener.guided_sharpen(sharpened, strength=1.2)
                
        except Exception as e:
            print(f"⚠️  锐化处理失败: {e}, 返回未锐化图像")
            return image
    
    def _post_processing_optimization(self, processed_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        后处理优化 - 确保最终质量
        
        参数:
            processed_image: 处理后的图像
            original_image: 原始噪声图像
        
        返回:
            优化后的图像
        """
        try:
            # 检查图像质量指标
            current_psnr = calculate_psnr(original_image, processed_image)
            
            # 如果质量较差，应用轻度降噪
            if current_psnr < 25:  # PSNR 阈值
                print("🔄 检测到质量较低，应用轻度优化...")
                optimized = cv2.bilateralFilter(processed_image, 5, 25, 25)
                return optimized
            
            return processed_image
            
        except Exception as e:
            print(f"⚠️  后处理优化失败: {e}")
            return processed_image
    
    def hybrid_denoise_v2(self, image: np.ndarray, noise_types: List[str] = None, 
                         intensities: List[float] = None) -> np.ndarray:
        """
        保持向后兼容的 V2 方法 - 默认使用+10锐化强度
        
        参数:
            image: 输入图像
            noise_types: 噪声类型列表
            intensities: 噪声强度列表
            
        返回:
            去噪后的图像
        """
        return self.hybrid_denoise_v2_enhanced(
            image, 
            noise_types, 
            intensities, 
            sharpen_strength=10  # 默认+10锐化
        )
    
    def compare_sharpening_effects(self, image: np.ndarray, noise_types: List[str] = None,
                                 intensities: List[float] = None) -> Dict[str, np.ndarray]:
        """
        比较不同锐化强度的效果
        
        参数:
            image: 输入图像
            noise_types: 噪声类型列表
            intensities: 噪声强度列表
            
        返回:
            不同锐化强度的结果字典
        """
        results = {}
        
        print("\n🔍 比较不同锐化强度效果:")
        print("-" * 40)
        
        # 测试不同锐化强度
        sharpen_strengths = [5, 10, 15, 20]
        
        for strength in sharpen_strengths:
            print(f"测试锐化强度 +{strength}...")
            result = self.hybrid_denoise_v2_enhanced(
                image, noise_types, intensities, sharpen_strength=strength
            )
            results[f'V2_Sharpness_{strength}'] = result
        
        # 包含无锐化版本作为对比
        print("测试无锐化版本...")
        no_sharpen_result = self.hybrid_denoise_v1(image)
        results['V1_No_Sharpening'] = no_sharpen_result
        
        return results
    

    def hybrid_denoise_v3(self, image):
        """混合去噪方法3"""
        dl_result = self.deep_learning_denoise(image)
        wavelet_result = self.traditional_denoiser.wavelet_denoise_robust(image)
        bilateral_result = self.traditional_denoiser.bilateral_denoise_basic(image)
        
        # 加权融合
        fused = cv2.addWeighted(dl_result, 0.5, wavelet_result, 0.3, 0)
        fused = cv2.addWeighted(fused, 0.7, bilateral_result, 0.3, 0)
        
        return fused
    
    def wavelet_bilateral_hybrid(self, image):
        """小波双边混合"""
        return self.traditional_denoiser.wavelet_bilateral_hybrid(image)

# 使用示例
if __name__ == "__main__":
    # 测试修复后的代码
    denoiser = AdvancedDenoiser()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 测试各种方法
    methods = {
        'Wavelet': denoiser.traditional_denoiser.wavelet_denoise_robust,
        'Bilateral': denoiser.traditional_denoiser.bilateral_denoise_adaptive,
        'DnCNN': denoiser.deep_learning_denoise,
        'Hybrid_V1': denoiser.hybrid_denoise_v1,
        'Hybrid_V2': denoiser.hybrid_denoise_v2,
        'Hybrid_V3': denoiser.hybrid_denoise_v3,
    }
    
    for name, method in methods.items():
        try:
            result = method(test_image)
            psnr = calculate_psnr(test_image, result)
            ssim_val = calculate_ssim(test_image, result)
            print(f"{name}: PSNR={psnr:.2f}dB, SSIM={ssim_val:.4f}")
        except Exception as e:
            print(f"{name} 失败: {e}")

