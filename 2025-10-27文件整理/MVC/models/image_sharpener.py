import os
import cv2
import numpy as np
import torch
import torch.nn as nn

class ImageSharpener:
    """图像锐化处理类"""
    
    @staticmethod
    def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=1.5, threshold=0):
        """
        非锐化掩蔽锐化
        Args:
            image: 输入图像
            kernel_size: 高斯核大小
            sigma: 高斯核标准差
            strength: 锐化强度
            threshold: 锐化阈值
        """
        try:
            # 高斯模糊
            blurred = cv2.GaussianBlur(image, kernel_size, sigma)
            
            # 计算细节层
            detail = image.astype(np.float32) - blurred.astype(np.float32)
            
            # 应用阈值
            if threshold > 0:
                mask = np.abs(detail) > threshold
                detail = detail * mask
            
            # 应用锐化
            sharpened = image.astype(np.float32) + strength * detail
            
            # 限制到有效范围
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            print(f"非锐化掩蔽: 强度={strength}, 阈值={threshold}")
            return sharpened
            
        except Exception as e:
            print(f"非锐化掩蔽出错: {e}")
            return image
    
    @staticmethod
    def laplacian_sharpen(image, strength=0.5):
        """
        拉普拉斯锐化
        Args:
            image: 输入图像
            strength: 锐化强度
        """
        try:
            # 拉普拉斯算子
            kernel = np.array([[0, -1, 0],
                              [-1, 4, -1],
                              [0, -1, 0]], dtype=np.float32)
            
            # 应用拉普拉斯滤波
            laplacian = cv2.filter2D(image.astype(np.float32), -1, kernel)
            
            # 锐化
            sharpened = image.astype(np.float32) - strength * laplacian
            
            # 限制到有效范围
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            print(f"拉普拉斯锐化: 强度={strength}")
            return sharpened
            
        except Exception as e:
            print(f"拉普拉斯锐化出错: {e}")
            return image
    
    @staticmethod
    def high_pass_filter(image, cutoff_frequency=30):
        """
        高通滤波锐化
        Args:
            image: 输入图像
            cutoff_frequency: 截止频率
        """
        try:
            # 转换为float32
            image_float = image.astype(np.float32) / 255.0
            
            # 傅里叶变换
            dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # 创建高通滤波器
            rows, cols = image.shape[:2]
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols, 2), np.float32)
            r = cutoff_frequency
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 0
            
            # 应用滤波器
            fshift = dft_shift * mask
            
            # 逆傅里叶变换
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            
            # 归一化并锐化
            img_back = cv2.normalize(img_back, None, 0, 1, cv2.NORM_MINMAX)
            sharpened = image_float + 0.5 * img_back
            
            # 限制到有效范围
            sharpened = np.clip(sharpened, 0, 1)
            sharpened = (sharpened * 255).astype(np.uint8)
            
            print(f"高通滤波锐化: 截止频率={cutoff_frequency}")
            return sharpened
            
        except Exception as e:
            print(f"高通滤波锐化出错: {e}")
            return image
    
    @staticmethod
    def guided_sharpen(image, radius=2, eps=0.01, strength=1.0):
        """
        引导滤波锐化
        Args:
            image: 输入图像
            radius: 引导滤波半径
            eps: 正则化参数
            strength: 锐化强度
        """
        try:
            # 使用引导滤波获取基础层
            if hasattr(cv2, 'ximgproc'):
                base_layer = cv2.ximgproc.guidedFilter(
                    guide=image, 
                    src=image, 
                    radius=radius, 
                    eps=eps
                )
            else:
                # 如果OpenCV没有ximgproc，使用高斯模糊代替
                base_layer = cv2.GaussianBlur(image, (5, 5), 1.0)
            
            # 计算细节层
            detail_layer = image.astype(np.float32) - base_layer.astype(np.float32)
            
            # 锐化
            sharpened = image.astype(np.float32) + strength * detail_layer
            
            # 限制到有效范围
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            print(f"引导滤波锐化: 强度={strength}, 半径={radius}")
            return sharpened
            
        except Exception as e:
            print(f"引导滤波锐化出错: {e}")
            return image
    
    @staticmethod
    def deep_learning_sharpen(image, model_path=None):
        """
        深度学习锐化
        Args:
            image: 输入图像
            model_path: 模型路径
        """
        try:
            # 这里可以集成深度学习锐化模型
            # 目前使用传统方法作为placeholder
            print("深度学习锐化: 使用非锐化掩蔽作为临时实现")
            return ImageSharpener.unsharp_mask(image, strength=1.2)
            
        except Exception as e:
            print(f"深度学习锐化出错: {e}")
            return image
    
    @staticmethod
    def adaptive_sharpen(image, method='unsharp', **kwargs):
        """
        自适应锐化 - 根据图像特性选择最佳方法
        Args:
            image: 输入图像
            method: 锐化方法 ('unsharp', 'laplacian', 'highpass', 'guided')
            **kwargs: 各方法的参数
        """
        # 计算图像梯度（评估图像细节程度）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        detail_level = np.mean(gradient_magnitude)
        
        print(f"图像细节水平: {detail_level:.2f}")
        
        # 根据细节水平调整参数
        if detail_level < 10:
            # 低细节图像 - 使用较强锐化
            default_kwargs = {'strength': 1.8, 'sigma': 1.5}
        elif detail_level > 50:
            # 高细节图像 - 使用较弱锐化
            default_kwargs = {'strength': 1.0, 'sigma': 0.8}
        else:
            # 中等细节图像
            default_kwargs = {'strength': 1.3, 'sigma': 1.0}
        
        # 合并参数
        sharpening_kwargs = {**default_kwargs, **kwargs}
        
        # 选择锐化方法
        if method == 'laplacian':
            return ImageSharpener.laplacian_sharpen(image, **sharpening_kwargs)
        elif method == 'highpass':
            return ImageSharpener.high_pass_filter(image, **sharpening_kwargs)
        elif method == 'guided':
            return ImageSharpener.guided_sharpen(image, **sharpening_kwargs)
        else:  # 默认使用非锐化掩蔽
            return ImageSharpener.unsharp_mask(image, **sharpening_kwargs)
    
    @staticmethod
    def compare_sharpening_methods(image):
        """
        比较不同锐化方法的效果
        Returns:
            dict: 各种锐化方法的结果
        """
        results = {}
        
        print("\n比较锐化方法:")
        print("-" * 30)
        
        # 非锐化掩蔽
        print("1. 非锐化掩蔽...")
        results['Unsharp_Mask'] = ImageSharpener.unsharp_mask(image, strength=1.5)
        
        # 拉普拉斯锐化
        print("2. 拉普拉斯锐化...")
        results['Laplacian'] = ImageSharpener.laplacian_sharpen(image, strength=0.3)
        
        # 引导滤波锐化
        print("3. 引导滤波锐化...")
        results['Guided_Filter'] = ImageSharpener.guided_sharpen(image, strength=1.2)
        
        # 自适应锐化
        print("4. 自适应锐化...")
        results['Adaptive'] = ImageSharpener.adaptive_sharpen(image)
        
        return results