import os
import cv2
import numpy as np
from config import SUPPORTED_IMAGE_FORMATS, MAX_PIXEL

class ImageView:
    """图像相关视图"""
    
    @staticmethod
    def find_image_files(folder_path):
        """查找文件夹中的图像文件"""
        image_files = []
        print(f"正在搜索文件夹: {folder_path}")
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        if not image_files:
            for file in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file)):
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                        full_path = os.path.join(folder_path, file)
                        image_files.append(full_path)
        
        # 去重和排序
        image_files = list(set(image_files))
        image_files.sort()
        
        print(f"找到 {len(image_files)} 个图像文件:")
        for i, file_path in enumerate(image_files[:5]):
            file_size = os.path.getsize(file_path) // 1024
            print(f"  {i+1}. {os.path.basename(file_path)} ({file_size} KB)")
        if len(image_files) > 5:
            print(f"  ... 还有 {len(image_files) - 5} 个文件")
            
        return image_files
    
    @staticmethod
    def resize_image(image):
        """调整图像尺寸"""
        h, w = image.shape[:2]
        if w > MAX_PIXEL:
            scale = MAX_PIXEL / w
            new_w = MAX_PIXEL
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h))
        return image
    
    @staticmethod
    def show_image_info(image_path, image):
        """显示图像信息"""
        h, w = image.shape[:2]
        file_size = os.path.getsize(image_path) // 1024
        print(f"图像: {os.path.basename(image_path)} | 尺寸: {w}x{h} | 大小: {file_size}KB")