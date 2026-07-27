import numpy as np
import os
import cv2

def calculate_psnr(original, denoised):
    """计算PSNR"""
    original_float = original.astype(np.float64)
    denoised_float = denoised.astype(np.float64)
    
    mse = np.mean((original_float - denoised_float) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(original, denoised):
    """计算SSIM"""
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
    
    return [(psnr - min_psnr) / (max_psnr - min_psnr) for psnr in psnr_values]