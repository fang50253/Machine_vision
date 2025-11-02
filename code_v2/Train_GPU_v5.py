import torch
import os
import cv2
from datetime import datetime

# 视图层导入
from views.cli_view import CLIView

# 控制器层导入
from controllers.denoise_controller import DenoiseController
from controllers.batch_controller import BatchController
from controllers.sharpening_controller import SharpeningController
from controllers.training_controller import TrainingController

# 工具函数导入
from utils.image_utils import get_model_path

# 如果需要直接使用模型类
from models.denoiser_models import ImprovedDnCNN
from models.traditional_denoiser import TraditionalDenoiser, AdvancedDenoiser
from models.image_sharpener import ImageSharpener
from models.trainer_model import EarlyStopping, AdvancedDenoisingDataset, ModelTrainer

# 如果需要设备工具
from utils.device_utils import setup_device, check_pytorch_cuda_support

from config import NUM_LAYERS

# 其他可能需要的标准库
import sys
import numpy as np
import matplotlib.pyplot as plt

def model_training():
    """模型训练模式"""
    try:
        controller = TrainingController()
        
        # 设置环境
        controller.setup_environment()
        
        # 获取训练参数
        params = controller.get_training_parameters()
        
        # 获取图像文件夹
        image_folder = input("\n请输入包含训练图像的文件夹路径: ").strip().strip('"\'')
        if not os.path.exists(image_folder):
            print(f"错误：文件夹 '{image_folder}' 不存在！")
            return
        
        # 准备数据
        train_loader, val_loader = controller.prepare_datasets(
            image_folder, 
            params['image_size'], 
            params['max_samples']
        )
        
        # 初始化模型
        controller.initialize_model(NUM_LAYERS)
        
        # 开始训练
        results = controller.start_training(train_loader, val_loader, params)
        
        # 显示结果
        controller.display_training_results(results)
        
    except Exception as e:
        print(f"训练过程中出错: {e}")

if __name__ == "__main__":
    model_training()
    