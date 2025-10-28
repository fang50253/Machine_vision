import torch
import os
import sys
from models.trainer_model import AdvancedDenoisingDataset, ModelTrainer
from models.denoiser_models import ImprovedDnCNN
from utils.device_utils import setup_device, check_pytorch_cuda_support
import cv2

class TrainingController:
    """训练控制器"""
    
    def __init__(self):
        self.device = None
        self.model = None
        self.trainer = None
    
    def setup_environment(self):
        """设置训练环境"""
        print("\n" + "="*60)
        print("图像去噪模型训练系统")
        print("="*60)
        
        check_pytorch_cuda_support()
        self.device = setup_device()
        return self.device
    
    def get_training_parameters(self):
        """获取训练参数"""
        print("\n请设置训练参数:")
        
        try:
            epochs = int(input("训练轮数 (默认50): ").strip() or "50")
            batch_size = int(input("批量大小 (默认8): ").strip() or "8")
            image_size = int(input("图像尺寸 (默认128): ").strip() or "128")
            patience = int(input("早停耐心值 (默认10): ").strip() or "10")
            
            max_samples_input = input("最大图像文件数 (默认全部): ").strip()
            max_samples = int(max_samples_input) if max_samples_input else None
            
            return {
                'epochs': epochs,
                'batch_size': batch_size,
                'image_size': image_size,
                'patience': patience,
                'max_samples': max_samples
            }
        except ValueError:
            print("输入参数错误，使用默认值")
            return {
                'epochs': 50,
                'batch_size': 8,
                'image_size': 128,
                'patience': 10,
                'max_samples': None
            }
    
    def prepare_datasets(self, image_folder, target_size, max_samples=None):
        """准备数据集"""
        print(f"\n正在准备训练数据...")
        print(f"图像文件夹: {image_folder}")
        print(f"目标尺寸: {target_size}x{target_size}")
        print("噪声类型: 高斯噪声 + 椒盐噪声 + 泊松噪声 + 散斑噪声 + 量化噪声")
        
        dataset = AdvancedDenoisingDataset(
            image_folder=image_folder,
            target_size=(target_size, target_size),
            max_samples=max_samples
        )
        
        if len(dataset) == 0:
            raise ValueError("无法创建训练数据！请检查图像文件夹路径。")
        
        # 数据集分割
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        from torch.utils.data import random_split, DataLoader
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 数据加载器
        num_workers = 4 if torch.cuda.is_available() else 0
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8,  # 会在训练时调整
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        
        return train_loader, val_loader
    
    def initialize_model(self, num_layers=17):
        """初始化模型"""
        print(f"\n初始化ImprovedDnCNN模型 ({num_layers}层)...")
        self.model = ImprovedDnCNN(channels=3, num_layers=num_layers, num_features=64)
        self.trainer = ModelTrainer(self.model, self.device, "trained_models")
        return self.model
    
    def start_training(self, train_loader, val_loader, params):
        """开始训练"""
        print("\n开始模型训练...")
        print(f"训练参数: {params}")
        
        train_losses, val_losses, best_val_loss = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=params['epochs'],
            lr=0.001,
            patience=params['patience']
        )
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'model_save_dir': self.trainer.model_save_dir,
            'model_base_name': self.trainer.model_base_name
        }
    
    def display_training_results(self, results):
        """显示训练结果"""
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print(f"最佳验证损失: {results['best_val_loss']:.6f}")
        print(f"模型文件保存在: {results['model_save_dir']}")
        print(f"最佳模型: {results['model_base_name']}_best.pth")
        print(f"最终模型: {results['model_base_name']}_final.pth")
        
        # 绘制训练曲线
        self._plot_training_curve(results['train_losses'], results['val_losses'])
    
    def _plot_training_curve(self, train_losses, val_losses):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join("trained_models", "training_loss_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线图已保存: {plot_path}")
        plt.show()