import torch
import os
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 视图层导入
from views.cli_view import CLIView

# 控制器层导入
from controllers.denoise_controller import DenoiseController
from controllers.batch_controller import BatchController
from controllers.sharpening_controller import SharpeningController

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

class TrainingController:
    """训练控制器 - 添加损失曲线显示"""
    
    def __init__(self):
        self.model = None
        self.trainer = None
        self.device = setup_device()
        self.train_losses = []
        self.val_losses = []
    
    def setup_environment(self):
        """设置训练环境"""
        print("🚀 初始化训练环境...")
        check_pytorch_cuda_support()
        print(f"使用设备: {self.device}")
    
    def get_training_parameters(self):
        """获取训练参数"""
        print("\n📋 训练参数设置:")
        
        params = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'patience': 10,
            'image_size': (256, 256),
            'max_samples': 5000
        }
        
        try:
            params['epochs'] = int(input(f"训练轮数 (默认 {params['epochs']}): ") or params['epochs'])
            params['batch_size'] = int(input(f"批次大小 (默认 {params['batch_size']}): ") or params['batch_size'])
            params['learning_rate'] = float(input(f"学习率 (默认 {params['learning_rate']}): ") or params['learning_rate'])
            params['patience'] = int(input(f"早停耐心值 (默认 {params['patience']}): ") or params['patience'])
            
            size_input = input(f"图像尺寸 (默认 {params['image_size'][0]}x{params['image_size'][1]}): ") or f"{params['image_size'][0]}x{params['image_size'][1]}"
            if 'x' in size_input:
                w, h = map(int, size_input.split('x'))
                params['image_size'] = (h, w)  # OpenCV 使用 (height, width)
            
            params['max_samples'] = int(input(f"最大样本数 (默认 {params['max_samples']}): ") or params['max_samples'])
            
        except ValueError as e:
            print(f"参数输入错误，使用默认值: {e}")
        
        print("\n✅ 训练参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        return params
    
    def prepare_datasets(self, image_folder, image_size, max_samples):
        """准备数据集"""
        print(f"\n📊 准备数据集...")
        print(f"图像文件夹: {image_folder}")
        print(f"图像尺寸: {image_size}")
        print(f"最大样本数: {max_samples}")
        
        # 创建数据集
        dataset = AdvancedDenoisingDataset(
            image_folder=image_folder,
            target_size=image_size,
            max_samples=max_samples
        )
        
        # 分割训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=0
        )
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        
        return train_loader, val_loader
    
    def initialize_model(self, num_layers=17):
        """初始化模型"""
        print(f"\n初始化ImprovedDnCNN模型 ({num_layers}层)...")
        self.model = ImprovedDnCNN(channels=3, num_layers=num_layers, num_features=64)
        self.trainer = ModelTrainer(self.model, "trained_models")
        return self.model
    
    def start_training(self, train_loader, val_loader, params):
        """开始训练"""
        print(f"\n🎯 开始训练...")
        print(f"总轮数: {params['epochs']}")
        print(f"批次大小: {params['batch_size']}")
        print(f"学习率: {params['learning_rate']}")
        
        # 开始训练并获取损失历史
        self.train_losses, self.val_losses, best_val_loss = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=params['epochs'],
            lr=params['learning_rate'],
            patience=params['patience']
        )
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.train_losses)
        }
    
    def display_training_results(self, results):
        """显示训练结果和损失曲线"""
        print(f"\n📈 训练完成!")
        print(f"训练轮数: {results['epochs_trained']}")
        print(f"最佳验证损失: {results['best_val_loss']:.6f}")
        
        # 显示损失曲线
        self._plot_loss_curves(results)
        
        # 显示训练统计
        self._display_training_stats(results)
    
    def _plot_loss_curves(self, results):
        """绘制损失曲线"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 绘制训练和验证损失
            epochs = range(1, len(results['train_losses']) + 1)
            
            plt.subplot(2, 1, 1)
            plt.plot(epochs, results['train_losses'], 'b-', label='训练损失', linewidth=2)
            plt.plot(epochs, results['val_losses'], 'r-', label='验证损失', linewidth=2)
            plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
            plt.xlabel('轮次')
            plt.ylabel('损失 (MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制对数尺度损失
            plt.subplot(2, 1, 2)
            plt.semilogy(epochs, results['train_losses'], 'b-', label='train loss', linewidth=2, alpha=0.7)
            plt.semilogy(epochs, results['val_losses'], 'r-', label='valid loss', linewidth=2, alpha=0.7)
            plt.title('对数尺度损失曲线', fontsize=14, fontweight='bold')
            plt.xlabel('轮次')
            plt.ylabel('损失 (log MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # 保存损失曲线
            self._save_loss_plot(results)
            
        except Exception as e:
            print(f"绘制损失曲线时出错: {e}")
    
    def _plot_loss_curves(self, results):
        """Plot loss curves"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot training and validation losses
            epochs = range(1, len(results['train_losses']) + 1)
            
            plt.subplot(2, 1, 1)
            plt.plot(epochs, results['train_losses'], 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, results['val_losses'], 'r-', label='Validation Loss', linewidth=2)
            plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot log-scale losses
            plt.subplot(2, 1, 2)
            plt.semilogy(epochs, results['train_losses'], 'b-', label='Training Loss', linewidth=2, alpha=0.7)
            plt.semilogy(epochs, results['val_losses'], 'r-', label='Validation Loss', linewidth=2, alpha=0.7)
            plt.title('Log-Scale Loss Curves', fontsize=14, fontweight='bold')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (log MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Save loss plot
            self._save_loss_plot(results)
        
        except Exception as e:
            print(f"Error plotting loss curves: {e}")
    
    def _display_training_stats(self, results):
        """显示训练统计信息"""
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        
        print(f"\n📊 训练统计:")
        print(f"最终训练损失: {train_losses[-1]:.6f}")
        print(f"最终验证损失: {val_losses[-1]:.6f}")
        
        # 计算改进程度
        initial_train_loss = train_losses[0] if train_losses else 0
        initial_val_loss = val_losses[0] if val_losses else 0
        
        if initial_train_loss > 0:
            train_improvement = (initial_train_loss - train_losses[-1]) / initial_train_loss * 100
            print(f"训练损失改进: {train_improvement:.1f}%")
        
        if initial_val_loss > 0:
            val_improvement = (initial_val_loss - val_losses[-1]) / initial_val_loss * 100
            print(f"验证损失改进: {val_improvement:.1f}%")
        
        # 显示损失范围
        if train_losses:
            print(f"训练损失范围: {min(train_losses):.6f} - {max(train_losses):.6f}")
        if val_losses:
            print(f"验证损失范围: {min(val_losses):.6f} - {max(val_losses):.6f}")

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
        
        # 显示结果和损失曲线
        controller.display_training_results(results)
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
    def _save_loss_plot(self, results, save_dir="training_results"):
        """
        保存损失曲线图到文件
        
        参数:
            results: 包含训练结果的字典
            save_dir: 保存目录
        """
        try:
            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成时间戳文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_curves_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            
            # 创建图表
            plt.figure(figsize=(15, 10))
            
            # 1. 训练和验证损失曲线
            plt.subplot(2, 2, 1)
            epochs = range(1, len(results['train_losses']) + 1)
            
            plt.plot(epochs, results['train_losses'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            plt.plot(epochs, results['val_losses'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. 对数尺度损失曲线
            plt.subplot(2, 2, 2)
            plt.semilogy(epochs, results['train_losses'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            plt.semilogy(epochs, results['val_losses'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            plt.title('Log-Scale Loss Curves', fontsize=12, fontweight='bold')
            plt.xlabel('Epochs')
            plt.ylabel('Log MSE Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. 损失改进率
            plt.subplot(2, 2, 3)
            if len(results['train_losses']) > 1:
                train_improvements = []
                val_improvements = []
                
                for i in range(1, len(results['train_losses'])):
                    train_imp = (results['train_losses'][i-1] - results['train_losses'][i]) / results['train_losses'][i-1] * 100
                    val_imp = (results['val_losses'][i-1] - results['val_losses'][i]) / results['val_losses'][i-1] * 100
                    train_improvements.append(train_imp)
                    val_improvements.append(val_imp)
                
                plt.plot(range(1, len(train_improvements) + 1), train_improvements, 'g-', 
                        linewidth=2, label='Training Improvement %', alpha=0.8)
                plt.plot(range(1, len(val_improvements) + 1), val_improvements, 'm-', 
                        linewidth=2, label='Validation Improvement %', alpha=0.8)
                plt.title('Epoch-to-Epoch Improvement Rate', fontsize=12, fontweight='bold')
                plt.xlabel('Epoch Transition')
                plt.ylabel('Improvement Percentage (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 4. 最终统计信息
            plt.subplot(2, 2, 4)
            # 清空这个子图，用于显示文本信息
            plt.axis('off')
            
            # 准备统计文本
            stats_text = [
                "TRAINING STATISTICS",
                "=" * 20,
                f"Total Epochs: {len(results['train_losses'])}",
                f"Best Val Loss: {results['best_val_loss']:.6f}",
                f"Final Train Loss: {results['train_losses'][-1]:.6f}",
                f"Final Val Loss: {results['val_losses'][-1]:.6f}",
                "",
                "IMPROVEMENTS",
                "=" * 20
            ]
            
            # 计算总体改进
            if len(results['train_losses']) > 1:
                total_train_imp = (results['train_losses'][0] - results['train_losses'][-1]) / results['train_losses'][0] * 100
                total_val_imp = (results['val_losses'][0] - results['val_losses'][-1]) / results['val_losses'][0] * 100
                stats_text.extend([
                    f"Train Improvement: {total_train_imp:.1f}%",
                    f"Val Improvement: {total_val_imp:.1f}%"
                ])
            
            # 添加训练参数信息
            stats_text.extend([
                "",
                "TRAINING INFO",
                "=" * 20,
                f"Timestamp: {timestamp}",
                f"Device: {self.device}",
                f"Model: ImprovedDnCNN-{NUM_LAYERS}L"
            ])
            
            # 显示文本
            plt.text(0.1, 0.95, '\n'.join(stats_text), transform=plt.gca().transAxes,
                    fontfamily='monospace', fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            
            # 设置总标题
            plt.suptitle('DnCNN Training Analysis Report', fontsize=16, fontweight='bold', y=0.98)
            
            # 调整布局并保存
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ 损失曲线已保存至: {filepath}")
            
            # 同时保存损失数据为CSV文件
            self._save_loss_data(results, save_dir, timestamp)
            
        except Exception as e:
            print(f"❌ 保存损失曲线时出错: {e}")

    def _save_loss_data(self, results, save_dir, timestamp):
        """
        保存损失数据为CSV文件
        
        参数:
            results: 训练结果
            save_dir: 保存目录
            timestamp: 时间戳
        """
        try:
            csv_filename = f"loss_data_{timestamp}.csv"
            csv_filepath = os.path.join(save_dir, csv_filename)
            
            # 准备数据
            epochs = range(1, len(results['train_losses']) + 1)
            
            # 创建DataFrame
            import pandas as pd
            loss_data = {
                'epoch': list(epochs),
                'train_loss': results['train_losses'],
                'val_loss': results['val_losses']
            }
            
            # 计算改进率
            if len(results['train_losses']) > 1:
                train_improvements = [0]  # 第一轮没有改进
                val_improvements = [0]
                
                for i in range(1, len(results['train_losses'])):
                    train_imp = (results['train_losses'][i-1] - results['train_losses'][i]) / results['train_losses'][i-1] * 100
                    val_imp = (results['val_losses'][i-1] - results['val_losses'][i]) / results['val_losses'][i-1] * 100
                    train_improvements.append(train_imp)
                    val_improvements.append(val_imp)
                
                loss_data['train_improvement_%'] = train_improvements
                loss_data['val_improvement_%'] = val_improvements
            
            df = pd.DataFrame(loss_data)
            df.to_csv(csv_filepath, index=False, encoding='utf-8')
            
            print(f"✅ 损失数据已保存至: {csv_filepath}")
            
        except Exception as e:
            print(f"❌ 保存损失数据时出错: {e}")

if __name__ == "__main__":
    model_training()