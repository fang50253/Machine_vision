import torch
import os  # 添加os导入
import cv2  # 添加cv2导入
from views.cli_view import CLIView
from controllers.denoise_controller import DenoiseController
from controllers.batch_controller import BatchController
from utils.image_utils import get_model_path
from config import NUM_LAYERS

def main():
    """主程序"""
    # 显示欢迎信息
    CLIView.show_welcome()
    print("PyTorch 版本:", torch.__version__)
    print("是否支持 CUDA:", torch.cuda.is_available())
    print("当前设备:", "cuda" if torch.cuda.is_available() else "cpu")
    
    while True:
        # 获取处理模式
        mode = CLIView.get_input_mode()
        
        if mode == '1':
            # 单张图像处理模式
            print("\n进入单张图像处理模式...")
            single_image_processing()
        elif mode == '2':
            # 批量处理模式
            print("\n进入批量处理模式...")
            batch_processing()
        elif mode == '3':
            print("程序退出。")
            break

def single_image_processing():
    """单张图像处理"""
    try:
        # 获取模型路径
        model_path = get_model_path()
        
        # 初始化控制器
        controller = DenoiseController()
        if not controller.initialize_denoiser(model_path):
            print("警告: 去噪器初始化失败")
            return
        
        # 获取图像路径
        image_path = input("\n请输入图像文件路径: ").strip().strip('"\'')
        if not os.path.exists(image_path):
            print("错误: 图像文件不存在")
            return
        
        # 获取噪声设置
        noise_types, intensities = CLIView.get_noise_settings()
        
        # 处理图像
        result = controller.process_single_image(image_path, noise_types, intensities)
        
        # 显示结果
        controller.display_metrics(result['metrics'])
        
        # 询问是否保存结果
        save_choice = input("\n是否保存处理结果？(y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            save_single_image_results(result, image_path)
            
    except Exception as e:
        print(f"处理过程中出错: {e}")

def batch_processing():
    """批量处理"""
    try:
        # 获取模型路径
        model_path = get_model_path()
        
        # 初始化控制器
        controller = BatchController()
        if not controller.initialize_denoiser(model_path):
            print("警告: 去噪器初始化失败")
            return
        
        # 获取文件夹路径
        folder_path = CLIView.get_folder_path()
        
        # 获取噪声设置
        from utils.image_utils import get_noise_settings_interactive
        noise_types, intensities = get_noise_settings_interactive()
        
        # 处理批量图像
        controller.process_batch(folder_path, noise_types, intensities)
        
    except Exception as e:
        print(f"批量处理过程中出错: {e}")

def save_single_image_results(result, original_path):
    """保存单张图像处理结果"""
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"single_result_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图像
    cv2.imwrite(os.path.join(output_dir, 'original.jpg'), result['original'])
    cv2.imwrite(os.path.join(output_dir, 'noisy.jpg'), result['noisy'])
    
    for method, image in result['results'].items():
        cv2.imwrite(os.path.join(output_dir, f'{method}.jpg'), image)
    
    # 保存指标
    with open(os.path.join(output_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("去噪效果评估\n")
        f.write("="*50 + "\n")
        f.write(f"原图像: {os.path.basename(original_path)}\n")
        f.write(f"噪声类型: {result['noise_types']}\n")
        f.write(f"噪声强度: {result['intensities']}\n")
        f.write(f"噪声图像 PSNR: {result['metrics']['noisy_psnr']:.2f} dB\n")
        if result['metrics']['noisy_ssim'] > 0:
            f.write(f"噪声图像 SSIM: {result['metrics']['noisy_ssim']:.3f}\n")
        
        f.write("\n各方法效果对比:\n")
        f.write("-" * 50 + "\n")
        
        for method, metrics in result['metrics']['methods'].items():
            f.write(f"{method:12}: PSNR: {metrics['psnr']:6.2f} dB "
                   f"(+{metrics['improvement_psnr']:5.2f}) | "
                   f"SSIM: {metrics['ssim']:.3f} "
                   f"(+{metrics['improvement_ssim']:.3f})\n")
    
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()