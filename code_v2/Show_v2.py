"""
Real_Show.py
实际应用图像降噪 - 基于原有接口的直接降噪版本
"""

import torch
import os
import cv2
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from views.cli_view import CLIView
from controllers.denoise_controller import DenoiseController
from utils.image_utils import get_model_path

def show_welcome():
    """显示欢迎信息"""
    print("=" * 60)
    print("         实际图像降噪处理器 (DnCNN)")
    print("=" * 60)
    print("PyTorch 版本:", torch.__version__)
    print("是否支持 CUDA:", torch.cuda.is_available())
    print("是否支持 MPS:", torch.backends.mps.is_available())
    
    # 确定设备
    if torch.cuda.is_available():
        device = "cuda"
        print("当前设备: CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("当前设备: Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("当前设备: CPU")

def get_image_path():
    """获取图像文件路径"""
    while True:
        image_path = input("\n请输入要降噪的图像文件路径: ").strip().strip('"\'').strip()
        
        if not image_path:
            print("错误: 请输入有效的文件路径")
            continue
            
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 文件 '{image_path}' 不存在")
            print("请检查路径是否正确，注意：")
            print("1. 使用绝对路径（推荐）")
            print("2. 或者确保相对路径正确")
            print("3. 路径中不要有特殊字符")
            continue
            
        # 检查是否为图像文件
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in valid_extensions:
            print(f"警告: 文件扩展名 '{file_ext}' 可能不是标准图像格式")
            confirm = input("是否继续处理? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                continue
        
        return image_path

def get_output_option(original_path):
    """获取输出选项"""
    print("\n请选择输出选项:")
    print("1. 自动保存 (在原图目录生成降噪版本)")
    print("2. 指定保存路径")
    print("3. 仅显示不保存")
    
    while True:
        choice = input("请选择 (1-3): ").strip()
        
        if choice == '1':
            # 自动生成输出路径
            dir_name = os.path.dirname(original_path)
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            ext_name = os.path.splitext(original_path)[1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(dir_name or ".", f"{base_name}_denoised_{timestamp}{ext_name}")
            return output_path, True
            
        elif choice == '2':
            # 用户指定路径
            while True:
                output_path = input("请输入保存路径: ").strip().strip('"\'').strip()
                if not output_path:
                    print("错误: 请输入有效的路径")
                    continue
                
                # 检查目录是否存在
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    print(f"警告: 目录 '{output_dir}' 不存在")
                    create = input("是否创建该目录? (y/n): ").strip().lower()
                    if create in ['y', 'yes']:
                        os.makedirs(output_dir, exist_ok=True)
                    else:
                        continue
                
                return output_path, True
                
        elif choice == '3':
            return None, False
            
        else:
            print("错误: 请输入 1, 2 或 3")

def save_results(result, output_path, original_path):
    """保存处理结果"""
    try:
        # 保存降噪图像
        if 'dncnn' in result['results']:
            denoised_image = result['results']['dncnn']
        else:
            # 如果有多种方法，使用第一个
            first_method = list(result['results'].keys())[0]
            denoised_image = result['results'][first_method]
            print(f"注意: 使用 {first_method} 方法的降噪结果")
        
        # 保存图像
        success = cv2.imwrite(output_path, denoised_image)
        if not success:
            print(f"警告: 保存图像失败，尝试调整格式...")
            # 尝试使用.jpg格式
            if not output_path.lower().endswith('.jpg'):
                output_path = os.path.splitext(output_path)[0] + '.jpg'
                success = cv2.imwrite(output_path, denoised_image)
        
        if success:
            print(f"\n✓ 降噪图像已保存到: {output_path}")
            
            # 保存处理信息
            info_path = os.path.splitext(output_path)[0] + "_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("降噪处理信息\n")
                f.write("=" * 50 + "\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"原图像: {os.path.basename(original_path)}\n")
                f.write(f"图像尺寸: {result['original'].shape[1]}x{result['original'].shape[0]}\n")
                f.write(f"降噪方法: DnCNN\n")
                f.write(f"模型文件: {os.path.basename(model_path) if 'model_path' in locals() else '未知'}\n")
            
            print(f"✓ 处理信息已保存到: {info_path}")
            return True
        else:
            print("错误: 无法保存图像，请检查文件权限或磁盘空间")
            return False
            
    except Exception as e:
        print(f"保存结果时出错: {e}")
        return False

def display_image_comparison(original, denoised):
    """显示图像对比"""
    try:
        import matplotlib.pyplot as plt
        
        # 转换为RGB格式
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original_rgb)
        axes[0].set_title("原始图像", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(denoised_rgb)
        axes[1].set_title("降噪图像 (DnCNN)", fontsize=12)
        axes[1].axis('off')
        
        plt.suptitle("降噪效果对比", fontsize=14)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\n未安装matplotlib，使用OpenCV显示对比...")
        # 调整图像大小以便显示
        max_height = 800
        scale = max_height / original.shape[0]
        new_width = int(original.shape[1] * scale)
        new_height = int(original.shape[0] * scale)
        
        original_resized = cv2.resize(original, (new_width, new_height))
        denoised_resized = cv2.resize(denoised, (new_width, new_height))
        
        cv2.imshow("Original Image (按任意键继续)", original_resized)
        cv2.waitKey(0)
        cv2.imshow("Denoised Image (按任意键继续)", denoised_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """主程序"""
    try:
        # 显示欢迎信息
        show_welcome()
        
        # 获取模型路径
        print("\n" + "-" * 60)
        print("选择降噪模型:")
        model_path = get_model_path()
        if not model_path:
            print("错误: 无法获取模型路径")
            return
        
        # 初始化控制器
        print("\n初始化降噪器...")
        controller = DenoiseController()
        if not controller.initialize_denoiser(model_path):
            print("错误: 降噪器初始化失败")
            return
        print("✓ 降噪器初始化成功")
        
        # 获取图像路径
        image_path = get_image_path()
        
        # 获取输出选项
        output_path, should_save = get_output_option(image_path)
        
        # 处理图像（不添加噪声）
        print("\n正在处理图像，请稍候...")
        
        # 使用空噪声列表表示不添加噪声
        result = controller.process_single_image(
            image_path=image_path,
            noise_types=[],      # 空列表 - 不添加噪声
            intensities=[]       # 空列表 - 不添加噪声
        )
        
        print("✓ 图像处理完成")
        
        # 获取降噪结果
        if 'dncnn' in result['results']:
            denoised_image = result['results']['dncnn']
            method_name = "DnCNN"
        else:
            # 如果有其他方法，使用第一个
            first_method = list(result['results'].keys())[0]
            denoised_image = result['results'][first_method]
            method_name = first_method.upper()
        
        # 显示基本信息
        print("\n" + "=" * 60)
        print("处理结果摘要:")
        print(f"原图像: {os.path.basename(image_path)}")
        print(f"图像尺寸: {result['original'].shape[1]}x{result['original'].shape[0]}")
        print(f"降噪方法: {method_name}")
        
        # 显示对比
        show_comparison = input("\n是否显示处理前后对比? (y/n): ").strip().lower()
        if show_comparison in ['y', 'yes']:
            display_image_comparison(result['original'], denoised_image)
        
        # 保存结果
        if should_save and output_path:
            save_results(result, output_path, image_path)
        
        print("\n" + "=" * 60)
        print("降噪处理完成！")
        
        # 询问是否处理其他图像
        another = input("\n是否处理另一张图像? (y/n): ").strip().lower()
        if another in ['y', 'yes']:
            main()
        else:
            print("感谢使用！")
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供错误处理建议
        print("\n" + "=" * 60)
        print("故障排除建议:")
        print("1. 检查图像文件是否损坏")
        print("2. 确保模型文件完整")
        print("3. 检查文件路径是否正确")
        print("4. 确保有足够的磁盘空间")
        print("=" * 60)

if __name__ == "__main__":
    main()