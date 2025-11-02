import os
from utils.image_utils import get_noise_settings_interactive

class CLIView:
    """命令行界面视图"""
    
    @staticmethod
    def get_noise_settings():
        """获取噪声设置 - 现在支持随机选项"""
        return get_noise_settings_interactive()
    
    @staticmethod
    def show_welcome():
        print("PyTorch版本检查...")
        print("\n" + "="*60)
        print("高级图像去噪测试系统 - MVC架构版")
        print("="*60)
    
    @staticmethod
    def get_input_mode():
        print("\n请选择处理模式：")
        print("1. 单张图像处理")
        print("2. 批量文件夹处理")
        print("3. 退出程序")
        
        while True:
            choice = input("\n请选择 (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                return choice
            print("无效选择，请重新输入。")
    
    @staticmethod
    def get_folder_path():
        while True:
            folder_path = input("\n请输入包含图像的文件夹路径: ").strip().strip('"\'')
            if not os.path.exists(folder_path):
                print(f"错误：文件夹 '{folder_path}' 不存在，请重新输入。")
                continue
            return folder_path
    
    @staticmethod
    def get_noise_settings():
        print("\n请选择噪声类型：")
        print("1. 高斯噪声")
        print("2. 椒盐噪声") 
        print("3. 混合噪声 (高斯+椒盐)")
        print("4. 自定义混合噪声")
        
        noise_choice = input("请选择 (1/2/3/4): ").strip() or '1'
        
        if noise_choice == '2':
            noise_types = ['salt_pepper']
        elif noise_choice == '3':
            noise_types = ['gaussian', 'salt_pepper']
        elif noise_choice == '4':
            custom_types = input("请输入噪声类型(用逗号分隔, 如: gaussian,salt_pepper): ").strip()
            noise_types = [t.strip() for t in custom_types.split(',')]
        else:
            noise_types = ['gaussian']
        
        intensities = []
        if len(noise_types) == 1:
            try:
                intensity = int(input(f"请输入噪声强度 (1-100, 默认25): ").strip() or '25')
                intensity = max(1, min(100, intensity))
                intensities = [intensity]
            except ValueError:
                intensities = [25]
                print("使用默认噪声强度: 25")
        else:
            print("\n请为每种噪声类型设置强度 (1-100):")
            for n_type in noise_types:
                try:
                    intensity = int(input(f"  {n_type} 噪声强度 (默认25): ").strip() or '25')
                    intensity = max(1, min(100, intensity))
                    intensities.append(intensity)
                except ValueError:
                    intensities.append(25)
                    print(f"  {n_type} 使用默认强度: 25")
        
        return noise_types, intensities
    
    @staticmethod
    def show_processing_progress(current, total, message=""):
        print(f"\r处理进度: {current}/{total} {message}", end="", flush=True)
        if current == total:
            print()
    
    @staticmethod
    def show_results_summary(results_data):
        if not results_data:
            return
        
        print(f"\n{'='*50}")
        print("批量处理完成！")
        print(f"{'='*50}")
        print(f"处理图像数量: {len(results_data)}")