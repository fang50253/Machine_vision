import torch
import sys
import cv2
import platform

def setup_device():
    """设置训练设备 - 跨平台兼容版本"""
    print("正在检测可用设备...")
    
    system = platform.system()
    print(f"当前操作系统: {system}")
    
    if system == "Darwin":  # macOS
        return _setup_macos_device()
    elif system == "Windows":  # Windows
        return _setup_windows_device()
    else:  # Linux 或其他系统
        return _setup_linux_device()

def _setup_macos_device():
    """设置 macOS 设备"""
    print("检测到 macOS 系统")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # 检查 MPS 可用性
        if torch.backends.mps.is_built():
            device = torch.device('mps')
            
            try:
                # 测试 MPS 设备
                test_tensor = torch.tensor([1.0]).to(device)
                result = test_tensor * 2
                
                print("✓ Apple Silicon GPU (MPS) 可用")
                print(f"  正在使用 MPS 设备进行训练")
                
                # 设置 MPS 优化
                torch.backends.mps.enabled = True
                
                return device
                
            except Exception as e:
                print(f"✗ MPS 设备测试失败: {e}")
                print("  回退到 CPU")
                return torch.device('cpu')
        else:
            print("✗ PyTorch 未构建 MPS 支持")
            print("  请安装支持 MPS 的 PyTorch 版本")
            return torch.device('cpu')
    else:
        print("✗ MPS 不可用，使用 CPU 训练")
        return torch.device('cpu')

def _setup_windows_device():
    """设置 Windows 设备"""
    print("检测到 Windows 系统")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        
        gpu_count = torch.cuda.device_count()
        print(f"发现 {gpu_count} 个 NVIDIA GPU 设备:")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            compute_capability = f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            print(f"  GPU {i}: {gpu_name}")
            print(f"    显存: {gpu_memory:.1f} GB")
            print(f"    计算能力: {compute_capability}")
        
        # 选择最佳 GPU（通常是 0 号）
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        try:
            # 测试 GPU 计算
            test_tensor = torch.tensor([1.0]).cuda()
            if test_tensor.is_cuda:
                print("✓ GPU 测试通过，正在使用 GPU 进行训练")
                return device
            else:
                print("✗ GPU 测试失败，回退到 CPU")
                return torch.device('cpu')
        except Exception as e:
            print(f"✗ GPU 测试失败: {e}")
            return torch.device('cpu')
    else:
        print("✗ 未检测到可用的 CUDA 设备，使用 CPU 训练")
        return torch.device('cpu')

def _setup_linux_device():
    """设置 Linux 设备"""
    print("检测到 Linux 系统")
    
    if torch.cuda.is_available():
        # 使用与 Windows 相同的 CUDA 设置
        return _setup_windows_device()
    else:
        print("✗ 未检测到可用的 CUDA 设备，使用 CPU 训练")
        return torch.device('cpu')

def check_pytorch_device_support():
    """检查 PyTorch 设备支持 - 跨平台版本"""
    print("\n" + "="*60)
    print("PyTorch 设备支持诊断")
    print("="*60)
    
    system = platform.system()
    print(f"操作系统: {system}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检查 CUDA 支持
    print(f"\nCUDA 支持:")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"  torch.version.cuda: {torch.version.cuda}")
        print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 检查 MPS 支持 (macOS)
    print(f"\nApple Silicon GPU (MPS) 支持:")
    if hasattr(torch.backends, 'mps'):
        print(f"  torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
        print(f"  torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")
    else:
        print("  MPS 后端不可用")
    
    # 设备测试
    print(f"\n设备测试:")
    devices_to_test = []
    
    if torch.cuda.is_available():
        devices_to_test.append(('CUDA', torch.device('cuda')))
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append(('MPS', torch.device('mps')))
    
    devices_to_test.append(('CPU', torch.device('cpu')))
    
    for device_name, device in devices_to_test:
        try:
            x = torch.randn(3, 3).to(device)
            y = torch.randn(3, 3).to(device)
            z = x + y
            print(f"  ✓ {device_name} 计算测试通过")
            
            # 测试矩阵乘法（更复杂的操作）
            if device_name != 'CPU':  # CPU 总是能工作
                a = torch.randn(100, 100).to(device)
                b = torch.randn(100, 100).to(device)
                c = torch.mm(a, b)
                print(f"  ✓ {device_name} 矩阵乘法测试通过")
                
        except Exception as e:
            print(f"  ✗ {device_name} 计算测试失败: {e}")
    
    # 推荐的最佳设备
    best_device = setup_device()
    print(f"\n🎯 推荐使用设备: {best_device}")
    
    print("="*60)
    return best_device

def get_device_info(device):
    """获取设备详细信息"""
    info = {
        'type': str(device),
        'system': platform.system(),
        'pytorch_version': torch.__version__
    }
    
    if device.type == 'cuda':
        info['gpu_name'] = torch.cuda.get_device_name(device)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(device).total_memory / 1024**3
        info['cuda_version'] = torch.version.cuda
    elif device.type == 'mps':
        info['device_name'] = 'Apple Silicon GPU'
        info['backend'] = 'MPS (Metal Performance Shaders)'
    
    return info

def print_device_info(device):
    """打印设备信息"""
    info = get_device_info(device)
    
    print("\n" + "🎯 当前训练设备信息:")
    print(f"  设备类型: {info['type']}")
    print(f"  操作系统: {info['system']}")
    print(f"  PyTorch版本: {info['pytorch_version']}")
    
    if device.type == 'cuda':
        print(f"  GPU名称: {info['gpu_name']}")
        print(f"  显存: {info['gpu_memory_gb']:.1f} GB")
        print(f"  CUDA版本: {info['cuda_version']}")
    elif device.type == 'mps':
        print(f"  设备: {info['device_name']}")
        print(f"  后端: {info['backend']}")
    else:
        print(f"  使用CPU进行训练")

# 使用示例
if __name__ == "__main__":
    # 运行设备诊断
    device = check_pytorch_device_support()
    
    # 打印最终选择的设备信息
    print_device_info(device)
    
    # 示例：创建一些测试数据
    print(f"\n🧪 设备性能测试:")
    x = torch.randn(1000, 1000).to(device)
    
    import time
    start_time = time.time()
    
    # 执行一些计算密集型操作
    for _ in range(100):
        x = torch.matmul(x, x) * 0.99
    
    end_time = time.time()
    print(f"  计算耗时: {end_time - start_time:.2f} 秒")
    print(f"  最终设备: {x.device}")
# 在 device_utils.py 文件末尾添加以下代码：

def check_pytorch_cuda_support():
    """
    兼容性函数 - 保持旧代码的导入正常工作
    注意：这个函数只检查 CUDA，不检查 MPS
    """
    print("\n" + "="*50)
    print("PyTorch CUDA支持诊断 (兼容模式)")
    print("="*50)
    
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
        
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print("✓ GPU计算测试通过")
        except Exception as e:
            print(f"✗ GPU计算测试失败: {e}")
    else:
        print("✗ CUDA不可用")
        print(f"PyTorch版本: {torch.__version__}")
    
    print("="*50)
    
    # 返回设备（为了兼容性）
    return setup_device()

# 可选：添加其他兼容性函数
def get_available_device():
    """兼容性函数 - 替代旧的设备获取方式"""
    return setup_device()