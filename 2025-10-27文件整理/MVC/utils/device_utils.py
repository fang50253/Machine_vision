import torch
import sys
import cv2

def setup_device():
    """设置训练设备"""
    print("正在检测GPU设备...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
        
        gpu_count = torch.cuda.device_count()
        print(f"发现 {gpu_count} 个GPU设备:")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"    显存: {gpu_memory:.1f} GB")
        
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        test_tensor = torch.tensor([1.0]).cuda()
        if test_tensor.is_cuda:
            print("✓ GPU测试通过，正在使用GPU进行训练")
        else:
            print("✗ GPU测试失败，回退到CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("✗ 未检测到可用的CUDA设备，使用CPU训练")
    
    return device

def check_pytorch_cuda_support():
    """检查PyTorch的CUDA支持"""
    print("\n" + "="*50)
    print("PyTorch CUDA支持诊断")
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