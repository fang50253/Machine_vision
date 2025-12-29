import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from models.edge_enhancer import EdgeEnhancementNetwork

class EdgeEnhancerInference:
    """边缘增强推理类"""
    def __init__(self, model_path, device='auto'):
        """
        初始化边缘增强器
        
        Args:
            model_path: 模型权重路径
            device: 'auto', 'cuda', 'cpu', 'mps'
        """
        # 设置设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("🚀 使用 CUDA")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("🚀 使用 MPS")
            else:
                self.device = torch.device('cpu')
                print("⚙️ 使用 CPU")
        else:
            self.device = torch.device(device)
        
        print(f"设备: {self.device}")
        
        # 加载模型
        self.model = EdgeEnhancementNetwork().to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        print("✅ 边缘增强模型加载完成")
    
    def load_model(self, model_path):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理可能的'module.'前缀
        from collections import OrderedDict
        if all(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 去掉'module.'前缀
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"📦 加载模型: {os.path.basename(model_path)}")
    
    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):
            # 读取图像文件
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 保存原始尺寸
        original_h, original_w = image.shape[:2]
        
        # 调整尺寸为32的倍数（方便网络处理）
        new_h = (original_h // 32) * 32
        new_w = (original_w // 32) * 32
        
        if new_h != original_h or new_w != original_w:
            image = cv2.resize(image, (new_w, new_h))
            print(f"调整尺寸: {original_w}x{original_h} -> {new_w}x{new_h}")
        
        # 转换为tensor
        image_tensor = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device), original_h, original_w
    
    def postprocess_image(self, tensor, original_h, original_w):
        """后处理图像"""
        # 转换回numpy
        image = tensor.squeeze(0).cpu().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image = image.transpose(1, 2, 0)
        
        # 恢复原始尺寸
        if image.shape[0] != original_h or image.shape[1] != original_w:
            image = cv2.resize(image, (original_w, original_h))
        
        return image
    
    def enhance_image(self, image, strength=1.0):
        """
        增强图像边缘细节
        
        Args:
            image: 输入图像路径或numpy数组
            strength: 增强强度 (0.0-2.0)
        
        Returns:
            enhanced: 增强后的图像
            edge_map: 边缘图
        """
        # 预处理
        image_tensor, original_h, original_w = self.preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            enhanced_tensor, edge_tensor = self.model(image_tensor)
            
            # 应用强度控制
            if strength != 1.0:
                residual = enhanced_tensor - image_tensor
                enhanced_tensor = image_tensor + residual * strength
        
        # 后处理
        enhanced = self.postprocess_image(enhanced_tensor, original_h, original_w)
        edge_map = self.postprocess_image(edge_tensor, original_h, original_w)
        
        return enhanced, edge_map
    
    def enhance_image_file(self, input_path, output_dir='output', strength=1.0, 
                          show_comparison=True, save_edge_map=False):
        """
        增强图像文件
        
        Args:
            input_path: 输入图像路径
            output_dir: 输出目录
            strength: 增强强度
            show_comparison: 是否显示对比
            save_edge_map: 是否保存边缘图
        
        Returns:
            输出图像路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取原始图像
        original_bgr = cv2.imread(input_path)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        # 增强图像
        enhanced_rgb, edge_rgb = self.enhance_image(original_rgb, strength)
        
        # 转换为BGR保存
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        edge_bgr = cv2.cvtColor(edge_rgb, cv2.COLOR_RGB2BGR)
        
        # 生成输出路径
        input_name = Path(input_path).stem
        output_path = os.path.join(
            output_dir, 
            f"{input_name}_enhanced_strength{strength:.1f}.jpg"
        )
        
        # 保存结果
        cv2.imwrite(output_path, enhanced_bgr)
        print(f"✅ 增强图像保存到: {output_path}")
        
        if save_edge_map:
            edge_path = os.path.join(output_dir, f"{input_name}_edges.jpg")
            cv2.imwrite(edge_path, edge_bgr)
            print(f"✅ 边缘图保存到: {edge_path}")
        
        # 显示对比
        if show_comparison:
            self.display_comparison(original_rgb, enhanced_rgb, edge_rgb, strength)
        
        return output_path
    
    def display_comparison(self, original, enhanced, edge_map, strength):
        """显示对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # 增强图像
        axes[0, 1].imshow(enhanced)
        axes[0, 1].set_title(f'Enhanced Image\n(Strength: {strength:.1f})', fontsize=12)
        axes[0, 1].axis('off')
        
        # 边缘图
        axes[0, 2].imshow(edge_map.mean(axis=2), cmap='gray')
        axes[0, 2].set_title('Edge Map', fontsize=12)
        axes[0, 2].axis('off')
        
        # 差异图
        diff = np.abs(original.astype(float) - enhanced.astype(float))
        im = axes[1, 0].imshow(diff.mean(axis=2), cmap='hot', vmax=30)
        axes[1, 0].set_title('Difference Map', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 局部放大
        h, w = original.shape[:2]
        crop_size = min(200, h//4, w//4)
        y, x = h//2 - crop_size//2, w//2 - crop_size//2
        
        # 原始局部
        axes[1, 1].imshow(original[y:y+crop_size, x:x+crop_size])
        axes[1, 1].set_title('Original (Zoomed)', fontsize=12)
        axes[1, 1].axis('off')
        
        # 增强局部
        axes[1, 2].imshow(enhanced[y:y+crop_size, x:x+crop_size])
        axes[1, 2].set_title('Enhanced (Zoomed)', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.suptitle('Edge Enhancement Results', fontsize=16)
        plt.tight_layout()
        plt.show()

def batch_enhance(input_dir, output_dir, model_path, strength=1.0):
    """批量增强图像"""
    enhancer = EdgeEnhancerInference(model_path)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
    
    print(f"找到 {len(image_files)} 张图像")
    
    for img_path in image_files:
        try:
            print(f"\n处理: {img_path.name}")
            output_path = enhancer.enhance_image_file(
                str(img_path), output_dir, strength, 
                show_comparison=False, save_edge_map=True
            )
        except Exception as e:
            print(f"❌ 处理失败 {img_path.name}: {e}")
    
    print(f"\n✅ 批量处理完成!")
    print(f"输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Edge Enhancement Inference')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', '-o', type=str, default='enhanced_results',
                       help='Output directory')
    parser.add_argument('--model', '-m', type=str, default='checkpoints/dncnn_to_original/best_model.pth',
                       help='Model checkpoint path')
    parser.add_argument('--strength', '-s', type=float, default=1.0,
                       help='Enhancement strength (0.0-2.0)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Batch process directory')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理
        batch_enhance(args.input, args.output, args.model, args.strength)
    else:
        # 单张图像处理
        enhancer = EdgeEnhancerInference(args.model)
        enhancer.enhance_image_file(
            args.input, args.output, args.strength,
            show_comparison=True, save_edge_map=True
        )

if __name__ == "__main__":
    main()