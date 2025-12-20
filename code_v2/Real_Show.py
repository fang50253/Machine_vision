"""
debug_dncnn_enhanced.py
Enhanced DnCNN Denoising with Strength Control
"""

import torch
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_dncnn_model_debug(model_path):
    """Load DnCNN model - debug version"""
    print("=" * 60)
    print("🔧 DnCNN Model Loading (Debug)")
    print("=" * 60)
    
    try:
        # Check model file
        print(f"📁 Model path: {model_path}")
        print(f"📊 File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        if not os.path.exists(model_path):
            print("❌ Model file does not exist")
            return None, None
        
        # Import modules
        from config import NUM_LAYERS
        from models.dncnn import DnCNN
        
        print(f"🔧 NUM_LAYERS config: {NUM_LAYERS}")
        
        # Set device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🚀 Using CUDA (NVIDIA GPU)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🚀 Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device('cpu')
            print("⚙️ Using CPU")
        
        # Create model
        print(f"🤖 Creating DnCNN model: channels=3, num_layers={NUM_LAYERS}")
        model = DnCNN(channels=3, num_layers=NUM_LAYERS)
        
        # Load weights
        print("📦 Loading model weights...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Debug: show checkpoint info
        print(f"🔍 Checkpoint type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"📋 Using 'model_state_dict' key")
            else:
                state_dict = checkpoint
                print(f"📋 Using checkpoint directly as state_dict")
        else:
            state_dict = checkpoint
            print(f"📋 Checkpoint is not dict, using directly")
        
        # Handle 'module.' prefix
        from collections import OrderedDict
        if all(key.startswith('module.') for key in state_dict.keys()):
            print("🔄 Removing 'module.' prefix...")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # Remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        # Show weight info
        print(f"📋 State dict keys count: {len(state_dict)}")
        print("🔍 First 5 weight layers:")
        for i, (key, value) in enumerate(list(state_dict.items())[:5]):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def add_controllable_noise(image, noise_level=25, noise_type='gaussian'):
    """Add controllable noise for testing"""
    print(f"🔊 Adding {noise_type} noise (level={noise_level})...")
    
    if noise_type == 'gaussian':
        # Gaussian noise
        noise = np.random.randn(*image.shape) * noise_level
        noisy_image = image.astype(np.float32) + noise
    elif noise_type == 'salt_pepper':
        # Salt & Pepper noise
        noisy_image = image.copy().astype(np.float32)
        prob = noise_level / 100.0
        mask = np.random.random(image.shape[:2]) < prob
        noisy_image[mask] = 255  # Salt
        mask = np.random.random(image.shape[:2]) < prob
        noisy_image[mask] = 0    # Pepper
    else:
        # Uniform noise
        noise = np.random.uniform(-noise_level, noise_level, image.shape)
        noisy_image = image.astype(np.float32) + noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # Calculate PSNR
    mse = np.mean((image.astype(float) - noisy_image.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"📊 Noisy image PSNR: {psnr:.2f} dB")
    
    return noisy_image

def process_image_with_strength(model, device, image_path, strength=1.0, 
                                add_noise=False, noise_level=25, noise_type='gaussian'):
    """Process image with adjustable denoising strength"""
    print("\n" + "=" * 60)
    print(f"🖼️ Image Processing (Strength: {strength:.2f})")
    print("=" * 60)
    
    # 1. Read image
    print(f"📸 Reading image: {os.path.basename(image_path)}")
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    print(f"📐 Original size: {original.shape[1]}x{original.shape[0]}")
    print(f"🎨 Color mode: {'Color' if len(original.shape) == 3 else 'Grayscale'}")
    
    # 2. Add test noise if needed
    if add_noise:
        noisy = add_controllable_noise(original, noise_level, noise_type)
        image_to_denoise = noisy
        print(f"🎭 Noise type: {noise_type}, Level: {noise_level}")
    else:
        image_to_denoise = original
        print("⚡ Using original image (no noise added)")
    
    # 3. Preprocessing
    print("🔄 Preprocessing...")
    original_shape = image_to_denoise.shape
    
    # Convert to float32 and normalize
    image_float = image_to_denoise.astype(np.float32) / 255.0
    
    print(f"📊 Before preprocessing range: [{image_to_denoise.min()}, {image_to_denoise.max()}]")
    print(f"📊 After preprocessing range: [{image_float.min():.3f}, {image_float.max():.3f}]")
    
    # Convert to tensor
    if len(image_float.shape) == 3:
        image_tensor = torch.from_numpy(image_float.transpose(2, 0, 1))
    else:
        image_tensor = torch.from_numpy(image_float).unsqueeze(0)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    print(f"📊 Tensor shape: {image_tensor.shape}")
    print(f"📊 Tensor device: {image_tensor.device}")
    
    # 4. Model inference with strength control
    print(f"🤖 Model inference (Strength: {strength:.2f})...")
    with torch.no_grad():
        start_time = datetime.now()
        noise_pred = model(image_tensor)
        end_time = datetime.now()
        
        inference_time = (end_time - start_time).total_seconds()
        print(f"⏱️ Inference time: {inference_time:.3f} seconds")
        
        # ADJUSTABLE STRENGTH CONTROL
        # Method 1: Adjust noise prediction scale
        adjusted_noise = noise_pred * strength
        
        # Method 2: Blended output (interpolation)
        if strength > 1.0:
            # Stronger denoising: remove more noise
            output_tensor = image_tensor - adjusted_noise
        elif strength < 1.0:
            # Weaker denoising: partial noise removal
            output_tensor = image_tensor * (1 - 0.5 * strength) + (image_tensor - adjusted_noise) * (0.5 * strength)
        else:
            # Standard denoising
            output_tensor = image_tensor - noise_pred
        
        # Debug info
        print(f"📊 Noise prediction range: [{noise_pred.min().item():.4f}, {noise_pred.max().item():.4f}]")
        print(f"📊 Adjusted noise range: [{adjusted_noise.min().item():.4f}, {adjusted_noise.max().item():.4f}]")
        print(f"📊 Output range: [{output_tensor.min().item():.4f}, {output_tensor.max().item():.4f}]")
    
    # 5. Post-processing
    print("🔄 Post-processing...")
    
    # Move to CPU and convert to numpy
    output_tensor = output_tensor.cpu()
    output_np = output_tensor.squeeze(0).numpy()
    
    if len(output_np.shape) == 3:
        output_np = output_np.transpose(1, 2, 0)
    
    # De-normalize and clip
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    
    print(f"📊 Processed range: [{output_np.min()}, {output_np.max()}]")
    
    # Resize if needed
    if output_np.shape != original_shape:
        print(f"🔄 Resizing output: {output_np.shape} -> {original_shape}")
        output_np = cv2.resize(output_np, (original_shape[1], original_shape[0]))
    
    # 6. Calculate statistics
    print("📈 Calculating statistics...")
    
    if add_noise:
        diff_original_noisy = np.abs(original.astype(float) - noisy.astype(float))
        diff_original_denoised = np.abs(original.astype(float) - output_np.astype(float))
        
        print(f"📊 Original↔Noisy mean diff: {diff_original_noisy.mean():.2f}")
        print(f"📊 Original↔Denoised mean diff: {diff_original_denoised.mean():.2f}")
        print(f"📊 Difference reduction: {100*(1 - diff_original_denoised.mean()/diff_original_noisy.mean()):.1f}%")
    else:
        diff = np.abs(image_to_denoise.astype(float) - output_np.astype(float))
        print(f"📊 Before↔After mean diff: {diff.mean():.2f}")
        print(f"📊 Before↔After max diff: {diff.max():.2f}")
        print(f"📊 Pixels changed >10: {(diff > 10).sum()}")
    
    return original, image_to_denoise, output_np

def display_comparison_debug(original, noisy, denoised, add_noise, strength):
    """Display detailed comparison - debug version"""
    if add_noise:
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'DnCNN Denoising Comparison (Strength: {strength:.2f})', fontsize=16)
        
        # Three images: Original, Noisy, Denoised
        titles = ['Original Image', f'Noisy Image\n(Test Noise)', f'Denoised Image\n(Strength: {strength:.1f})']
        images = [original, noisy, denoised]
        
        for i in range(3):
            ax = plt.subplot(2, 3, i+1)
            if len(images[i].shape) == 3:
                img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            else:
                ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i], fontsize=12)
            ax.axis('off')
        
        # Show difference maps
        ax4 = plt.subplot(2, 3, 4)
        diff_noisy = np.abs(original.astype(float) - noisy.astype(float))
        im4 = ax4.imshow(diff_noisy.mean(axis=2) if len(diff_noisy.shape)==3 else diff_noisy, 
                        cmap='hot', vmax=50)
        ax4.set_title('Original-Noisy Difference', fontsize=12)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        ax5 = plt.subplot(2, 3, 5)
        diff_denoised = np.abs(original.astype(float) - denoised.astype(float))
        im5 = ax5.imshow(diff_denoised.mean(axis=2) if len(diff_denoised.shape)==3 else diff_denoised, 
                        cmap='hot', vmax=50)
        ax5.set_title('Original-Denoised Difference', fontsize=12)
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        ax6 = plt.subplot(2, 3, 6)
        improvement = diff_noisy - diff_denoised
        im6 = ax6.imshow(improvement.mean(axis=2) if len(improvement.shape)==3 else improvement, 
                        cmap='coolwarm', vmin=-30, vmax=30)
        ax6.set_title('Denoising Improvement\n(Noisy-Denoised Difference)', fontsize=12)
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
    else:
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(f'DnCNN Processing Comparison (Strength: {strength:.2f})', fontsize=16)
        
        # Two images: Original, Processed
        titles = ['Original Image', f'Processed Image\n(Strength: {strength:.1f})']
        images = [original, denoised]
        
        for i in range(2):
            ax = plt.subplot(2, 4, i*4+1)
            if len(images[i].shape) == 3:
                img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            else:
                ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i], fontsize=12)
            ax.axis('off')
        
        # Show difference map
        ax3 = plt.subplot(2, 4, 3)
        diff = np.abs(original.astype(float) - denoised.astype(float))
        im3 = ax3.imshow(diff.mean(axis=2) if len(diff.shape)==3 else diff, 
                        cmap='hot', vmax=30)
        ax3.set_title('Difference Map', fontsize=12)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Show histogram of differences
        ax4 = plt.subplot(2, 4, 4)
        diff_flat = diff.flatten()
        ax4.hist(diff_flat[diff_flat > 0], bins=50, alpha=0.7, color='blue')
        ax4.set_title('Difference Histogram\n(Positive differences only)', fontsize=12)
        ax4.set_xlabel('Difference value')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Show zoomed areas
        h, w = original.shape[:2]
        crop_size = min(200, h//4, w//4)
        y, x = h//2 - crop_size//2, w//2 - crop_size//2
        
        for i, img in enumerate([original, denoised]):
            ax = plt.subplot(2, 4, 5 + i)
            if len(img.shape) == 3:
                crop = img[y:y+crop_size, x:x+crop_size, :]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
            else:
                crop = img[y:y+crop_size, x:x+crop_size]
                ax.imshow(crop, cmap='gray')
            ax.set_title(f'{titles[i]}\n(Zoomed Area)', fontsize=10)
            ax.axis('off')
        
        # Show edge comparison (using Sobel)
        ax7 = plt.subplot(2, 4, 7)
        edges_original = cv2.Sobel(original.mean(axis=2).astype(np.float32), 
                                 cv2.CV_32F, 1, 1, ksize=3)
        edges_original = np.abs(edges_original)
        ax7.imshow(edges_original, cmap='gray')
        ax7.set_title('Original Edges (Sobel)', fontsize=10)
        ax7.axis('off')
        
        ax8 = plt.subplot(2, 4, 8)
        edges_denoised = cv2.Sobel(denoised.mean(axis=2).astype(np.float32), 
                                 cv2.CV_32F, 1, 1, ksize=3)
        edges_denoised = np.abs(edges_denoised)
        ax8.imshow(edges_denoised, cmap='gray')
        ax8.set_title('Processed Edges (Sobel)', fontsize=10)
        ax8.axis('off')
    
    plt.tight_layout()
    plt.show()

def run_strength_comparison(model, device, image_path, add_noise=True, 
                           noise_level=25, noise_type='gaussian'):
    """Compare different denoising strengths"""
    print("\n" + "=" * 70)
    print("          STRENGTH COMPARISON EXPERIMENT")
    print("=" * 70)
    
    # Read image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Add noise for testing
    if add_noise:
        noisy = add_controllable_noise(original, noise_level, noise_type)
        image_to_process = noisy
    else:
        image_to_process = original
    
    # Test different strengths
    strengths = [0.5, 1.0, 1.5, 2.0]
    results = {}
    
    for strength in strengths:
        print(f"\n🔧 Testing strength: {strength:.1f}")
        _, _, denoised = process_image_with_strength(
            model, device, image_path, strength, 
            add_noise=False,  # Already added noise
            noise_level=noise_level, 
            noise_type=noise_type
        )
        results[f'Strength_{strength:.1f}'] = denoised
    
    # Display comparison
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'DnCNN Denoising Strength Comparison\n(Noise: {noise_type}, Level: {noise_level})', 
                 fontsize=16)
    
    # Show original and noisy
    ax1 = plt.subplot(3, 4, 1)
    if len(original.shape) == 3:
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    if add_noise:
        if len(image_to_process.shape) == 3:
            ax2.imshow(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB))
        else:
            ax2.imshow(image_to_process, cmap='gray')
    ax2.set_title(f'Noisy Image\n({noise_type}, {noise_level})', fontsize=12)
    ax2.axis('off')
    
    # Show different strength results
    for idx, (label, image) in enumerate(results.items()):
        ax = plt.subplot(3, 4, idx + 3)
        if len(image.shape) == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap='gray')
        
        # Extract strength value from label
        strength_val = float(label.split('_')[1])
        ax.set_title(f'Strength: {strength_val:.1f}', fontsize=12)
        ax.axis('off')
    
    # Show difference maps (last row)
    for idx, (label, image) in enumerate(results.items()):
        ax = plt.subplot(3, 4, idx + 7)
        
        if add_noise:
            diff = np.abs(image_to_process.astype(float) - image.astype(float))
        else:
            diff = np.abs(original.astype(float) - image.astype(float))
        
        im = ax.imshow(diff.mean(axis=2) if len(diff.shape)==3 else diff, 
                      cmap='hot', vmax=50)
        
        strength_val = float(label.split('_')[1])
        ax.set_title(f'Difference\n(Strength: {strength_val:.1f})', fontsize=10)
        ax.axis('off')
        
        if idx == len(results) - 1:  # Add colorbar for last one
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"strength_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f"{output_dir}/original.jpg", original)
    if add_noise:
        cv2.imwrite(f"{output_dir}/noisy.jpg", image_to_process)
    
    for label, image in results.items():
        cv2.imwrite(f"{output_dir}/{label}.jpg", image)
    
    print(f"\n✅ Comparison results saved to: {output_dir}/")
    return results

def main():
    """Main program"""
    print("=" * 70)
    print("          ENHANCED DnCNN DENOISING DEBUG TOOL")
    print("          with Adjustable Denoising Strength")
    print("=" * 70)
    print("Options:")
    print("  1. Single image with specific strength")
    print("  2. Compare different strengths")
    print("  3. Exit")
    
    try:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '3':
            print("Exiting...")
            return
        
        # Get image path
        image_path = input("\nEnter image path: ").strip().strip('"\'').strip()
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Get model path
        model_dir = "trained_models"
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if not model_files:
            print("❌ No model files found")
            return
        
        # Use first model by default
        model_path = os.path.join(model_dir, model_files[0])
        print(f"🤖 Using model: {model_files[0]}")
        
        # Load model
        model, device = load_dncnn_model_debug(model_path)
        if model is None:
            return
        
        if choice == '1':
            # Single image with specific strength
            try:
                strength = float(input("\nEnter denoising strength (0.5-2.0, 1.0=default): ").strip())
                strength = max(0.1, min(3.0, strength))  # Clamp to reasonable range
            except:
                strength = 1.0
                print(f"Using default strength: {strength}")
            
            # Noise options
            add_noise = input("\nAdd test noise? (y/n): ").strip().lower() in ['y', 'yes']
            noise_level = 25
            noise_type = 'gaussian'
            
            if add_noise:
                try:
                    noise_level = float(input("Noise level (10-50, default=25): ").strip())
                    noise_level = max(5, min(100, noise_level))
                except:
                    noise_level = 25
                
                noise_types = ['gaussian', 'salt_pepper', 'uniform']
                print("Noise types: 1. Gaussian, 2. Salt & Pepper, 3. Uniform")
                try:
                    noise_choice = int(input("Select noise type (1-3, default=1): ").strip())
                    noise_type = noise_types[min(max(1, noise_choice), 3) - 1]
                except:
                    noise_type = 'gaussian'
            
            # Process image
            original, input_image, denoised = process_image_with_strength(
                model, device, image_path, strength, 
                add_noise, noise_level, noise_type
            )
            
            # Display comparison
            display_comparison_debug(original, input_image, denoised, add_noise, strength)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "debug_results"
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            cv2.imwrite(f"{output_dir}/{base_name}_original_{timestamp}.jpg", original)
            cv2.imwrite(f"{output_dir}/{base_name}_input_{timestamp}.jpg", input_image)
            cv2.imwrite(f"{output_dir}/{base_name}_denoised_strength{strength:.1f}_{timestamp}.jpg", denoised)
            
            print(f"\n✅ Results saved to {output_dir}/")
            
        elif choice == '2':
            # Compare different strengths
            add_noise = input("\nAdd test noise for comparison? (y/n): ").strip().lower() in ['y', 'yes']
            noise_level = 25
            noise_type = 'gaussian'
            
            if add_noise:
                try:
                    noise_level = float(input("Noise level (10-50, default=25): ").strip())
                    noise_level = max(5, min(100, noise_level))
                except:
                    noise_level = 25
                
                noise_types = ['gaussian', 'salt_pepper', 'uniform']
                print("Noise types: 1. Gaussian, 2. Salt & Pepper, 3. Uniform")
                try:
                    noise_choice = int(input("Select noise type (1-3, default=1): ").strip())
                    noise_type = noise_types[min(max(1, noise_choice), 3) - 1]
                except:
                    noise_type = 'gaussian'
            
            run_strength_comparison(model, device, image_path, add_noise, noise_level, noise_type)
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()