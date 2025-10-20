import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_brightness(image_path):
    """
    计算图像亮度
    返回：平均亮度值（0-255）和亮度等级
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图像 {image_path}")
            return None, None
        
        print(f"图像尺寸: {img.shape}")
        
        # 方法1：转换为灰度图计算平均亮度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # 方法2：HSV空间的V通道（亮度通道）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        hsv_brightness = np.mean(v_channel)
        
        # 方法3：RGB空间计算亮度（加权平均）
        b, g, r = cv2.split(img)
        rgb_brightness = 0.299 * np.mean(r) + 0.587 * np.mean(g) + 0.114 * np.mean(b)
        
        print(f"灰度平均亮度: {avg_brightness:.2f}")
        print(f"HSV亮度: {hsv_brightness:.2f}")
        print(f"RGB加权亮度: {rgb_brightness:.2f}")
        
        return avg_brightness, classify_brightness(avg_brightness)
        
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None, None

def classify_brightness(brightness_value):
    """简单的亮度分类"""
    if brightness_value < 85:
        return "过暗"
    elif brightness_value < 170:
        return "正常" 
    else:
        return "过亮"

def analyze_image_histogram(image_path):
    """分析图像直方图"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    plt.figure(figsize=(12, 4))
    
    # 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')
    
    # 显示直方图
    plt.subplot(1, 2, 2)
    plt.plot(hist, color='black')
    plt.title('灰度直方图')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 主程序
def main():
    # 测试图片列表
    test_images = ['day.jpg', 'night.jpg', 'FZY09912.JPG']
    
    results = []
    
    print("=== 图像亮度分析 ===")
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n分析图像: {img_path}")
            brightness, level = calculate_brightness(img_path)
            
            if brightness is not None:
                results.append({
                    'image': img_path,
                    'brightness': brightness,
                    'level': level
                })
                
                # 显示直方图
                analyze_image_histogram(img_path)
        else:
            print(f"文件不存在: {img_path}")
    
    # 打印汇总结果
    print("\n=== 亮度分析汇总 ===")
    for result in results:
        print(f"{result['image']}: {result['brightness']:.2f} - {result['level']}")

# 运行程序
if __name__ == "__main__":
    main()