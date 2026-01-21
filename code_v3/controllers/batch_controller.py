import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from collections import Counter

class BatchController:
    """批量处理控制器 - 使用 DenoiseController 完成单张处理并保存结果"""
    def __init__(self):
        self.denoise_controller = None
        self.image_view = None
        self.results_data = []  # 存储所有图像的处理结果
        self.collage_output_dir = None  # 存储拼图输出目录

    def initialize_denoiser(self, model_path: str = None, edge_model_path: str = None) -> bool:
        """
        初始化去噪器，支持传入去噪模型和边缘增强模型路径。
        返回 True 表示初始化成功或至少已创建控制器实例。
        """
        try:
            # 延迟导入，避免循环依赖
            from .denoise_controller import DenoiseController
        except Exception:
            from controllers.denoise_controller import DenoiseController

        if self.denoise_controller is None:
            self.denoise_controller = DenoiseController()

        # 转发到 DenoiseController 的 initialize 方法
        try:
            return self.denoise_controller.initialize_denoiser(model_path, edge_model_path)
        except TypeError:
            # 向后兼容：如果 DenoiseController 只接受一个参数
            return self.denoise_controller.initialize_denoiser(model_path)

    def process_batch(self, folder_path: str, noise_types: List[str], intensities: List[float], 
                      output_root: str = "batch_results", extensions=None):
        """
        批量处理文件夹内图片。
        - folder_path: 待处理目录
        - noise_types, intensities: 传递给 process_single_image 的噪声设置
        - output_root: 输出目录根
        """
        if self.denoise_controller is None:
            raise RuntimeError("去噪器未初始化，请先调用 initialize_denoiser()")

        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        input_path = Path(folder_path)
        if not input_path.exists() or not input_path.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {folder_path}")

        # 收集图像文件
        image_files = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]
        print(f"找到 {len(image_files)} 张图像用于批量处理")

        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_root}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建统一的img文件夹用于存放所有拼图
        self.collage_output_dir = os.path.join(output_dir, 'img')
        os.makedirs(self.collage_output_dir, exist_ok=True)
        
        # 重置结果数据
        self.results_data = []
        
        print(f"\n开始批量处理 {len(image_files)} 张图像...")
        print(f"噪声设置: {noise_types}, 强度: {intensities}")
        print(f"结果将保存到: {output_dir}")
        print(f"所有拼图将保存到: {self.collage_output_dir}")

        # 存储所有拼图信息
        collage_paths = []

        for idx, img_path in enumerate(sorted(image_files), 1):
            try:
                print(f"[{idx}/{len(image_files)}] 处理: {img_path.name}")
                result = self.denoise_controller.process_single_image(str(img_path), noise_types, intensities)
                
                # 为当前图像创建结果字典
                image_result = self._prepare_image_result(img_path, result, noise_types, intensities)
                self.results_data.append(image_result)

                # 输出子目录（按原文件名）
                out_dir = os.path.join(output_dir, 'images', img_path.stem)
                os.makedirs(out_dir, exist_ok=True)

                # 保存原图与噪声图
                cv2.imwrite(os.path.join(out_dir, "original.jpg"), result['original'])
                cv2.imwrite(os.path.join(out_dir, "noisy.jpg"), result['noisy'])

                # 保存各方法结果
                for method, im in result['results'].items():
                    safe_name = method.replace(" ", "_")
                    cv2.imwrite(os.path.join(out_dir, f"{safe_name}.jpg"), im)

                # 保存简单的指标文件
                metrics_path = os.path.join(out_dir, "metrics.txt")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    f.write("去噪指标\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"原图: {img_path.name}\n")
                    f.write(f"图像尺寸: {result['original'].shape[1]}x{result['original'].shape[0]}\n")
                    f.write(f"噪声类型: {result.get('noise_types')}\n")
                    f.write(f"噪声强度: {result.get('intensities')}\n")
                    f.write(f"噪声图 PSNR: {result['metrics'].get('noisy_psnr', 0):.2f} dB\n")
                    f.write(f"噪声图 SSIM: {result['metrics'].get('noisy_ssim', 0):.4f}\n")
                    f.write("\n各方法指标:\n")
                    f.write("-" * 40 + "\n")
                    for method, m in result['metrics']['methods'].items():
                        f.write(f"{method}: PSNR={m.get('psnr',0):.2f} dB, SSIM={m.get('ssim',0):.4f}")
                        if 'normalized_psnr' in m:
                            f.write(f", NormPSNR={m.get('normalized_psnr',0):.3f}")
                        f.write("\n")

                # 创建3x3排列图
                collage_path = self._create_3x3_collage(result, img_path.name)
                if collage_path:
                    collage_paths.append(collage_path)
                
                print(f"✅ 保存结果到: {out_dir}")
            except Exception as e:
                print(f"❌ 处理失败 {img_path.name}: {e}")
                continue
        
        # 创建拼图索引页面
        if collage_paths:
            self._create_collage_index(collage_paths)
        
        # 保存CSV结果文件
        if self.results_data:
            self._save_batch_results(output_dir)
        
        print(f"\n✅ 批量处理完成!")
        print(f"📁 详细结果保存到: {output_dir}")
        print(f"📊 所有拼图保存到: {self.collage_output_dir}")
        return self.results_data

    def _create_3x3_collage(self, result: Dict[str, Any], original_filename: str) -> str:
        """
        创建3x3排列的图片，包含所有处理结果，保存到统一的img文件夹
        
        Args:
            result: 处理结果字典
            original_filename: 原始文件名（用于命名输出文件）
        
        Returns:
            str: 拼图文件路径
        """
        try:
            # 定义3x3网格中每张图片的目标尺寸（固定盒子大小）
            CELL_WIDTH = 400  # 每个单元格的宽度
            CELL_HEIGHT = 300  # 每个单元格的高度
            
            # 定义标题区域高度
            TITLE_HEIGHT = 50
            CELL_TITLE_HEIGHT = 40  # 每个单元格的标题高度
            
            # 整体画布尺寸
            CANVAS_WIDTH = CELL_WIDTH * 3
            CANVAS_HEIGHT = (CELL_HEIGHT + CELL_TITLE_HEIGHT) * 3 + TITLE_HEIGHT + 50
            
            # 创建画布
            canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 240  # 浅灰色背景
            
            # 设置字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            title_font_scale = 1.3
            title_font_thickness = 3
            cell_font_scale = 0.6
            cell_font_thickness = 1
            psnr_font_scale = 0.5
            psnr_font_thickness = 1
            
            # 1. 添加主标题
            main_title = f"Denoising Results: {os.path.splitext(original_filename)[0]}"
            main_title_size = cv2.getTextSize(main_title, font, title_font_scale, title_font_thickness)[0]
            main_title_x = (CANVAS_WIDTH - main_title_size[0]) // 2
            main_title_y = 40
            
            # 绘制主标题背景
            padding = 15
            cv2.rectangle(canvas, 
                        (main_title_x - padding, main_title_y - main_title_size[1] - padding),
                        (main_title_x + main_title_size[0] + padding, main_title_y + padding),
                        (50, 50, 50), -1)
            
            # 绘制主标题文本
            cv2.putText(canvas, main_title, (main_title_x, main_title_y),
                    font, title_font_scale, (255, 255, 255), title_font_thickness)
            
            # 2. 定义9个位置的方法顺序和标题
            positions = [
                # (行, 列, 方法名称, 显示标题)
                (0, 0, 'original', 'Original'),
                (0, 1, 'noisy', 'Noisy'),
                (0, 2, 'DnCNN', 'DnCNN'),
                (1, 0, 'Hybrid V4', 'Hybrid V4'),
                (1, 1, 'Hybrid V1', 'Hybrid V1'),
                (1, 2, 'Hybrid V2', 'Hybrid V2'),
                (2, 0, 'Wavelet', 'Wavelet'),
                (2, 1, 'Bilateral', 'Bilateral'),
                (2, 2, 'Traditional Hybrid', 'Trad Hybrid'),
            ]
            
            # 3. 为每个位置放置图片
            for row, col, method_key, display_name in positions:
                # 获取对应图像
                img = None
                psnr_value = None
                ssim_value = None
                
                if method_key == 'original':
                    img = result.get('original')
                    # 原图没有PSNR/SSIM
                elif method_key == 'noisy':
                    img = result.get('noisy')
                    psnr_value = result['metrics'].get('noisy_psnr', 0)
                    ssim_value = result['metrics'].get('noisy_ssim', 0)
                else:
                    img = result['results'].get(method_key)
                    if method_key in result['metrics']['methods']:
                        method_metrics = result['metrics']['methods'][method_key]
                        psnr_value = method_metrics.get('psnr', 0)
                        ssim_value = method_metrics.get('ssim', 0)
                
                if img is None:
                    # 创建空白图像
                    img = np.zeros((200, 200, 3), dtype=np.uint8) if result.get('original') is None else np.zeros_like(result['original'])
                
                # 计算单元格位置
                cell_x = col * CELL_WIDTH
                cell_y = row * (CELL_HEIGHT + CELL_TITLE_HEIGHT) + TITLE_HEIGHT
                
                # 添加单元格背景
                cv2.rectangle(canvas, 
                            (cell_x, cell_y), 
                            (cell_x + CELL_WIDTH, cell_y + CELL_HEIGHT + CELL_TITLE_HEIGHT),
                            (255, 255, 255), -1)  # 白色背景
                cv2.rectangle(canvas, 
                            (cell_x, cell_y), 
                            (cell_x + CELL_WIDTH, cell_y + CELL_HEIGHT + CELL_TITLE_HEIGHT),
                            (200, 200, 200), 2)  # 灰色边框
                
                # 计算图片在单元格中的位置（保持原比例自适应）
                img_height, img_width = img.shape[:2]
                
                # 计算缩放比例，使图片适应单元格（不拉伸）
                scale_width = CELL_WIDTH / img_width
                scale_height = CELL_HEIGHT / img_height
                scale = min(scale_width, scale_height) * 0.9  # 90%缩放，留出边距
                
                # 计算新尺寸
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # 调整图片尺寸
                resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                # 计算图片在单元格中的居中位置
                img_x = cell_x + (CELL_WIDTH - new_width) // 2
                img_y = cell_y + CELL_TITLE_HEIGHT + (CELL_HEIGHT - new_height) // 2
                
                # 将图片放置到画布上
                canvas[img_y:img_y+new_height, img_x:img_x+new_width] = resized_img
                
                # 添加方法标题
                title_y = cell_y + 25
                title_size = cv2.getTextSize(display_name, font, cell_font_scale, cell_font_thickness)[0]
                title_x = cell_x + (CELL_WIDTH - title_size[0]) // 2
                
                # 添加标题背景
                title_bg_padding = 5
                cv2.rectangle(canvas,
                            (title_x - title_bg_padding, title_y - title_size[1] - title_bg_padding),
                            (title_x + title_size[0] + title_bg_padding, title_y + title_bg_padding),
                            (80, 80, 80), -1)
                
                cv2.putText(canvas, display_name, (title_x, title_y),
                          font, cell_font_scale, (255, 255, 255), cell_font_thickness)
                
                # 添加PSNR/SSIM信息（如果有）
                if psnr_value is not None:
                    psnr_text = f"PSNR: {psnr_value:.2f}dB"
                    ssim_text = f"SSIM: {ssim_value:.4f}"
                    
                    # 计算文本位置
                    text_y = cell_y + CELL_TITLE_HEIGHT + CELL_HEIGHT - 10
                    
                    # 添加PSNR
                    psnr_size = cv2.getTextSize(psnr_text, font, psnr_font_scale, psnr_font_thickness)[0]
                    psnr_x = cell_x + (CELL_WIDTH - psnr_size[0]) // 2
                    
                    cv2.putText(canvas, psnr_text, (psnr_x, text_y),
                              font, psnr_font_scale, (30, 30, 30), psnr_font_thickness)
                    
                    # 添加SSIM
                    text_y -= 15
                    ssim_size = cv2.getTextSize(ssim_text, font, psnr_font_scale, psnr_font_thickness)[0]
                    ssim_x = cell_x + (CELL_WIDTH - ssim_size[0]) // 2
                    
                    cv2.putText(canvas, ssim_text, (ssim_x, text_y),
                              font, psnr_font_scale, (30, 30, 30), psnr_font_thickness)
            
            # 4. 添加网格分隔线
            line_color = (180, 180, 180)
            line_thickness = 1
            
            # 垂直分隔线
            for i in range(1, 3):
                x = i * CELL_WIDTH
                y_start = TITLE_HEIGHT
                y_end = CANVAS_HEIGHT - 50
                cv2.line(canvas, (x, y_start), (x, y_end), line_color, line_thickness)
            
            # 水平分隔线
            for i in range(1, 3):
                y = TITLE_HEIGHT + i * (CELL_HEIGHT + CELL_TITLE_HEIGHT)
                cv2.line(canvas, (0, y), (CANVAS_WIDTH, y), line_color, line_thickness)
            
            # 5. 添加外边框
            cv2.rectangle(canvas, (5, 5), (CANVAS_WIDTH-5, CANVAS_HEIGHT-5), (30, 30, 30), 3)
            
            # 6. 添加页脚信息
            footer_y = CANVAS_HEIGHT - 20
            noise_info = f"Noise: {result.get('noise_types', 'N/A')}, Intensity: {result.get('intensities', 'N/A')}"
            footer_size = cv2.getTextSize(noise_info, font, 0.5, 1)[0]
            footer_x = (CANVAS_WIDTH - footer_size[0]) // 2
            
            cv2.putText(canvas, noise_info, (footer_x, footer_y),
                       font, 0.5, (100, 100, 100), 1)
            
            # 7. 保存图片
            base_name = os.path.splitext(original_filename)[0]
            collage_path = os.path.join(self.collage_output_dir, f"{base_name}_3x3.jpg")
            cv2.imwrite(collage_path, canvas)
            
            # # 8. 同时保存缩略图
            # thumbnail_height = 400
            # thumbnail_width = int(canvas.shape[1] * thumbnail_height / canvas.shape[0])
            # thumbnail = cv2.resize(canvas, (thumbnail_width, thumbnail_height))
            # thumbnail_path = os.path.join(self.collage_output_dir, f"{base_name}_thumb.jpg")
            # cv2.imwrite(thumbnail_path, thumbnail)
            
            print(f"  📊 3x3排列图已保存: {collage_path}")
            
            return collage_path
            
        except Exception as e:
            print(f"  创建3x3排列图失败: {e}")
            return None

    def _create_collage_index(self, collage_paths: List[str]):
        """创建所有拼图的索引页面"""
        try:
            if not collage_paths:
                return
            
            print(f"\n📋 创建拼图索引页面...")
            
            # 加载所有拼图的缩略图
            thumbnails = []
            thumb_names = []
            
            for collage_path in collage_paths:
                # 找到对应的缩略图
                base_name = os.path.basename(collage_path).replace('_3x3.jpg', '')
                thumb_path = os.path.join(self.collage_output_dir, f"{base_name}_thumb.jpg")
                
                if os.path.exists(thumb_path):
                    thumb_img = cv2.imread(thumb_path)
                    if thumb_img is not None:
                        thumbnails.append(thumb_img)
                        thumb_names.append(base_name)
            
            if not thumbnails:
                return
            
            # 创建HTML索引
            html_path = os.path.join(self.collage_output_dir, "index.html")
            self._create_html_index(thumbnails, thumb_names, html_path)
            
            print(f"  🌐 HTML索引已保存: {html_path}")
            
        except Exception as e:
            print(f"  创建索引页面失败: {e}")

    def _create_html_index(self, thumbnails, thumb_names, html_path):
        """创建HTML版本的索引页面"""
        try:
            html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Denoising Results Index</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .title {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            font-size: 18px;
            color: #666;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .item {{
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        .item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}
        .thumb {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
        }}
        .filename {{
            font-weight: bold;
            color: #333;
            margin-top: 10px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #888;
            font-size: 14px;
        }}
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">Denoising Results Index</div>
        <div class="subtitle">Total Images: {len(thumb_names)} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <div class="grid">
'''

            for i, (thumb, name) in enumerate(zip(thumbnails, thumb_names)):
                thumb_filename = f"{name}_thumb.jpg"
                full_filename = f"{name}_3x3.jpg"
                
                html_content += f'''        <div class="item">
            <a href="{full_filename}" target="_blank">
                <img src="{thumb_filename}" alt="{name}" class="thumb">
            </a>
            <div class="filename">{name}</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">
                Click image to view full size
            </div>
        </div>
'''

            html_content += f'''    </div>
    
    <div class="footer">
        Denoising Analysis Results | Generated by Batch Processor
    </div>
</body>
</html>'''

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            print(f"  创建HTML索引失败: {e}")

    def _prepare_image_result(self, img_path: Path, result: Dict[str, Any], 
                            noise_types: List[str], intensities: List[float]) -> Dict[str, Any]:
        """
        准备单张图像的结果字典，用于保存到CSV
        """
        from utils.metrics import normalize_psnr  # 在函数内部导入
        
        image_result = {
            'image_name': img_path.name,
            'image_path': str(img_path),
            'image_size': f"{result['original'].shape[1]}x{result['original'].shape[0]}",
            'noise_types': str(noise_types),
            'noise_intensities': str(intensities),
            'noisy_psnr': result['metrics'].get('noisy_psnr', 0),
            'noisy_ssim': result['metrics'].get('noisy_ssim', 0)
        }
        
        # 收集所有方法的PSNR值用于标准化
        psnr_values = []
        for method, metrics in result['metrics']['methods'].items():
            psnr_values.append(metrics.get('psnr', 0))
        
        # 标准化PSNR
        if psnr_values:
            normalized_psnr = normalize_psnr(psnr_values)
        
        # 添加各方法的指标
        for i, (method, metrics) in enumerate(result['metrics']['methods'].items()):
            image_result[f'{method}_psnr'] = metrics.get('psnr', 0)
            image_result[f'{method}_ssim'] = metrics.get('ssim', 0)
            if psnr_values and i < len(normalized_psnr):
                image_result[f'{method}_psnr_norm'] = normalized_psnr[i]
            else:
                image_result[f'{method}_psnr_norm'] = 0
        
        return image_result
    
    def _save_batch_results(self, output_dir: str):
        """
        保存批量处理结果到CSV文件并显示统计信息
        """
        if not self.results_data:
            print("⚠️ 没有处理结果可保存")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(self.results_data)
        
        # 保存为CSV文件
        csv_path = os.path.join(output_dir, 'denoising_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存为Excel文件（可选）
        try:
            excel_path = os.path.join(output_dir, 'denoising_results.xlsx')
            df.to_excel(excel_path, index=False)
            print(f"📊 Excel文件已保存: {excel_path}")
        except Exception as e:
            print(f"⚠️ 无法保存Excel文件: {e}")
        
        # 显示统计信息
        self._display_statistics(df, csv_path, output_dir)
        
        return csv_path
    
    def _display_statistics(self, df: pd.DataFrame, csv_path: str, output_dir: str):
        """
        显示批量处理的统计信息
        """
        print(f"\n{'='*60}")
        print("批量处理统计信息")
        print(f"{'='*60}")
        print(f"处理图像数量: {len(df)}")
        print(f"CSV结果文件: {csv_path}")
        print(f"图像输出目录: {output_dir}/images/")
        print(f"拼图目录: {output_dir}/img/")
        
        # 显示平均PSNR
        print(f"\n{'='*60}")
        print("各方法平均PSNR:")
        print(f"{'='*60}")
        
        # 检测所有可能的方法列
        method_columns = {}
        for col in df.columns:
            if col.endswith('_psnr') and not col.endswith('psnr_norm'):
                method_name = col.replace('_psnr', '')
                if method_name != 'noisy':
                    method_columns[method_name] = col
        
        # 按字母顺序排序方法名
        sorted_methods = sorted(method_columns.keys())
        
        for method in sorted_methods:
            psnr_col = method_columns[method]
            ssim_col = f'{method}_ssim'
            norm_col = f'{method}_psnr_norm'
            
            # 计算平均值
            avg_psnr = df[psnr_col].mean() if psnr_col in df.columns else 0
            avg_ssim = df[ssim_col].mean() if ssim_col in df.columns else 0
            avg_norm = df[norm_col].mean() if norm_col in df.columns else 0
            
            # 显示结果
            print(f"  {method:20}: {avg_psnr:7.2f} dB | SSIM: {avg_ssim:.4f} | 标准化PSNR: {avg_norm:.3f}")
        
        # 显示噪声图像的PSNR
        print(f"\n噪声图像平均PSNR: {df['noisy_psnr'].mean():.2f} dB")
        print(f"噪声图像平均SSIM: {df['noisy_ssim'].mean():.4f}")
        
        # 显示最佳方法
        print(f"\n{'='*60}")
        print("最佳方法统计:")
        print(f"{'='*60}")
        
        # 找出每张图像的最佳方法
        best_methods = []
        for idx, row in df.iterrows():
            max_psnr = 0
            best_method = ''
            for method in sorted_methods:
                psnr_col = f'{method}_psnr'
                if psnr_col in row and row[psnr_col] > max_psnr:
                    max_psnr = row[psnr_col]
                    best_method = method
            if best_method:
                best_methods.append(best_method)
        
        # 统计最佳方法出现的次数
        best_counts = Counter(best_methods)
        print("各方法成为最佳的次数:")
        for method, count in best_counts.items():
            percentage = (count / len(best_methods)) * 100
            print(f"  {method:20}: {count:3d} 次 ({percentage:.1f}%)")
    
    def get_available_methods(self) -> List[str]:
        """
        获取可用的去噪方法列表
        """
        if not self.results_data or len(self.results_data) == 0:
            return []
        
        methods = []
        for col in self.results_data[0].keys():
            if col.endswith('_psnr') and not col.endswith('psnr_norm'):
                method_name = col.replace('_psnr', '')
                if method_name != 'noisy':
                    methods.append(method_name)
        
        return sorted(methods)










# import os
# import cv2
# from pathlib import Path
# from typing import List
# import pandas as pd
# from datetime import datetime
# from models.traditional_denoiser import AdvancedDenoiser
# from utils.image_utils import add_mixed_noise, generate_random_noise_types, generate_random_intensities
# from utils.metrics import calculate_psnr, calculate_ssim, normalize_psnr
# from views.image_view import ImageView

# class BatchController:
#     """批量处理控制器 - 使用 DenoiseController 完成单张处理并保存结果"""
#     def __init__(self):
#         self.denoise_controller = None
#         self.image_view = ImageView()

#     def initialize_denoiser(self, model_path: str = None, edge_model_path: str = None) -> bool:
#         """
#         初始化去噪器，支持传入去噪模型和边缘增强模型路径。
#         返回 True 表示初始化成功或至少已创建控制器实例。
#         """
#         try:
#             # 延迟导入，避免循环依赖
#             from .denoise_controller import DenoiseController
#         except Exception:
#             from controllers.denoise_controller import DenoiseController

#         if self.denoise_controller is None:
#             self.denoise_controller = DenoiseController()

#         # 转发到 DenoiseController 的 initialize 方法，兼容单参数或双参数调用
#         try:
#             return self.denoise_controller.initialize_denoiser(model_path, edge_model_path)
#         except TypeError:
#             # 向后兼容：如果 DenoiseController 只接受一个参数
#             return self.denoise_controller.initialize_denoiser(model_path)

#     def process_batch(self, folder_path: str, noise_types: List[str], intensities: List[float], 
#                       output_root: str = "batch_results", extensions=None):
#         """
#         批量处理文件夹内图片。
#         - folder_path: 待处理目录
#         - noise_types, intensities: 传递给 process_single_image 的噪声设置
#         - output_root: 输出目录根
#         """
#         if self.denoise_controller is None:
#             raise RuntimeError("去噪器未初始化，请先调用 initialize_denoiser()")

#         if extensions is None:
#             extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

#         input_path = Path(folder_path)
#         if not input_path.exists() or not input_path.is_dir():
#             raise FileNotFoundError(f"输入目录不存在: {folder_path}")

#         # 收集图像文件
#         image_files = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]
#         print(f"找到 {len(image_files)} 张图像用于批量处理")

#         os.makedirs(output_root, exist_ok=True)

#         for idx, img_path in enumerate(sorted(image_files), 1):
#             try:
#                 print(f"\n[{idx}/{len(image_files)}] 处理: {img_path.name}")
#                 result = self.denoise_controller.process_single_image(str(img_path), noise_types, intensities)

#                 # 输出子目录（按原文件名）
#                 out_dir = os.path.join(output_root, img_path.stem)
#                 os.makedirs(out_dir, exist_ok=True)

#                 # 保存原图与噪声图（BGR numpy）
#                 cv2.imwrite(os.path.join(out_dir, "original.jpg"), result['original'])
#                 cv2.imwrite(os.path.join(out_dir, "noisy.jpg"), result['noisy'])

#                 # 保存各方法结果
#                 for method, im in result['results'].items():
#                     safe_name = method.replace(" ", "_")
#                     cv2.imwrite(os.path.join(out_dir, f"{safe_name}.jpg"), im)

#                 # 保存简单的指标文件
#                 metrics_path = os.path.join(out_dir, "metrics.txt")
#                 with open(metrics_path, 'w', encoding='utf-8') as f:
#                     f.write("去噪指标\n")
#                     f.write(f"原图: {img_path.name}\n")
#                     f.write(f"噪声类型: {result.get('noise_types')}\n")
#                     f.write(f"噪声强度: {result.get('intensities')}\n")
#                     f.write(f"噪声图 PSNR: {result['metrics'].get('noisy_psnr', 0):.2f}\n")
#                     # 写入每种方法的PSNR/SSIM
#                     for method, m in result['metrics']['methods'].items():
#                         f.write(f"{method}: PSNR={m.get('psnr',0):.2f}, SSIM={m.get('ssim',0):.3f}\n")

#                 print(f"✅ 保存结果到: {out_dir}")
#             except Exception as e:
#                 print(f"❌ 处理失败 {img_path.name}: {e}")

#         print("\n✅ 批量处理完成")