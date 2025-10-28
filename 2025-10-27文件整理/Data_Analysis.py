import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DenoisingAnalyzer:
    def __init__(self):
        self.df = None
        self.output_dir = None
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """设置绘图样式"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def select_csv_file(self):
        """选择CSV文件"""
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 首先检查当前目录下的CSV文件
        csv_files = glob.glob("*.csv")
        
        if csv_files:
            print("发现当前目录下的CSV文件:")
            for i, file in enumerate(csv_files, 1):
                print(f"{i}. {file}")
            
            choice = input("请选择文件编号 (或输入 'm' 手动选择): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
                selected_file = csv_files[int(choice) - 1]
            else:
                selected_file = filedialog.askopenfilename(
                    title="选择CSV文件",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
        else:
            selected_file = filedialog.askopenfilename(
                title="选择CSV文件",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        
        root.destroy()
        return selected_file
    
    def load_data(self, file_path):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"成功加载数据: {len(self.df)} 行, {len(self.df.columns)} 列")
            print(f"列名: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def create_output_directory(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"result_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"结果将保存到: {self.output_dir}")
    
    def basic_analysis(self):
        """基础数据分析"""
        print("\n" + "="*50)
        print("基础数据分析")
        print("="*50)
        
        # 基本统计信息
        print(f"数据总行数: {len(self.df)}")
        print(f"唯一图像数量: {self.df['image_name'].nunique()}")
        print(f"噪声类型: {self.df['noise_type'].unique().tolist()}")
        print(f"噪声强度范围: {self.df['noise_intensities'].min()} - {self.df['noise_intensities'].max()}")
        
        # 方法列表
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        # 各方法的平均PSNR和SSIM
        print("\n各方法平均性能:")
        for method in methods:
            psnr_col = f"{method}_psnr"
            ssim_col = f"{method}_ssim"
            if psnr_col in self.df.columns and ssim_col in self.df.columns:
                avg_psnr = self.df[psnr_col].mean()
                avg_ssim = self.df[ssim_col].mean()
                print(f"{method}: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
    
    def plot_performance_comparison(self):
        """绘制性能比较图"""
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        # 准备数据
        psnr_data = []
        ssim_data = []
        
        for method in methods:
            psnr_col = f"{method}_psnr"
            ssim_col = f"{method}_ssim"
            
            if psnr_col in self.df.columns:
                psnr_data.append(self.df[psnr_col].values)
            if ssim_col in self.df.columns:
                ssim_data.append(self.df[ssim_col].values)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PSNR箱线图
        if psnr_data:
            bp1 = ax1.boxplot(psnr_data, labels=methods, patch_artist=True)
            # 设置颜色
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
            ax1.set_title('各方法PSNR分布比较', fontsize=14, fontweight='bold')
            ax1.set_ylabel('PSNR (dB)')
            ax1.grid(True, alpha=0.3)
        
        # SSIM箱线图
        if ssim_data:
            bp2 = ax2.boxplot(ssim_data, labels=methods, patch_artist=True)
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
            ax2.set_title('各方法SSIM分布比较', fontsize=14, fontweight='bold')
            ax2.set_ylabel('SSIM')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_noise_intensity_analysis(self):
        """噪声强度分析"""
        if 'noise_intensities' not in self.df.columns:
            return
        
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        # 按噪声强度分组计算平均性能
        noise_levels = sorted(self.df['noise_intensities'].unique())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for method in methods:
            psnr_col = f"{method}_psnr"
            ssim_col = f"{method}_ssim"
            
            if psnr_col in self.df.columns:
                psnr_by_noise = self.df.groupby('noise_intensities')[psnr_col].mean()
                ax1.plot(noise_levels, [psnr_by_noise.get(level, 0) for level in noise_levels], 
                        marker='o', label=method, linewidth=2)
            
            if ssim_col in self.df.columns:
                ssim_by_noise = self.df.groupby('noise_intensities')[ssim_col].mean()
                ax2.plot(noise_levels, [ssim_by_noise.get(level, 0) for level in noise_levels], 
                        marker='s', label=method, linewidth=2)
        
        ax1.set_title('不同噪声强度下的PSNR表现', fontsize=14, fontweight='bold')
        ax1.set_xlabel('噪声强度')
        ax1.set_ylabel('平均PSNR (dB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('不同噪声强度下的SSIM表现', fontsize=14, fontweight='bold')
        ax2.set_xlabel('噪声强度')
        ax2.set_ylabel('平均SSIM')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/noise_intensity_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_method_ranking(self):
        """方法排名分析"""
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        # 计算每个方法在每个测试案例中的排名
        ranking_data = []
        
        for _, row in self.df.iterrows():
            psnr_scores = {}
            for method in methods:
                psnr_col = f"{method}_psnr"
                if psnr_col in row:
                    psnr_scores[method] = row[psnr_col]
            
            # 按PSNR排序
            sorted_methods = sorted(psnr_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (method, score) in enumerate(sorted_methods, 1):
                ranking_data.append({'method': method, 'rank': rank})
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # 计算排名分布
        rank_counts = ranking_df.groupby(['method', 'rank']).size().unstack(fill_value=0)
        
        # 绘制排名分布图
        plt.figure(figsize=(12, 8))
        rank_counts.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('各方法排名分布', fontsize=16, fontweight='bold')
        plt.xlabel('去噪方法')
        plt.ylabel('出现次数')
        plt.legend(title='排名', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/method_ranking.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算平均排名
        avg_ranks = ranking_df.groupby('method')['rank'].mean().sort_values()
        print("\n方法平均排名:")
        for method, avg_rank in avg_ranks.items():
            print(f"{method}: {avg_rank:.2f}")
    
    def plot_correlation_analysis(self):
        """相关性分析"""
        # 选择数值列进行相关性分析
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_cols if any(method in col for method in 
                            ['psnr', 'ssim', 'noise_intensities'])]
        
        if len(correlation_cols) > 1:
            corr_matrix = self.df[correlation_cols].corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('性能指标相关性热力图', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_report(self):
        """生成总结报告"""
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        report = []
        report.append("="*60)
        report.append("去噪性能分析报告")
        report.append("="*60)
        report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据总量: {len(self.df)} 行")
        report.append(f"唯一图像: {self.df['image_name'].nunique()} 张")
        report.append("")
        
        # 最佳方法统计
        best_method_count = {method: 0 for method in methods}
        
        for _, row in self.df.iterrows():
            psnr_scores = {}
            for method in methods:
                psnr_col = f"{method}_psnr"
                if psnr_col in row and not pd.isna(row[psnr_col]):
                    psnr_scores[method] = row[psnr_col]
            
            if psnr_scores:
                best_method = max(psnr_scores.items(), key=lambda x: x[1])[0]
                best_method_count[best_method] += 1
        
        report.append("各方法最佳表现次数:")
        for method, count in sorted(best_method_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.df)) * 100
            report.append(f"  {method}: {count} 次 ({percentage:.1f}%)")
        
        # 保存报告
        report_path = f"{self.output_dir}/analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("\n".join(report))
        print(f"\n详细报告已保存到: {report_path}")
    
    def run_complete_analysis(self, csv_file_path):
        """运行完整分析流程"""
        if not self.load_data(csv_file_path):
            return False
        
        self.create_output_directory()
        
        print("开始数据分析...")
        
        # 执行各项分析
        self.basic_analysis()
        self.plot_performance_comparison()
        self.plot_noise_intensity_analysis()
        self.plot_method_ranking()
        self.plot_correlation_analysis()
        self.generate_summary_report()
        
        print(f"\n所有分析完成！结果保存在: {self.output_dir}")
        return True

def main():
    """主函数"""
    print("="*60)
    print("图像去噪性能分析工具")
    print("="*60)
    
    analyzer = DenoisingAnalyzer()
    
    while True:
        print("\n选项:")
        print("1. 选择CSV文件并分析")
        print("2. 手动输入文件路径")
        print("3. 退出")
        
        choice = input("请选择操作 (1/2/3): ").strip()
        
        if choice == '1':
            csv_file = analyzer.select_csv_file()
            if csv_file:
                analyzer.run_complete_analysis(csv_file)
            else:
                print("未选择文件")
        
        elif choice == '2':
            csv_file = input("请输入CSV文件完整路径: ").strip().strip('"\'')
            if os.path.exists(csv_file):
                analyzer.run_complete_analysis(csv_file)
            else:
                print("文件不存在，请检查路径")
        
        elif choice == '3':
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")
        
        # 询问是否继续
        continue_choice = input("\n是否继续分析其他文件? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("程序结束")
            break

if __name__ == "__main__":
    main()