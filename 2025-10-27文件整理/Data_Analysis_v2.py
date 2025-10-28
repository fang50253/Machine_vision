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
    def __init__(self, psnr_weight=0.5, ssim_weight=0.5):
        self.df = None
        self.output_dir = None
        self.psnr_weight = psnr_weight  # PSNR权重
        self.ssim_weight = ssim_weight  # SSIM权重
        self.setup_plot_style()
        
        # 验证权重
        self._validate_weights()
    
    def _validate_weights(self):
        """验证权重设置"""
        total_weight = self.psnr_weight + self.ssim_weight
        if abs(total_weight - 1.0) > 0.001:
            print(f"警告: 权重总和不为1 (PSNR: {self.psnr_weight}, SSIM: {self.ssim_weight})")
            # 自动归一化
            self.psnr_weight = self.psnr_weight / total_weight
            self.ssim_weight = self.ssim_weight / total_weight
            print(f"已自动归一化: PSNR权重={self.psnr_weight:.3f}, SSIM权重={self.ssim_weight:.3f}")
    
    def set_weights(self, psnr_weight=None, ssim_weight=None):
        """动态设置权重"""
        if psnr_weight is not None:
            self.psnr_weight = psnr_weight
        if ssim_weight is not None:
            self.ssim_weight = ssim_weight
        self._validate_weights()
        print(f"权重设置更新: PSNR={self.psnr_weight:.3f}, SSIM={self.ssim_weight:.3f}")
    
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
        root.withdraw()
        
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
            print(f"当前权重设置: PSNR={self.psnr_weight:.3f}, SSIM={self.ssim_weight:.3f}")
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def create_output_directory(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_info = f"psnr{self.psnr_weight:.2f}_ssim{self.ssim_weight:.2f}"
        self.output_dir = f"result_{weight_info}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"结果将保存到: {self.output_dir}")
    
    def calculate_composite_score(self, psnr_values, ssim_values):
        """计算综合得分"""
        # 归一化PSNR (假设PSNR在0-50dB范围内)
        psnr_normalized = np.clip(psnr_values / 50.0, 0, 1)
        
        # SSIM已经在0-1范围内
        ssim_normalized = ssim_values
        
        # 加权综合得分
        composite_scores = (self.psnr_weight * psnr_normalized + 
                          self.ssim_weight * ssim_normalized)
        
        return composite_scores
    
    def basic_analysis(self):
        """基础数据分析"""
        print("\n" + "="*50)
        print("基础数据分析")
        print("="*50)
        
        print(f"数据总行数: {len(self.df)}")
        print(f"唯一图像数量: {self.df['image_name'].nunique()}")
        
        if 'noise_type' in self.df.columns:
            print(f"噪声类型: {self.df['noise_type'].unique().tolist()}")
        
        if 'noise_intensities' in self.df.columns:
            print(f"噪声强度范围: {self.df['noise_intensities'].min()} - {self.df['noise_intensities'].max()}")
        
        # 方法列表
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        print(f"\n各方法平均性能 (权重: PSNR={self.psnr_weight:.3f}, SSIM={self.ssim_weight:.3f}):")
        for method in methods:
            psnr_col = f"{method}_psnr"
            ssim_col = f"{method}_ssim"
            
            if psnr_col in self.df.columns and ssim_col in self.df.columns:
                avg_psnr = self.df[psnr_col].mean()
                avg_ssim = self.df[ssim_col].mean()
                
                # 计算综合得分
                composite_score = self.calculate_composite_score(
                    np.array([avg_psnr]), np.array([avg_ssim])
                )[0]
                
                print(f"{method:12}: PSNR={avg_psnr:6.2f}dB, SSIM={avg_ssim:6.4f}, 综合得分={composite_score:.4f}")
    
    def plot_performance_comparison(self):
        """绘制性能比较图"""
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        # 准备数据
        psnr_data = []
        ssim_data = []
        composite_data = []
        
        for method in methods:
            psnr_col = f"{method}_psnr"
            ssim_col = f"{method}_ssim"
            
            if psnr_col in self.df.columns and ssim_col in self.df.columns:
                psnr_values = self.df[psnr_col].values
                ssim_values = self.df[ssim_col].values
                
                psnr_data.append(psnr_values)
                ssim_data.append(ssim_values)
                
                # 计算综合得分
                composite_scores = self.calculate_composite_score(psnr_values, ssim_values)
                composite_data.append(composite_scores)
        
        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # PSNR箱线图
        if psnr_data:
            bp1 = ax1.boxplot(psnr_data, labels=methods, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
            ax1.set_title('PSNR分布比较', fontsize=12, fontweight='bold')
            ax1.set_ylabel('PSNR (dB)')
            ax1.grid(True, alpha=0.3)
        
        # SSIM箱线图
        if ssim_data:
            bp2 = ax2.boxplot(ssim_data, labels=methods, patch_artist=True)
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
            ax2.set_title('SSIM分布比较', fontsize=12, fontweight='bold')
            ax2.set_ylabel('SSIM')
            ax2.grid(True, alpha=0.3)
        
        # 综合得分箱线图
        if composite_data:
            bp3 = ax3.boxplot(composite_data, labels=methods, patch_artist=True)
            for patch, color in zip(bp3['boxes'], colors):
                patch.set_facecolor(color)
            ax3.set_title(f'综合得分分布\n(PSNR权重:{self.psnr_weight:.2f}, SSIM权重:{self.ssim_weight:.2f})', 
                         fontsize=12, fontweight='bold')
            ax3.set_ylabel('综合得分')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weighted_ranking(self):
        """基于权重的排名分析"""
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        # 计算每个测试案例中的加权排名
        ranking_data = []
        composite_scores_data = []
        
        for _, row in self.df.iterrows():
            method_scores = {}
            
            for method in methods:
                psnr_col = f"{method}_psnr"
                ssim_col = f"{method}_ssim"
                
                if psnr_col in row and ssim_col in row and not pd.isna(row[psnr_col]) and not pd.isna(row[ssim_col]):
                    # 计算综合得分
                    composite_score = self.calculate_composite_score(
                        np.array([row[psnr_col]]), np.array([row[ssim_col]])
                    )[0]
                    method_scores[method] = composite_score
            
            if method_scores:
                # 按综合得分排序
                sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
                
                for rank, (method, score) in enumerate(sorted_methods, 1):
                    ranking_data.append({'method': method, 'rank': rank})
                    composite_scores_data.append({'method': method, 'composite_score': score})
        
        ranking_df = pd.DataFrame(ranking_data)
        scores_df = pd.DataFrame(composite_scores_data)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 排名分布图
        if not ranking_df.empty:
            rank_counts = ranking_df.groupby(['method', 'rank']).size().unstack(fill_value=0)
            rank_counts.plot(kind='bar', stacked=True, ax=ax1)
            ax1.set_title('各方法排名分布（基于综合得分）', fontsize=14, fontweight='bold')
            ax1.set_xlabel('去噪方法')
            ax1.set_ylabel('出现次数')
            ax1.legend(title='排名', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 平均综合得分图
        if not scores_df.empty:
            avg_scores = scores_df.groupby('method')['composite_score'].mean().sort_values(ascending=False)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            bars = ax2.bar(avg_scores.index, avg_scores.values, color=colors[:len(avg_scores)])
            ax2.set_title('各方法平均综合得分', fontsize=14, fontweight='bold')
            ax2.set_ylabel('平均综合得分')
            ax2.grid(True, alpha=0.3)
            
            # 在柱子上显示数值
            for bar, value in zip(bars, avg_scores.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/weighted_ranking.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出排名结果
        if not ranking_df.empty:
            avg_ranks = ranking_df.groupby('method')['rank'].mean().sort_values()
            print("\n基于综合得分的方法排名:")
            for method, avg_rank in avg_ranks.items():
                print(f"  {method}: 平均排名 {avg_rank:.2f}")
        
        if not scores_df.empty:
            avg_composite_scores = scores_df.groupby('method')['composite_score'].mean().sort_values(ascending=False)
            print("\n各方法平均综合得分:")
            for method, score in avg_composite_scores.items():
                print(f"  {method}: {score:.4f}")
    
    def plot_noise_intensity_analysis(self):
        """噪声强度分析"""
        if 'noise_intensities' not in self.df.columns:
            return
        
        methods = ['Wavelet', 'Bilateral', 'DnCNN', 'Hybrid_V1', 'Hybrid_V2']
        
        # 按噪声强度分组计算平均性能
        noise_levels = sorted(self.df['noise_intensities'].unique())
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        for method in methods:
            psnr_col = f"{method}_psnr"
            ssim_col = f"{method}_ssim"
            
            if psnr_col in self.df.columns and ssim_col in self.df.columns:
                # PSNR
                psnr_by_noise = self.df.groupby('noise_intensities')[psnr_col].mean()
                ax1.plot(noise_levels, [psnr_by_noise.get(level, 0) for level in noise_levels], 
                        marker='o', label=method, linewidth=2)
                
                # SSIM
                ssim_by_noise = self.df.groupby('noise_intensities')[ssim_col].mean()
                ax2.plot(noise_levels, [ssim_by_noise.get(level, 0) for level in noise_levels], 
                        marker='s', label=method, linewidth=2)
                
                # 综合得分
                composite_scores = []
                for level in noise_levels:
                    level_data = self.df[self.df['noise_intensities'] == level]
                    if not level_data.empty:
                        psnr_val = level_data[psnr_col].mean()
                        ssim_val = level_data[ssim_col].mean()
                        composite_score = self.calculate_composite_score(
                            np.array([psnr_val]), np.array([ssim_val])
                        )[0]
                        composite_scores.append(composite_score)
                    else:
                        composite_scores.append(0)
                
                ax3.plot(noise_levels, composite_scores, marker='^', label=method, linewidth=2)
        
        ax1.set_title('不同噪声强度下的PSNR表现', fontsize=12, fontweight='bold')
        ax1.set_xlabel('噪声强度')
        ax1.set_ylabel('平均PSNR (dB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('不同噪声强度下的SSIM表现', fontsize=12, fontweight='bold')
        ax2.set_xlabel('噪声强度')
        ax2.set_ylabel('平均SSIM')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title(f'不同噪声强度下的综合得分\n(PSNR权重:{self.psnr_weight:.2f}, SSIM权重:{self.ssim_weight:.2f})', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('噪声强度')
        ax3.set_ylabel('平均综合得分')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/noise_intensity_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self):
        """相关性分析"""
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
        report.append("去噪性能分析报告（加权综合评估）")
        report.append("="*60)
        report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"权重设置: PSNR={self.psnr_weight:.3f}, SSIM={self.ssim_weight:.3f}")
        report.append(f"数据总量: {len(self.df)} 行")
        report.append(f"唯一图像: {self.df['image_name'].nunique()} 张")
        report.append("")
        
        # 基于综合得分的最佳方法统计
        best_method_count = {method: 0 for method in methods}
        total_cases = 0
        
        for _, row in self.df.iterrows():
            method_scores = {}
            valid_methods = 0
            
            for method in methods:
                psnr_col = f"{method}_psnr"
                ssim_col = f"{method}_ssim"
                
                if psnr_col in row and ssim_col in row and not pd.isna(row[psnr_col]) and not pd.isna(row[ssim_col]):
                    composite_score = self.calculate_composite_score(
                        np.array([row[psnr_col]]), np.array([row[ssim_col]])
                    )[0]
                    method_scores[method] = composite_score
                    valid_methods += 1
            
            if valid_methods >= 2:  # 至少有两个有效方法才统计
                best_method = max(method_scores.items(), key=lambda x: x[1])[0]
                best_method_count[best_method] += 1
                total_cases += 1
        
        report.append("基于综合得分的最佳方法统计:")
        for method, count in sorted(best_method_count.items(), key=lambda x: x[1], reverse=True):
            if total_cases > 0:
                percentage = (count / total_cases) * 100
                report.append(f"  {method}: {count} 次 ({percentage:.1f}%)")
        
        # 各方法平均性能
        report.append("\n各方法平均性能:")
        for method in methods:
            psnr_col = f"{method}_psnr"
            ssim_col = f"{method}_ssim"
            
            if psnr_col in self.df.columns and ssim_col in self.df.columns:
                avg_psnr = self.df[psnr_col].mean()
                avg_ssim = self.df[ssim_col].mean()
                composite_score = self.calculate_composite_score(
                    np.array([avg_psnr]), np.array([avg_ssim])
                )[0]
                
                report.append(f"  {method}: PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, 综合得分={composite_score:.4f}")
        
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
        self.plot_weighted_ranking()
        self.plot_noise_intensity_analysis()
        self.plot_correlation_analysis()
        self.generate_summary_report()
        
        print(f"\n所有分析完成！结果保存在: {self.output_dir}")
        return True

def get_weight_settings():
    """获取权重设置"""
    print("\n请设置PSNR和SSIM的权重（总和应为1）:")
    
    while True:
        try:
            psnr_weight = float(input("PSNR权重 (默认0.5): ").strip() or "0.5")
            ssim_weight = float(input("SSIM权重 (默认0.5): ").strip() or "0.5")
            
            total = psnr_weight + ssim_weight
            if abs(total - 1.0) > 0.001:
                print(f"权重总和为 {total:.3f}，不等于1，请重新输入")
                continue
                
            return psnr_weight, ssim_weight
            
        except ValueError:
            print("请输入有效的数字")

def main():
    """主函数"""
    print("="*60)
    print("图像去噪性能分析工具（支持权重设置）")
    print("="*60)
    
    # 获取权重设置
    psnr_weight, ssim_weight = get_weight_settings()
    
    # 创建分析器
    analyzer = DenoisingAnalyzer(psnr_weight=psnr_weight, ssim_weight=ssim_weight)
    
    while True:
        print(f"\n当前权重: PSNR={analyzer.psnr_weight:.3f}, SSIM={analyzer.ssim_weight:.3f}")
        print("\n选项:")
        print("1. 选择CSV文件并分析")
        print("2. 手动输入文件路径")
        print("3. 修改权重设置")
        print("4. 退出")
        
        choice = input("请选择操作 (1/2/3/4): ").strip()
        
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
            new_psnr_weight, new_ssim_weight = get_weight_settings()
            analyzer.set_weights(new_psnr_weight, new_ssim_weight)
        
        elif choice == '4':
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