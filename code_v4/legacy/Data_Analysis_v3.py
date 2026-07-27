import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DenoisingAlgorithmAnalyzer:
    def __init__(self, psnr_weight=0.5, ssim_weight=0.5):
        """
        初始化分析器
        
        Args:
            psnr_weight: PSNR权重 (0-1)
            ssim_weight: SSIM权重 (0-1)
        """
        self.df = None
        self.output_dir = None
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        
        # 根据你的CSV文件定义算法列表
        self.algorithms = [
            'Wavelet',
            'Bilateral', 
            'DnCNN',
            'Hybrid V4',
            'Hybrid V1',
            'Hybrid_V2',
            'Traditional Hybrid'
        ]
        
        # 算法显示名称映射（用于图表）
        self.algorithm_names = {
            'Wavelet': '小波去噪',
            'Bilateral': '双边滤波',
            'DnCNN': 'DnCNN',
            'Hybrid V4': '混合V4',
            'Hybrid V1': '混合V1', 
            'Hybrid_V2': '混合V2',
            'Traditional Hybrid': '传统混合'
        }
        
        # 设置绘图样式
        self.setup_plot_style()
        
        # 验证权重
        self._validate_weights()
    
    def _validate_weights(self):
        """验证权重设置"""
        total_weight = self.psnr_weight + self.ssim_weight
        if abs(total_weight - 1.0) > 0.001:
            print(f"⚠️ 权重总和不为1 (PSNR: {self.psnr_weight}, SSIM: {self.ssim_weight})")
            self.psnr_weight = self.psnr_weight / total_weight
            self.ssim_weight = self.ssim_weight / total_weight
            print(f"✅ 已自动归一化: PSNR权重={self.psnr_weight:.3f}, SSIM权重={self.ssim_weight:.3f}")
    
    def setup_plot_style(self):
        """设置绘图样式"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 为每个算法分配颜色
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def create_output_directory(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_info = f"psnr{self.psnr_weight:.2f}_ssim{self.ssim_weight:.2f}"
        self.output_dir = f"algorithm_analysis_{weight_info}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 结果将保存到: {self.output_dir}")
    
    def load_csv_from_current_dir(self):
        """从当前目录加载CSV文件"""
        csv_files = glob.glob("*.csv")
        
        if not csv_files:
            print("❌ 当前目录下没有找到CSV文件")
            return None
        
        print("📄 找到以下CSV文件:")
        for i, file in enumerate(csv_files, 1):
            print(f"  {i}. {file}")
        
        if len(csv_files) == 1:
            selected_file = csv_files[0]
            print(f"✅ 自动选择文件: {selected_file}")
        else:
            choice = input("请选择文件编号: ").strip()
            try:
                selected_file = csv_files[int(choice) - 1]
            except (ValueError, IndexError):
                print("❌ 选择无效，使用第一个文件")
                selected_file = csv_files[0]
        
        return self.load_data(selected_file)
    
    def load_data(self, file_path):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ 成功加载数据: {len(self.df)} 行, {len(self.df.columns)} 列")
            print(f"⚖️  当前权重设置: PSNR={self.psnr_weight:.3f}, SSIM={self.ssim_weight:.3f}")
            
            # 检查算法列是否存在
            available_algorithms = []
            for algo in self.algorithms:
                psnr_col = f"{algo}_psnr"
                if psnr_col in self.df.columns:
                    available_algorithms.append(algo)
            
            self.algorithms = available_algorithms
            print(f"🔍 可用的算法: {', '.join([self.algorithm_names.get(a, a) for a in self.algorithms])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def calculate_composite_score(self, psnr, ssim):
        """计算综合得分"""
        # PSNR归一化到0-1范围（假设最大PSNR为50dB）
        psnr_normalized = np.clip(psnr / 50.0, 0, 1)
        
        # SSIM已经在0-1范围
        ssim_normalized = ssim
        
        # 加权综合得分
        composite_score = (self.psnr_weight * psnr_normalized + 
                          self.ssim_weight * ssim_normalized)
        
        return composite_score
    
    def analyze_best_performance(self):
        """分析每种算法表现最好的次数和百分比"""
        print("\n" + "="*60)
        print("🏆 算法表现最佳次数分析")
        print("="*60)
        
        # 初始化计数器
        best_counts = {algo: 0 for algo in self.algorithms}
        total_valid_cases = 0
        
        # 逐行分析
        for idx, row in self.df.iterrows():
            algorithm_scores = {}
            
            # 计算每个算法的综合得分
            for algo in self.algorithms:
                psnr_col = f"{algo}_psnr"
                ssim_col = f"{algo}_ssim"
                
                if psnr_col in row and ssim_col in row and not pd.isna(row[psnr_col]) and not pd.isna(row[ssim_col]):
                    composite_score = self.calculate_composite_score(row[psnr_col], row[ssim_col])
                    algorithm_scores[algo] = composite_score
            
            # 如果有至少2个算法有数据，找出最佳算法
            if len(algorithm_scores) >= 2:
                best_algo = max(algorithm_scores.items(), key=lambda x: x[1])[0]
                best_counts[best_algo] += 1
                total_valid_cases += 1
        
        # 输出结果
        print(f"有效案例总数: {total_valid_cases}")
        print("\n各算法表现最佳次数和百分比:")
        print("-" * 50)
        
        # 按最佳次数排序
        sorted_counts = sorted(best_counts.items(), key=lambda x: x[1], reverse=True)
        
        for algo, count in sorted_counts:
            if total_valid_cases > 0:
                percentage = (count / total_valid_cases) * 100
                algo_name = self.algorithm_names.get(algo, algo)
                print(f"  {algo_name:15}: {count:3d} 次 ({percentage:5.1f}%)")
            else:
                algo_name = self.algorithm_names.get(algo, algo)
                print(f"  {algo_name:15}: {count:3d} 次 (0.0%)")
        
        # 找出整体最佳算法
        if total_valid_cases > 0:
            overall_best = max(best_counts.items(), key=lambda x: x[1])[0]
            best_percentage = (best_counts[overall_best] / total_valid_cases) * 100
            overall_best_name = self.algorithm_names.get(overall_best, overall_best)
            print(f"\n🎯 整体最佳算法: {overall_best_name} ({best_percentage:.1f}% 的情况下表现最好)")
        
        return best_counts, total_valid_cases
    
    def plot_best_performance_chart(self, best_counts, total_valid_cases):
        """绘制最佳表现次数图表"""
        if total_valid_cases == 0:
            print("⚠️ 没有有效数据绘制图表")
            return
        
        # 准备数据
        algorithms = []
        counts = []
        percentages = []
        
        # 按次数排序
        sorted_counts = sorted(best_counts.items(), key=lambda x: x[1], reverse=True)
        
        for algo, count in sorted_counts:
            algorithms.append(self.algorithm_names.get(algo, algo))
            counts.append(count)
            percentages.append((count / total_valid_cases) * 100)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 柱状图（次数）
        bars1 = ax1.bar(range(len(algorithms)), counts, color=self.colors[:len(algorithms)])
        ax1.set_title('各算法表现最佳次数', fontsize=14, fontweight='bold')
        ax1.set_xlabel('算法')
        ax1.set_ylabel('最佳次数')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 饼图（百分比）
        ax2.pie(counts, labels=algorithms, autopct='%1.1f%%', 
               colors=self.colors[:len(algorithms)], startangle=90)
        ax2.set_title('各算法最佳表现占比', fontsize=14, fontweight='bold')
        
        # 添加权重信息
        weight_text = f"权重: PSNR={self.psnr_weight:.2f}, SSIM={self.ssim_weight:.2f}"
        fig.suptitle(f'算法性能最佳次数分析\n{weight_text}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = f"{self.output_dir}/best_performance_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 图表已保存到: {chart_path}")
    
    def calculate_average_performance(self):
        """计算各算法的平均性能"""
        print("\n" + "="*60)
        print("📊 各算法平均性能分析")
        print("="*60)
        
        performance_data = []
        
        for algo in self.algorithms:
            psnr_col = f"{algo}_psnr"
            ssim_col = f"{algo}_ssim"
            
            if psnr_col in self.df.columns and ssim_col in self.df.columns:
                avg_psnr = self.df[psnr_col].mean()
                avg_ssim = self.df[ssim_col].mean()
                avg_composite = self.calculate_composite_score(avg_psnr, avg_ssim)
                
                performance_data.append({
                    'algorithm': algo,
                    'display_name': self.algorithm_names.get(algo, algo),
                    'avg_psnr': avg_psnr,
                    'avg_ssim': avg_ssim,
                    'avg_composite': avg_composite
                })
        
        # 按综合得分排序
        performance_data.sort(key=lambda x: x['avg_composite'], reverse=True)
        
        print("各算法平均PSNR、SSIM和综合得分:")
        print("-" * 70)
        for data in performance_data:
            print(f"  {data['display_name']:15}: "
                  f"PSNR={data['avg_psnr']:6.2f}dB, "
                  f"SSIM={data['avg_ssim']:6.4f}, "
                  f"综合得分={data['avg_composite']:.4f}")
        
        return performance_data
    
    def plot_average_performance_chart(self, performance_data):
        """绘制平均性能图表"""
        if not performance_data:
            print("⚠️ 没有性能数据绘制图表")
            return
        
        # 准备数据
        algorithms = [data['display_name'] for data in performance_data]
        psnr_values = [data['avg_psnr'] for data in performance_data]
        ssim_values = [data['avg_ssim'] for data in performance_data]
        composite_values = [data['avg_composite'] for data in performance_data]
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 平均PSNR
        bars1 = ax1.bar(range(len(algorithms)), psnr_values, color=self.colors[:len(algorithms)])
        ax1.set_title('平均PSNR', fontsize=12, fontweight='bold')
        ax1.set_xlabel('算法')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars1, psnr_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(psnr_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 平均SSIM
        bars2 = ax2.bar(range(len(algorithms)), ssim_values, color=self.colors[:len(algorithms)])
        ax2.set_title('平均SSIM', fontsize=12, fontweight='bold')
        ax2.set_xlabel('算法')
        ax2.set_ylabel('SSIM')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, ssim_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ssim_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 平均综合得分
        bars3 = ax3.bar(range(len(algorithms)), composite_values, color=self.colors[:len(algorithms)])
        ax3.set_title('平均综合得分', fontsize=12, fontweight='bold')
        ax3.set_xlabel('算法')
        ax3.set_ylabel('综合得分')
        ax3.set_xticks(range(len(algorithms)))
        ax3.set_xticklabels(algorithms, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars3, composite_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(composite_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 添加权重信息
        weight_text = f"权重: PSNR={self.psnr_weight:.2f}, SSIM={self.ssim_weight:.2f}"
        fig.suptitle(f'各算法平均性能对比\n{weight_text}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = f"{self.output_dir}/average_performance_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 平均性能图表已保存到: {chart_path}")
    
    def plot_performance_boxplot(self):
        """绘制性能箱线图"""
        # 准备数据
        psnr_data = []
        ssim_data = []
        composite_data = []
        algorithm_names = []
        
        for algo in self.algorithms:
            psnr_col = f"{algo}_psnr"
            ssim_col = f"{algo}_ssim"
            
            if psnr_col in self.df.columns and ssim_col in self.df.columns:
                # 移除NaN值
                psnr_vals = self.df[psnr_col].dropna().values
                ssim_vals = self.df[ssim_col].dropna().values
                
                if len(psnr_vals) > 0 and len(ssim_vals) > 0:
                    psnr_data.append(psnr_vals)
                    ssim_data.append(ssim_vals)
                    
                    # 计算综合得分
                    composite_vals = []
                    for psnr, ssim in zip(psnr_vals, ssim_vals):
                        composite_vals.append(self.calculate_composite_score(psnr, ssim))
                    composite_data.append(composite_vals)
                    
                    algorithm_names.append(self.algorithm_names.get(algo, algo))
        
        if not psnr_data:
            print("⚠️ 没有足够数据绘制箱线图")
            return
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # PSNR箱线图
        if psnr_data:
            bp1 = ax1.boxplot(psnr_data, labels=algorithm_names, patch_artist=True)
            for i, patch in enumerate(bp1['boxes']):
                patch.set_facecolor(self.colors[i % len(self.colors)])
            ax1.set_title('PSNR分布', fontsize=12, fontweight='bold')
            ax1.set_ylabel('PSNR (dB)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # SSIM箱线图
        if ssim_data:
            bp2 = ax2.boxplot(ssim_data, labels=algorithm_names, patch_artist=True)
            for i, patch in enumerate(bp2['boxes']):
                patch.set_facecolor(self.colors[i % len(self.colors)])
            ax2.set_title('SSIM分布', fontsize=12, fontweight='bold')
            ax2.set_ylabel('SSIM')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # 综合得分箱线图
        if composite_data:
            bp3 = ax3.boxplot(composite_data, labels=algorithm_names, patch_artist=True)
            for i, patch in enumerate(bp3['boxes']):
                patch.set_facecolor(self.colors[i % len(self.colors)])
            ax3.set_title('综合得分分布', fontsize=12, fontweight='bold')
            ax3.set_ylabel('综合得分')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 添加权重信息
        weight_text = f"权重: PSNR={self.psnr_weight:.2f}, SSIM={self.ssim_weight:.2f}"
        fig.suptitle(f'算法性能分布对比\n{weight_text}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = f"{self.output_dir}/performance_boxplot.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 性能分布箱线图已保存到: {chart_path}")
    
    def save_analysis_report(self, best_counts, total_valid_cases, performance_data):
        """保存分析报告"""
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("去噪算法性能分析报告")
        report_lines.append("="*80)
        report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"权重设置: PSNR={self.psnr_weight:.3f}, SSIM={self.ssim_weight:.3f}")
        report_lines.append(f"数据总量: {len(self.df)} 行")
        report_lines.append(f"有效案例: {total_valid_cases} 个")
        report_lines.append("")
        
        # 1. 最佳表现次数分析
        report_lines.append("1. 算法表现最佳次数分析")
        report_lines.append("-" * 60)
        
        sorted_counts = sorted(best_counts.items(), key=lambda x: x[1], reverse=True)
        for algo, count in sorted_counts:
            if total_valid_cases > 0:
                percentage = (count / total_valid_cases) * 100
                algo_name = self.algorithm_names.get(algo, algo)
                report_lines.append(f"  {algo_name:15}: {count:3d} 次 ({percentage:5.1f}%)")
        
        if total_valid_cases > 0:
            overall_best = max(best_counts.items(), key=lambda x: x[1])[0]
            best_percentage = (best_counts[overall_best] / total_valid_cases) * 100
            overall_best_name = self.algorithm_names.get(overall_best, overall_best)
            report_lines.append(f"\n  整体最佳算法: {overall_best_name} ({best_percentage:.1f}% 的情况下表现最好)")
        
        report_lines.append("")
        
        # 2. 平均性能分析
        report_lines.append("2. 各算法平均性能分析")
        report_lines.append("-" * 60)
        
        for data in performance_data:
            report_lines.append(f"  {data['display_name']:15}: "
                              f"PSNR={data['avg_psnr']:6.2f}dB, "
                              f"SSIM={data['avg_ssim']:6.4f}, "
                              f"综合得分={data['avg_composite']:.4f}")
        
        report_lines.append("")
        
        # 3. 噪声图像质量
        report_lines.append("3. 噪声图像质量统计")
        report_lines.append("-" * 60)
        
        if 'noisy_psnr' in self.df.columns:
            avg_noisy_psnr = self.df['noisy_psnr'].mean()
            report_lines.append(f"  噪声图像平均PSNR: {avg_noisy_psnr:.2f} dB")
        
        if 'noisy_ssim' in self.df.columns:
            avg_noisy_ssim = self.df['noisy_ssim'].mean()
            report_lines.append(f"  噪声图像平均SSIM: {avg_noisy_ssim:.4f}")
        
        # 4. 噪声类型统计
        if 'noise_types' in self.df.columns:
            report_lines.append("")
            report_lines.append("4. 噪声类型统计")
            report_lines.append("-" * 60)
            
            noise_counts = self.df['noise_types'].value_counts()
            for noise_type, count in noise_counts.items():
                percentage = (count / len(self.df)) * 100
                report_lines.append(f"  {noise_type}: {count} 次 ({percentage:.1f}%)")
        
        report_lines.append("="*80)
        
        # 保存报告
        report_path = f"{self.output_dir}/analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📝 分析报告已保存到: {report_path}")
        
        # 同时输出到控制台
        print("\n" + "\n".join(report_lines))
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        # 1. 创建输出目录
        self.create_output_directory()
        
        # 2. 分析最佳表现次数
        print("\n" + "="*60)
        print("开始数据分析...")
        print("="*60)
        
        best_counts, total_valid_cases = self.analyze_best_performance()
        
        # 3. 计算平均性能
        performance_data = self.calculate_average_performance()
        
        # 4. 绘制图表
        self.plot_best_performance_chart(best_counts, total_valid_cases)
        self.plot_average_performance_chart(performance_data)
        self.plot_performance_boxplot()
        
        # 5. 保存报告
        self.save_analysis_report(best_counts, total_valid_cases, performance_data)
        
        print(f"\n✅ 所有分析完成！结果保存在: {self.output_dir}")

def get_weight_settings():
    """获取权重设置"""
    print("\n请设置PSNR和SSIM的权重（总和应为1）:")
    
    while True:
        try:
            psnr_input = input("PSNR权重 (默认0.5): ").strip()
            ssim_input = input("SSIM权重 (默认0.5): ").strip()
            
            psnr_weight = float(psnr_input) if psnr_input else 0.5
            ssim_weight = float(ssim_input) if ssim_input else 0.5
            
            total = psnr_weight + ssim_weight
            if abs(total - 1.0) > 0.001:
                print(f"⚠️ 权重总和为 {total:.3f}，不等于1，请重新输入")
                continue
                
            return psnr_weight, ssim_weight
            
        except ValueError:
            print("❌ 请输入有效的数字")

def main():
    """主函数"""
    print("="*60)
    print("去噪算法性能分析工具")
    print("="*60)
    print("功能：")
    print("  1. 分析每种算法表现最好的次数和百分比")
    print("  2. 支持自定义PSNR和SSIM权重")
    print("  3. 生成详细图表和分析报告")
    print("="*60)
    
    # 获取权重设置
    psnr_weight, ssim_weight = get_weight_settings()
    
    # 创建分析器
    analyzer = DenoisingAlgorithmAnalyzer(psnr_weight=psnr_weight, ssim_weight=ssim_weight)
    
    # 加载数据
    if not analyzer.load_csv_from_current_dir():
        print("❌ 数据加载失败，程序退出")
        return
    
    # 运行分析
    analyzer.run_complete_analysis()
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()