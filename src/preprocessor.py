"""
数据预处理与探索性分析模块

提供数据清洗、统计分析和可视化功能，用于理解 BTC 价格数据特征。

功能：
- 缺失值处理（前向填充）
- 异常值检测（Z-score / IQR）
- 收益率分布分析
- 平稳性检验（ADF）
- 正态性检验（Jarque-Bera）
- 自相关分析（ACF/PACF）
- 交互式可视化（K线图、成交量图）
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from typing import Tuple, Dict


class DataPreprocessor:
    """
    数据预处理器
    
    对原始 OHLCV 数据进行清洗和质量检查。
    
    使用示例：
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean(df_raw)
        stats = preprocessor.get_summary_stats(df_clean)
    """
    
    def __init__(self, zscore_threshold: float = 4.0, iqr_multiplier: float = 3.0):
        """
        Args:
            zscore_threshold: Z-score 异常值阈值（默认4.0，加密货币波动大）
            iqr_multiplier: IQR 异常值倍数（默认3.0）
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的数据清洗流程
        
        Args:
            df: 原始 OHLCV DataFrame（索引为 timestamp）
            
        Returns:
            清洗后的 DataFrame
        """
        print("🧹 开始数据清洗...")
        df_clean = df.copy()
        
        # 1. 去除重复时间戳
        n_duplicates = df_clean.index.duplicated().sum()
        if n_duplicates > 0:
            df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
            print(f"  ✅ 去除 {n_duplicates} 条重复时间戳")
        else:
            print(f"  ✅ 无重复时间戳")
        
        # 2. 排序
        df_clean.sort_index(inplace=True)
        
        # 3. 处理缺失值
        n_missing_before = df_clean.isnull().sum().sum()
        if n_missing_before > 0:
            # 使用前向填充
            df_clean.fillna(method='ffill', inplace=True)
            # 如果开头有缺失，使用后向填充
            df_clean.fillna(method='bfill', inplace=True)
            print(f"  ✅ 填充 {n_missing_before} 个缺失值（前向填充）")
        else:
            print(f"  ✅ 无缺失值")
        
        # 4. 异常值检测与标记（不删除，仅标记）
        n_outliers = self._detect_outliers(df_clean)
        print(f"  ⚠️ 检测到 {n_outliers} 个潜在异常值（已标记，未删除）")
        
        # 5. 数据完整性检查
        self._check_integrity(df_clean)
        
        print(f"  ✅ 清洗完成，最终数据: {len(df_clean)} 条")
        return df_clean
    
    def _detect_outliers(self, df: pd.DataFrame) -> int:
        """
        使用 Z-score 方法检测异常值
        
        对收益率进行异常值检测（价格本身不适合用Z-score）
        """
        returns = df['close'].pct_change().dropna()
        z_scores = np.abs(stats.zscore(returns))
        n_outliers = (z_scores > self.zscore_threshold).sum()
        
        # 标记异常值
        df['is_outlier'] = False
        outlier_idx = returns[z_scores > self.zscore_threshold].index
        df.loc[outlier_idx, 'is_outlier'] = True
        
        return n_outliers
    
    def _check_integrity(self, df: pd.DataFrame) -> None:
        """检查数据完整性（OHLC 逻辑关系）"""
        # high >= low
        invalid_hl = (df['high'] < df['low']).sum()
        # high >= open and high >= close
        invalid_ho = (df['high'] < df['open']).sum()
        invalid_hc = (df['high'] < df['close']).sum()
        # low <= open and low <= close
        invalid_lo = (df['low'] > df['open']).sum()
        invalid_lc = (df['low'] > df['close']).sum()
        
        total_invalid = invalid_hl + invalid_ho + invalid_hc + invalid_lo + invalid_lc
        if total_invalid > 0:
            print(f"  ⚠️ 发现 {total_invalid} 条 OHLC 逻辑异常数据")
        else:
            print(f"  ✅ OHLC 逻辑关系检查通过")
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        计算关键统计指标
        
        Args:
            df: 清洗后的 DataFrame
            
        Returns:
            统计指标字典
        """
        returns = df['close'].pct_change().dropna()
        
        stats_dict = {
            '数据条数': len(df),
            '时间范围': f"{df.index[0]} ~ {df.index[-1]}",
            '最低价': f"${df['low'].min():,.2f}",
            '最高价': f"${df['high'].max():,.2f}",
            '最新价': f"${df['close'].iloc[-1]:,.2f}",
            '平均日成交量': f"{df['volume'].mean():,.0f}",
            '收益率均值': f"{returns.mean()*100:.4f}%",
            '收益率标准差': f"{returns.std()*100:.4f}%",
            '收益率偏度': f"{returns.skew():.4f}",
            '收益率峰度': f"{returns.kurtosis():.4f}",
            '最大单周期涨幅': f"{returns.max()*100:.2f}%",
            '最大单周期跌幅': f"{returns.min()*100:.2f}%",
            '正收益率占比': f"{(returns > 0).mean()*100:.1f}%",
        }
        
        return stats_dict


class ExploratoryAnalyzer:
    """
    探索性数据分析器
    
    提供统计检验和可视化分析功能。
    
    使用示例：
        analyzer = ExploratoryAnalyzer()
        analyzer.plot_candlestick(df)
        analyzer.test_stationarity(df)
    """
    
    def plot_candlestick(self, df: pd.DataFrame, title: str = "BTC/USDT K线图",
                         last_n: int = 500) -> go.Figure:
        """
        绘制交互式 K 线图和成交量图
        
        Args:
            df: OHLCV DataFrame
            title: 图表标题
            last_n: 显示最近 N 条数据（None 显示全部）
            
        Returns:
            Plotly Figure 对象
        """
        plot_df = df.tail(last_n) if last_n else df
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, '成交量'),
            row_heights=[0.7, 0.3]
        )
        
        # K线图
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df['open'],
                high=plot_df['high'],
                low=plot_df['low'],
                close=plot_df['close'],
                name='K线'
            ),
            row=1, col=1
        )
        
        # 成交量柱状图
        colors = ['red' if row['close'] < row['open'] else 'green' 
                  for _, row in plot_df.iterrows()]
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=plot_df['volume'],
                marker_color=colors,
                name='成交量',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            template='plotly_dark'
        )
        
        return fig
    
    def plot_returns_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        绘制收益率分布直方图（与正态分布对比）
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Plotly Figure 对象
        """
        returns = df['close'].pct_change().dropna()
        
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('收益率分布', 'Q-Q 图'))
        
        # 直方图
        fig.add_trace(
            go.Histogram(
                x=returns, nbinsx=100, name='实际分布',
                opacity=0.7, marker_color='steelblue'
            ),
            row=1, col=1
        )
        
        # 正态分布拟合线
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_pdf = stats.norm.pdf(x_range, returns.mean(), returns.std())
        # 缩放到直方图高度
        bin_width = (returns.max() - returns.min()) / 100
        normal_scaled = normal_pdf * len(returns) * bin_width
        
        fig.add_trace(
            go.Scatter(
                x=x_range, y=normal_scaled, mode='lines',
                name='正态分布', line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Q-Q 图
        sorted_returns = np.sort(returns)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.001, 0.999, len(sorted_returns))
        )
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, y=sorted_returns,
                mode='markers', name='Q-Q',
                marker=dict(size=2, color='steelblue')
            ),
            row=1, col=2
        )
        
        # 参考线
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[sorted_returns.min(), sorted_returns.max()],
                mode='lines', name='参考线',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400, template='plotly_dark',
            title_text='BTC 收益率分布分析（尖峰厚尾特征）'
        )
        
        return fig
    
    def plot_rolling_volatility(self, df: pd.DataFrame, 
                                windows: list = [24, 72, 168]) -> go.Figure:
        """
        绘制滚动波动率曲线
        
        Args:
            df: OHLCV DataFrame
            windows: 滚动窗口列表（默认 24=4天, 72=12天, 168=28天）
            
        Returns:
            Plotly Figure 对象
        """
        returns = df['close'].pct_change().dropna()
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, window in enumerate(windows):
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(365 * 6)  # 年化（4h有6个周期/天）
            days = window * 4 / 24  # 转换为天数
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index, y=rolling_vol,
                    mode='lines', name=f'{days:.0f}天滚动波动率',
                    line=dict(color=colors[i % len(colors)])
                )
            )
        
        fig.update_layout(
            title='BTC 年化滚动波动率',
            yaxis_title='年化波动率',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def plot_acf_pacf(self, df: pd.DataFrame, nlags: int = 40) -> go.Figure:
        """
        绘制自相关函数（ACF）和偏自相关函数（PACF）
        
        Args:
            df: OHLCV DataFrame
            nlags: 滞后阶数
            
        Returns:
            Plotly Figure 对象
        """
        returns = df['close'].pct_change().dropna()
        
        acf_values = acf(returns, nlags=nlags)
        pacf_values = pacf(returns, nlags=nlags)
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('自相关函数 (ACF)', '偏自相关函数 (PACF)'))
        
        # 置信区间
        conf_interval = 1.96 / np.sqrt(len(returns))
        
        # ACF
        fig.add_trace(
            go.Bar(x=list(range(nlags+1)), y=acf_values, 
                   name='ACF', marker_color='steelblue'),
            row=1, col=1
        )
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=1, col=1)
        
        # PACF
        fig.add_trace(
            go.Bar(x=list(range(nlags+1)), y=pacf_values, 
                   name='PACF', marker_color='coral'),
            row=2, col=1
        )
        fig.add_hline(y=conf_interval, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=500, template='plotly_dark', showlegend=False)
        
        return fig
    
    def test_stationarity(self, df: pd.DataFrame) -> Dict:
        """
        ADF 平稳性检验
        
        对价格序列和收益率序列分别进行 ADF 检验。
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            检验结果字典
        """
        results = {}
        
        # 价格序列 ADF 检验
        price_adf = adfuller(df['close'].dropna(), autolag='AIC')
        results['价格序列'] = {
            'ADF统计量': price_adf[0],
            'p值': price_adf[1],
            '滞后阶数': price_adf[2],
            '观测数': price_adf[3],
            '临界值(1%)': price_adf[4]['1%'],
            '临界值(5%)': price_adf[4]['5%'],
            '临界值(10%)': price_adf[4]['10%'],
            '结论': '平稳 ✅' if price_adf[1] < 0.05 else '非平稳 ❌（需差分处理）'
        }
        
        # 收益率序列 ADF 检验
        returns = df['close'].pct_change().dropna()
        returns_adf = adfuller(returns, autolag='AIC')
        results['收益率序列'] = {
            'ADF统计量': returns_adf[0],
            'p值': returns_adf[1],
            '滞后阶数': returns_adf[2],
            '观测数': returns_adf[3],
            '临界值(1%)': returns_adf[4]['1%'],
            '临界值(5%)': returns_adf[4]['5%'],
            '临界值(10%)': returns_adf[4]['10%'],
            '结论': '平稳 ✅' if returns_adf[1] < 0.05 else '非平稳 ❌'
        }
        
        return results
    
    def test_normality(self, df: pd.DataFrame) -> Dict:
        """
        正态性检验（Jarque-Bera 检验）
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            检验结果字典
        """
        returns = df['close'].pct_change().dropna()
        
        # Jarque-Bera 检验
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Shapiro-Wilk 检验（取样本子集，因为该检验对大样本不适用）
        sample_size = min(5000, len(returns))
        sample = returns.sample(sample_size, random_state=42)
        sw_stat, sw_pvalue = stats.shapiro(sample)
        
        results = {
            'Jarque-Bera': {
                '统计量': jb_stat,
                'p值': jb_pvalue,
                '结论': '正态分布 ✅' if jb_pvalue > 0.05 else '非正态分布 ❌（存在尖峰厚尾）'
            },
            'Shapiro-Wilk': {
                '统计量': sw_stat,
                'p值': sw_pvalue,
                '样本量': sample_size,
                '结论': '正态分布 ✅' if sw_pvalue > 0.05 else '非正态分布 ❌'
            },
            '描述性统计': {
                '偏度(Skewness)': returns.skew(),
                '峰度(Kurtosis)': returns.kurtosis(),
                '说明': '正态分布偏度=0，峰度=3；BTC通常呈现负偏、高峰度（厚尾）'
            }
        }
        
        return results
    
    def print_summary_table(self, stats_dict: Dict) -> str:
        """
        生成 Markdown 格式的统计摘要表格
        
        Args:
            stats_dict: 统计指标字典
            
        Returns:
            Markdown 格式的表格字符串
        """
        lines = ["| 指标 | 值 |", "|------|-----|"]
        for key, value in stats_dict.items():
            lines.append(f"| {key} | {value} |")
        
        table = "\n".join(lines)
        print(table)
        return table
