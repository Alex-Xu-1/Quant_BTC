"""
回测结果分析模块 - PerformanceAnalyzer

全面评估策略表现，生成投资建议报告。

功能：
- 核心绩效指标计算（夏普、最大回撤、Calmar等）
- 月度/年度收益分析
- Buy & Hold 基准对比
- 回撤曲线和水下曲线
- 参数敏感性测试
- 模型衰减分析
- 结构化投资建议报告

作者: Quant_BTC 项目
"""

import os
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class PerformanceAnalyzer:
    """
    策略绩效分析器
    
    使用示例：
        analyzer = PerformanceAnalyzer(equity_curve, trades_df, benchmark)
        metrics = analyzer.calculate_metrics()
        analyzer.plot_monthly_heatmap()
        report = analyzer.generate_report()
    """
    
    def __init__(
        self,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        benchmark_curve: Optional[pd.Series] = None,
        initial_capital: float = 10000,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 365 * 6  # 4h K线，每天6根
    ):
        """
        Args:
            equity_curve: 资金曲线 Series（索引为时间戳）
            trades_df: 交易记录 DataFrame
            benchmark_curve: 基准策略资金曲线
            initial_capital: 初始资金
            risk_free_rate: 无风险利率（年化）
            periods_per_year: 每年的交易周期数
        """
        self.equity_curve = equity_curve
        self.trades_df = trades_df
        self.benchmark_curve = benchmark_curve
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # 计算收益率序列
        self.returns = equity_curve.pct_change().dropna()
    
    def calculate_metrics(self) -> Dict:
        """
        计算核心绩效指标
        
        Returns:
            绩效指标字典
        """
        equity = self.equity_curve
        returns = self.returns
        trades = self.trades_df
        
        # 总收益率
        total_return = (equity.iloc[-1] / self.initial_capital) - 1
        
        # 年化收益率
        n_years = len(equity) / self.periods_per_year
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(self.periods_per_year)
        
        # 夏普比率
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self.periods_per_year) \
            if returns.std() > 0 else 0
        
        # 索提诺比率（只考虑下行风险）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std \
            if downside_std > 0 else 0
        
        # 最大回撤
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 最大回撤持续时间
        dd_duration = self._calculate_max_dd_duration(equity)
        
        # Calmar 比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 交易统计
        if len(trades) > 0 and '净盈亏' in trades.columns:
            winning_trades = trades[trades['净盈亏'] > 0]
            losing_trades = trades[trades['净盈亏'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            avg_win = winning_trades['净盈亏'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['净盈亏'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = winning_trades['净盈亏'].sum() / abs(losing_trades['净盈亏'].sum()) \
                if len(losing_trades) > 0 and losing_trades['净盈亏'].sum() != 0 else float('inf')
            avg_holding = trades['持仓周期'].mean() if '持仓周期' in trades.columns else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding = 0
        
        metrics = {
            '总收益率': f"{total_return*100:.2f}%",
            '年化收益率': f"{annual_return*100:.2f}%",
            '年化波动率': f"{annual_volatility*100:.2f}%",
            '夏普比率': f"{sharpe_ratio:.3f}",
            '索提诺比率': f"{sortino_ratio:.3f}",
            '最大回撤': f"{max_drawdown*100:.2f}%",
            '最大回撤持续(天)': f"{dd_duration:.0f}",
            'Calmar比率': f"{calmar_ratio:.3f}",
            '总交易次数': len(trades),
            '胜率': f"{win_rate*100:.1f}%",
            '平均盈利': f"${avg_win:.2f}",
            '平均亏损': f"${avg_loss:.2f}",
            '盈亏比': f"{avg_win/avg_loss:.2f}" if avg_loss > 0 else "N/A",
            '利润因子': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
            '平均持仓周期': f"{avg_holding:.1f}",
        }
        
        # 数值版本（用于后续计算）
        self._metrics_numeric = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
        }
        
        return metrics
    
    def _calculate_max_dd_duration(self, equity: pd.Series) -> float:
        """计算最大回撤持续时间（天）"""
        running_max = equity.cummax()
        is_drawdown = equity < running_max
        
        if not is_drawdown.any():
            return 0
        
        # 找到最长连续回撤期
        dd_groups = (~is_drawdown).cumsum()
        dd_lengths = is_drawdown.groupby(dd_groups).sum()
        max_dd_periods = dd_lengths.max()
        
        # 转换为天数（4h周期，每天6根）
        return max_dd_periods / 6
    
    def plot_monthly_returns_heatmap(self) -> go.Figure:
        """
        绘制月度收益热力图
        
        Returns:
            Plotly Figure 对象
        """
        # 计算月度收益
        equity_monthly = self.equity_curve.resample('ME').last()
        monthly_returns = equity_monthly.pct_change().dropna()
        
        # 构建年-月矩阵
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values * 100
        })
        
        pivot = monthly_df.pivot_table(values='return', index='year', columns='month')
        pivot.columns = ['1月', '2月', '3月', '4月', '5月', '6月',
                        '7月', '8月', '9月', '10月', '11月', '12月'][:len(pivot.columns)]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title='收益率(%)')
        ))
        
        fig.update_layout(
            title='月度收益热力图',
            xaxis_title='月份',
            yaxis_title='年份',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def plot_annual_returns(self) -> go.Figure:
        """
        绘制年度收益柱状图
        
        Returns:
            Plotly Figure 对象
        """
        equity_yearly = self.equity_curve.resample('YE').last()
        yearly_returns = equity_yearly.pct_change().dropna() * 100
        
        colors = ['green' if r > 0 else 'red' for r in yearly_returns.values]
        
        fig = go.Figure(data=go.Bar(
            x=yearly_returns.index.year,
            y=yearly_returns.values,
            marker_color=colors,
            text=[f"{r:.1f}%" for r in yearly_returns.values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='年度收益率',
            xaxis_title='年份',
            yaxis_title='收益率 (%)',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def plot_drawdown_curve(self) -> go.Figure:
        """
        绘制回撤曲线和水下曲线
        
        Returns:
            Plotly Figure 对象
        """
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            mode='lines', name='回撤',
            fill='tozeroy',
            line=dict(color='red', width=1),
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig.update_layout(
            title='水下曲线（Underwater Curve）',
            xaxis_title='时间',
            yaxis_title='回撤 (%)',
            height=350,
            template='plotly_dark'
        )
        
        return fig
    
    def compare_with_benchmark(self) -> go.Figure:
        """
        与 Buy & Hold 基准策略对比
        
        Returns:
            Plotly Figure 对象
        """
        fig = go.Figure()
        
        # 策略收益率曲线
        strategy_returns = (self.equity_curve / self.initial_capital - 1) * 100
        fig.add_trace(go.Scatter(
            x=strategy_returns.index, y=strategy_returns.values,
            mode='lines', name='ML策略',
            line=dict(color='#00d4aa', width=2)
        ))
        
        # 基准收益率曲线
        if self.benchmark_curve is not None:
            benchmark_returns = (self.benchmark_curve / self.initial_capital - 1) * 100
            fig.add_trace(go.Scatter(
                x=benchmark_returns.index, y=benchmark_returns.values,
                mode='lines', name='Buy & Hold',
                line=dict(color='gray', width=1.5, dash='dash')
            ))
        
        fig.add_hline(y=0, line_dash="dot", line_color="yellow", opacity=0.5)
        
        fig.update_layout(
            title='策略 vs Buy & Hold 累计收益率对比',
            xaxis_title='时间',
            yaxis_title='累计收益率 (%)',
            height=450,
            template='plotly_dark'
        )
        
        return fig
    
    def analyze_model_decay(self, predictions_df: pd.DataFrame, 
                            window_size: int = 100) -> go.Figure:
        """
        分析模型预测准确率随时间的衰减
        
        Args:
            predictions_df: 包含 'prediction' 和 'actual' 列的 DataFrame
            window_size: 滚动窗口大小
            
        Returns:
            Plotly Figure 对象
        """
        if 'prediction' not in predictions_df.columns:
            print("  ⚠️ predictions_df 需包含 'prediction' 和 'actual' 列")
            return go.Figure()
        
        # 计算滚动准确率
        correct = (predictions_df['prediction'] == predictions_df['actual']).astype(int)
        rolling_accuracy = correct.rolling(window=window_size).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rolling_accuracy.index, y=rolling_accuracy.values * 100,
            mode='lines', name=f'{window_size}周期滚动准确率',
            line=dict(color='steelblue', width=2)
        ))
        
        fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            title='模型预测准确率随时间变化（衰减分析）',
            xaxis_title='时间',
            yaxis_title='准确率 (%)',
            height=350,
            template='plotly_dark'
        )
        
        return fig
    
    def generate_report(self) -> str:
        """
        生成结构化投资建议报告（Markdown格式）
        
        Returns:
            Markdown 格式的报告字符串
        """
        metrics = self.calculate_metrics()
        numeric = self._metrics_numeric
        
        # 评估策略质量
        quality = self._assess_strategy_quality(numeric)
        
        report = f"""
# 📊 回测结果分析报告

## 一、核心绩效指标

| 指标 | 值 |
|------|-----|
"""
        for key, value in metrics.items():
            report += f"| {key} | {value} |\n"
        
        report += f"""
## 二、策略优势总结

{quality['strengths']}

## 三、主要风险提示

{quality['risks']}

## 四、适用市场环境

{quality['suitable_markets']}

## 五、模型局限性

- 机器学习模型基于历史数据训练，对未来市场结构变化的适应能力有限
- 模型预测准确率会随时间衰减，建议每 1-3 个月重新训练
- 极端行情（黑天鹅事件）下模型可能失效
- 过拟合风险：需持续监控样本外表现
- 数据质量依赖：交易所数据可能存在延迟或错误

## 六、改进方向建议

{quality['improvements']}

## 七、实盘部署建议

1. **先使用 Paper Trading 模式验证至少 1 个月**
2. 初始实盘资金不超过可承受损失的 10%
3. 设置全局最大回撤熔断线（建议 -20%）
4. 定期（每月）重新训练模型并评估性能
5. 监控模型预测准确率，低于 52% 时暂停交易
6. 分散投资，不要将全部资金投入单一策略

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*⚠️ 本报告仅供学习研究参考，不构成任何投资建议*
"""
        return report
    
    def _assess_strategy_quality(self, metrics: Dict) -> Dict:
        """评估策略质量并生成文字建议"""
        strengths = []
        risks = []
        improvements = []
        
        # 夏普比率评估
        if metrics['sharpe_ratio'] > 1.5:
            strengths.append("- 夏普比率优秀（>1.5），风险调整后收益表现突出")
        elif metrics['sharpe_ratio'] > 1.0:
            strengths.append("- 夏普比率良好（>1.0），具有一定的风险调整收益优势")
        else:
            risks.append("- 夏普比率偏低，风险调整后收益不够理想")
            improvements.append("- 优化入场/出场阈值，提升信号质量")
        
        # 最大回撤评估
        if abs(metrics['max_drawdown']) < 0.15:
            strengths.append("- 最大回撤控制良好（<15%），风控机制有效")
        elif abs(metrics['max_drawdown']) < 0.30:
            risks.append("- 最大回撤中等（15%-30%），需关注极端行情下的风险")
        else:
            risks.append("- 最大回撤较大（>30%），建议加强风控或降低仓位")
            improvements.append("- 降低单笔仓位上限或增加止损灵敏度")
        
        # 胜率评估
        if metrics['win_rate'] > 0.55:
            strengths.append(f"- 胜率较高（{metrics['win_rate']*100:.1f}%），信号质量可靠")
        else:
            improvements.append("- 提高模型预测阈值，牺牲交易频率换取更高胜率")
        
        # 年化收益评估
        if metrics['annual_return'] > 0.3:
            strengths.append("- 年化收益率优秀，显著跑赢传统资产")
        elif metrics['annual_return'] > 0:
            strengths.append("- 策略实现正收益")
        else:
            risks.append("- 策略整体亏损，需要重新审视模型和参数")
        
        # 通用建议
        improvements.append("- 尝试加入更多特征（链上数据、市场情绪指标）")
        improvements.append("- 实验不同的模型集成权重和方法（Stacking）")
        improvements.append("- 增加市场状态识别，在不同市场环境使用不同参数")
        
        # 适用市场
        suitable = []
        suitable.append("- ✅ 具有明显趋势的市场（牛市/熊市初期）")
        suitable.append("- ✅ 波动率适中的市场环境")
        suitable.append("- ⚠️ 震荡市场中可能频繁止损")
        suitable.append("- ❌ 极端行情（闪崩/暴涨）下模型可能失效")
        
        return {
            'strengths': '\n'.join(strengths) if strengths else "- 暂无明显优势",
            'risks': '\n'.join(risks) if risks else "- 风险可控",
            'improvements': '\n'.join(improvements),
            'suitable_markets': '\n'.join(suitable)
        }
    
    def save_report(self, path: str = 'results/report.md'):
        """保存报告到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        report = self.generate_report()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  💾 报告已保存至: {path}")
    
    def print_metrics_table(self):
        """打印绩效指标表格"""
        metrics = self.calculate_metrics()
        print("\n📊 策略绩效指标：")
        print("=" * 40)
        for key, value in metrics.items():
            print(f"  {key:15s}: {value}")
        print("=" * 40)
