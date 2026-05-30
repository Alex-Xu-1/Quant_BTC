"""
回测引擎 - Backtester

事件驱动型回测引擎，逐K线模拟策略执行。
支持交易成本（手续费+滑点）、详细交易记录和资金曲线生成。

作者: Quant_BTC 项目
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from strategy import BaseStrategy, Signal


@dataclass
class Trade:
    """单笔交易记录"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    direction: int = 0              # 1=多, -1=空
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0           # 交易数量（BTC）
    position_size: float = 0.0     # 仓位比例
    commission: float = 0.0         # 总手续费
    slippage_cost: float = 0.0     # 滑点成本
    pnl: float = 0.0               # 净盈亏
    pnl_pct: float = 0.0           # 盈亏百分比
    exit_reason: str = ""           # 平仓原因
    holding_periods: int = 0        # 持仓周期数


class Backtester:
    """
    事件驱动型回测引擎
    
    逐K线遍历历史数据，模拟策略执行过程。
    
    功能：
    - 逐K线模拟执行
    - 交易成本模拟（手续费 + 滑点）
    - 详细交易记录
    - 资金曲线生成
    - 极端行情标注
    
    使用示例：
        backtester = Backtester(strategy, config)
        results = backtester.run(data)
        backtester.plot_equity_curve()
    """
    
    def __init__(self, strategy: BaseStrategy, config: Dict):
        """
        Args:
            strategy: 策略实例（继承 BaseStrategy）
            config: 回测配置字典，需包含：
                - initial_capital: 初始资金
                - commission_rate: 手续费率
                - slippage_rate: 滑点率
        """
        self.strategy = strategy
        self.config = config
        
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission_rate = config.get('commission_rate', 0.001)
        self.slippage_rate = config.get('slippage_rate', 0.0005)
        
        # 回测状态
        self.capital = self.initial_capital
        self.position = 0           # 当前持仓数量
        self.position_direction = 0  # 持仓方向
        self.entry_price = 0
        self.entry_index = 0
        self.current_stop = 0
        
        # 记录
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        self.extreme_events: List[Dict] = []
        
    def run(self, data: pd.DataFrame) -> Dict:
        """
        执行回测
        
        Args:
            data: 包含 OHLCV + 特征 + 预测概率的 DataFrame
                  需包含列: open, high, low, close, volume, ensemble_proba, atr_14
                  
        Returns:
            回测结果字典
        """
        print(f"🔄 开始回测...")
        print(f"   策略: {self.strategy.name}")
        print(f"   数据范围: {data.index[0]} ~ {data.index[-1]}")
        print(f"   数据条数: {len(data):,}")
        print(f"   初始资金: {self.initial_capital:,.2f} USDT")
        
        # 重置状态
        self._reset()
        
        # 逐K线遍历
        for i in tqdm(range(len(data)), desc="   回测进度", unit="bar"):
            row = data.iloc[i]
            timestamp = data.index[i]
            close = row['close']
            
            # 记录时间戳
            self.timestamps.append(timestamp)
            
            # 检测极端行情
            self._check_extreme_event(data, i)
            
            # 如果有持仓，检查是否需要平仓
            if self.position != 0:
                # 更新移动止损
                self.current_stop = self.strategy.update_trailing_stop(
                    data, i, self.entry_price, self.position_direction, self.current_stop
                )
                
                # 检查平仓条件
                should_exit, reason = self.strategy.should_exit(
                    data, i, self.entry_price, self.entry_index,
                    self.position_direction, self.current_stop
                )
                
                if should_exit:
                    self._close_position(data, i, reason)
            
            # 如果无持仓，检查是否有入场信号
            if self.position == 0:
                signal = self.strategy.generate_signal(data, i)
                if signal.direction != 0:
                    self._open_position(data, i, signal)
            
            # 记录当前权益
            current_equity = self._calculate_equity(close)
            self.equity_curve.append(current_equity)
        
        # 如果回测结束时仍有持仓，强制平仓
        if self.position != 0:
            self._close_position(data, len(data) - 1, "回测结束")
        
        # 生成结果
        results = self._generate_results(data)
        
        print(f"\n   ✅ 回测完成！")
        print(f"   最终权益: {self.equity_curve[-1]:,.2f} USDT")
        print(f"   总收益率: {(self.equity_curve[-1]/self.initial_capital - 1)*100:.2f}%")
        print(f"   交易次数: {len(self.trades)}")
        
        return results
    
    def _reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.position = 0
        self.position_direction = 0
        self.entry_price = 0
        self.entry_index = 0
        self.current_stop = 0
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        self.extreme_events = []
    
    def _calculate_equity(self, current_price: float) -> float:
        """计算当前总权益"""
        if self.position == 0:
            return self.capital
        
        # 未实现盈亏
        if self.position_direction == 1:  # 多头
            unrealized_pnl = (current_price - self.entry_price) * self.position
        else:  # 空头
            unrealized_pnl = (self.entry_price - current_price) * self.position
        
        return self.capital + unrealized_pnl
    
    def _open_position(self, data: pd.DataFrame, index: int, signal: Signal):
        """
        开仓
        
        Args:
            data: DataFrame
            index: 当前索引
            signal: 交易信号
        """
        row = data.iloc[index]
        close = row['close']
        
        # 计算滑点后的入场价格
        if signal.direction == 1:  # 做多，价格上滑
            entry_price = close * (1 + self.slippage_rate)
        else:  # 做空，价格下滑
            entry_price = close * (1 - self.slippage_rate)
        
        # 计算仓位大小
        position_value = self.capital * signal.position_size
        quantity = position_value / entry_price
        
        # 计算开仓手续费
        commission = position_value * self.commission_rate
        
        # 更新状态
        self.position = quantity
        self.position_direction = signal.direction
        self.entry_price = entry_price
        self.entry_index = index
        self.current_stop = signal.stop_loss
        self.capital -= commission  # 扣除手续费
        
        # 创建交易记录（平仓时补充）
        trade = Trade(
            entry_time=data.index[index],
            direction=signal.direction,
            entry_price=entry_price,
            quantity=quantity,
            position_size=signal.position_size,
            commission=commission
        )
        self.trades.append(trade)
    
    def _close_position(self, data: pd.DataFrame, index: int, reason: str):
        """
        平仓
        
        Args:
            data: DataFrame
            index: 当前索引
            reason: 平仓原因
        """
        row = data.iloc[index]
        close = row['close']
        
        # 计算滑点后的出场价格
        if self.position_direction == 1:  # 多头平仓，价格下滑
            exit_price = close * (1 - self.slippage_rate)
        else:  # 空头平仓，价格上滑
            exit_price = close * (1 + self.slippage_rate)
        
        # 计算盈亏
        if self.position_direction == 1:
            pnl = (exit_price - self.entry_price) * self.position
        else:
            pnl = (self.entry_price - exit_price) * self.position
        
        # 计算平仓手续费
        position_value = exit_price * self.position
        close_commission = position_value * self.commission_rate
        
        # 净盈亏
        net_pnl = pnl - close_commission
        
        # 更新资金
        self.capital += net_pnl + self.entry_price * self.position  # 归还本金 + 盈亏
        # 修正：实际上 capital 在开仓时只扣了手续费，持仓期间 capital 不变
        # 平仓时：capital += 持仓盈亏 - 平仓手续费
        self.capital = self.capital + pnl - close_commission
        
        # 更新最后一笔交易记录
        trade = self.trades[-1]
        trade.exit_time = data.index[index]
        trade.exit_price = exit_price
        trade.commission += close_commission
        trade.slippage_cost = abs(close - exit_price) * self.position + \
                             abs(data.iloc[self.entry_index]['close'] - self.entry_price) * self.position
        trade.pnl = net_pnl
        trade.pnl_pct = net_pnl / (self.entry_price * self.position) if self.position > 0 else 0
        trade.exit_reason = reason
        trade.holding_periods = index - self.entry_index
        
        # 重置持仓状态
        self.position = 0
        self.position_direction = 0
        self.entry_price = 0
        self.entry_index = 0
        self.current_stop = 0
    
    def _check_extreme_event(self, data: pd.DataFrame, index: int):
        """检测极端行情事件（单日跌幅>15%）"""
        if index < 6:  # 至少需要6个4h周期（1天）
            return
        
        # 计算过去24小时（6个4h周期）的涨跌幅
        current_close = data.iloc[index]['close']
        prev_close = data.iloc[index - 6]['close']
        daily_return = (current_close - prev_close) / prev_close
        
        if daily_return < -0.15:  # 跌幅超过15%
            self.extreme_events.append({
                'timestamp': data.index[index],
                'return': daily_return,
                'price': current_close,
                'type': '极端下跌'
            })
    
    def _generate_results(self, data: pd.DataFrame) -> Dict:
        """生成回测结果"""
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        
        # 交易记录 DataFrame
        trades_df = pd.DataFrame([{
            '入场时间': t.entry_time,
            '出场时间': t.exit_time,
            '方向': '多' if t.direction == 1 else '空',
            '入场价': t.entry_price,
            '出场价': t.exit_price,
            '数量': t.quantity,
            '仓位比例': t.position_size,
            '手续费': t.commission,
            '净盈亏': t.pnl,
            '盈亏%': t.pnl_pct * 100,
            '持仓周期': t.holding_periods,
            '平仓原因': t.exit_reason
        } for t in self.trades if t.exit_time is not None])
        
        results = {
            'equity_curve': equity_series,
            'trades': trades_df,
            'final_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital,
            'total_return': (self.equity_curve[-1] / self.initial_capital - 1) if self.equity_curve else 0,
            'n_trades': len([t for t in self.trades if t.exit_time is not None]),
            'extreme_events': self.extreme_events,
        }
        
        return results
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录 DataFrame"""
        return pd.DataFrame([{
            '入场时间': t.entry_time,
            '出场时间': t.exit_time,
            '方向': '多' if t.direction == 1 else '空',
            '入场价': round(t.entry_price, 2),
            '出场价': round(t.exit_price, 2),
            '数量(BTC)': round(t.quantity, 6),
            '手续费': round(t.commission, 2),
            '净盈亏': round(t.pnl, 2),
            '盈亏%': round(t.pnl_pct * 100, 2),
            '持仓周期': t.holding_periods,
            '平仓原因': t.exit_reason
        } for t in self.trades if t.exit_time is not None])
    
    def save_trades(self, path: str = 'results/trades.csv'):
        """保存交易记录为 CSV"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trades_df = self.get_trades_df()
        trades_df.to_csv(path, index=False)
        print(f"  💾 交易记录已保存至: {path}")
    
    def plot_equity_curve(self, benchmark: Optional[pd.Series] = None) -> go.Figure:
        """
        绘制交互式资金曲线
        
        Args:
            benchmark: 基准策略资金曲线（可选）
            
        Returns:
            Plotly Figure 对象
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('资金曲线', '回撤曲线'),
            row_heights=[0.7, 0.3]
        )
        
        # 资金曲线
        equity = pd.Series(self.equity_curve, index=self.timestamps)
        fig.add_trace(
            go.Scatter(
                x=equity.index, y=equity.values,
                mode='lines', name='策略资金',
                line=dict(color='#00d4aa', width=2)
            ),
            row=1, col=1
        )
        
        # 基准线
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index, y=benchmark.values,
                    mode='lines', name='Buy & Hold',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # 初始资金线
        fig.add_hline(
            y=self.initial_capital, line_dash="dot", 
            line_color="yellow", opacity=0.5, row=1, col=1
        )
        
        # 标注极端事件
        for event in self.extreme_events:
            fig.add_vline(
                x=event['timestamp'], line_dash="dash",
                line_color="red", opacity=0.3, row=1, col=1
            )
        
        # 回撤曲线
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown.values,
                mode='lines', name='回撤',
                fill='tozeroy',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{self.strategy.name} 回测资金曲线',
            height=600,
            template='plotly_dark',
            showlegend=True
        )
        
        fig.update_yaxes(title_text='资金 (USDT)', row=1, col=1)
        fig.update_yaxes(title_text='回撤 (%)', row=2, col=1)
        
        return fig
