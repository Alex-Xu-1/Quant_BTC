"""
交易策略模块 - Strategy

包含策略基类和 ML 策略实现：
- BaseStrategy: 策略抽象基类
- MLStrategy: 基于机器学习模型集成的交易策略

策略逻辑：
1. 模型集成：XGBoost + LSTM 加权投票
2. 入场规则：融合概率超过阈值触发信号
3. 出场规则：ATR止盈止损 + 移动止损 + 最大持仓时间
4. 仓位管理：Kelly公式 + ATR动态调整
5. 信号过滤：低波动率环境降低仓位

作者: Quant_BTC 项目
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class Signal:
    """交易信号数据类"""
    timestamp: pd.Timestamp
    direction: int          # 1=做多, -1=做空, 0=无信号
    probability: float      # 模型预测概率
    position_size: float    # 建议仓位比例 (0~1)
    stop_loss: float        # 止损价格
    take_profit: float      # 止盈价格
    atr: float              # 当前 ATR 值


class BaseStrategy(ABC):
    """
    策略抽象基类
    
    所有策略必须继承此类并实现 generate_signal() 方法。
    回测引擎和实盘引擎共享此接口。
    
    使用示例：
        class MyStrategy(BaseStrategy):
            def generate_signal(self, data, index):
                ...
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 策略配置字典
        """
        self.config = config
        self.name = "BaseStrategy"
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """
        生成交易信号
        
        Args:
            data: 包含特征和预测概率的 DataFrame
            index: 当前 K 线索引位置
            
        Returns:
            Signal 对象
        """
        pass
    
    @abstractmethod
    def should_exit(self, data: pd.DataFrame, index: int,
                    entry_price: float, entry_index: int,
                    direction: int, current_stop: float) -> Tuple[bool, str]:
        """
        判断是否应该平仓
        
        Args:
            data: DataFrame
            index: 当前索引
            entry_price: 入场价格
            entry_index: 入场索引
            direction: 持仓方向 (1=多, -1=空)
            current_stop: 当前止损价
            
        Returns:
            (是否平仓, 平仓原因)
        """
        pass


class MLStrategy(BaseStrategy):
    """
    机器学习集成策略
    
    基于 XGBoost + LSTM 模型集成预测信号的交易策略。
    
    决策流程：
    1. 获取模型融合预测概率
    2. 判断是否超过入场阈值
    3. 计算 ATR 确定止盈止损
    4. 使用 Kelly 公式计算仓位
    5. 低波动率环境过滤信号
    
    使用示例：
        strategy = MLStrategy(config=CONFIG)
        signal = strategy.generate_signal(data, current_index)
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 策略配置字典，需包含以下键：
                - long_threshold: 做多阈值
                - short_threshold: 做空阈值
                - stop_loss_atr_mult: 止损ATR倍数
                - take_profit_atr_mult: 止盈ATR倍数
                - max_holding_periods: 最大持仓周期
                - trailing_stop_atr_mult: 移动止损ATR倍数
                - xgb_weight: XGBoost权重
                - lstm_weight: LSTM权重
        """
        super().__init__(config)
        self.name = "ML_Ensemble_Strategy"
        
        # 策略参数
        self.long_threshold = config.get('long_threshold', 0.6)
        self.short_threshold = config.get('short_threshold', 0.4)
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 2.0)
        self.take_profit_atr_mult = config.get('take_profit_atr_mult', 3.0)
        self.max_holding_periods = config.get('max_holding_periods', 30)
        self.trailing_stop_atr_mult = config.get('trailing_stop_atr_mult', 1.5)
        self.xgb_weight = config.get('xgb_weight', 0.5)
        self.lstm_weight = config.get('lstm_weight', 0.5)
        
        # 波动率过滤参数
        self.min_volatility_percentile = config.get('min_volatility_percentile', 20)
    
    def ensemble_predict(self, xgb_proba: float, lstm_proba: float) -> float:
        """
        模型集成预测（加权投票）
        
        Args:
            xgb_proba: XGBoost 预测的正类概率
            lstm_proba: LSTM 预测的正类概率
            
        Returns:
            融合后的概率值
        """
        return self.xgb_weight * xgb_proba + self.lstm_weight * lstm_proba
    
    def calculate_kelly_position(self, win_rate: float, avg_win: float, 
                                  avg_loss: float, max_position: float = 0.3) -> float:
        """
        Kelly 公式计算最优仓位
        
        Kelly% = W - (1-W)/R
        其中 W=胜率, R=盈亏比
        
        Args:
            win_rate: 历史胜率
            avg_win: 平均盈利幅度
            avg_loss: 平均亏损幅度（正值）
            max_position: 最大仓位限制
            
        Returns:
            建议仓位比例 (0~max_position)
        """
        if avg_loss == 0:
            return max_position * 0.5
        
        payoff_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / payoff_ratio
        
        # Kelly 值限制在合理范围内
        kelly = max(0, min(kelly, max_position))
        
        # 使用半 Kelly（更保守）
        return kelly * 0.5
    
    def calculate_atr_position(self, atr: float, close: float, 
                                risk_per_trade: float = 0.02) -> float:
        """
        基于 ATR 的动态仓位计算
        
        仓位 = 风险金额 / (ATR * 止损倍数)
        
        Args:
            atr: 当前 ATR 值
            close: 当前收盘价
            risk_per_trade: 每笔交易风险比例（默认2%）
            
        Returns:
            建议仓位比例
        """
        if atr == 0 or close == 0:
            return 0.1
        
        # 止损距离 = ATR * 倍数
        stop_distance = atr * self.stop_loss_atr_mult
        # 止损距离占价格的比例
        stop_pct = stop_distance / close
        
        if stop_pct == 0:
            return 0.1
        
        # 仓位 = 风险比例 / 止损比例
        position = risk_per_trade / stop_pct
        
        # 限制最大仓位
        return min(position, 0.5)
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> Signal:
        """
        生成交易信号
        
        Args:
            data: DataFrame，需包含列：
                - 'ensemble_proba': 集成模型预测概率
                - 'atr_14': ATR值
                - 'close': 收盘价
                - 'hist_vol_24': 24周期历史波动率
            index: 当前行索引位置
            
        Returns:
            Signal 对象
        """
        row = data.iloc[index]
        timestamp = data.index[index]
        
        # 获取预测概率
        proba = row.get('ensemble_proba', 0.5)
        atr = row.get('atr_14', 0)
        close = row.get('close', 0)
        hist_vol = row.get('hist_vol_24', 0)
        
        # 默认无信号
        direction = 0
        position_size = 0
        stop_loss = 0
        take_profit = 0
        
        # 波动率过滤：如果波动率过低，不交易
        if index > 100:
            vol_series = data['hist_vol_24'].iloc[max(0, index-100):index]
            vol_percentile = (hist_vol > vol_series).mean() * 100
            if vol_percentile < self.min_volatility_percentile:
                return Signal(timestamp, 0, proba, 0, 0, 0, atr)
        
        # 做多信号
        if proba > self.long_threshold:
            direction = 1
            stop_loss = close - atr * self.stop_loss_atr_mult
            take_profit = close + atr * self.take_profit_atr_mult
            position_size = self.calculate_atr_position(atr, close)
        
        # 做空信号
        elif proba < self.short_threshold:
            direction = -1
            stop_loss = close + atr * self.stop_loss_atr_mult
            take_profit = close - atr * self.take_profit_atr_mult
            position_size = self.calculate_atr_position(atr, close)
        
        return Signal(timestamp, direction, proba, position_size, 
                     stop_loss, take_profit, atr)
    
    def should_exit(self, data: pd.DataFrame, index: int,
                    entry_price: float, entry_index: int,
                    direction: int, current_stop: float) -> Tuple[bool, str]:
        """
        判断是否应该平仓
        
        平仓条件：
        1. 触发止损
        2. 触发止盈
        3. 超过最大持仓时间
        4. 移动止损被触发
        
        Args:
            data: DataFrame
            index: 当前索引
            entry_price: 入场价格
            entry_index: 入场索引
            direction: 持仓方向
            current_stop: 当前止损价
            
        Returns:
            (是否平仓, 平仓原因)
        """
        row = data.iloc[index]
        close = row['close']
        high = row['high']
        low = row['low']
        atr = row.get('atr_14', 0)
        
        # 1. 止损检查
        if direction == 1:  # 多头
            if low <= current_stop:
                return True, "止损"
        elif direction == -1:  # 空头
            if high >= current_stop:
                return True, "止损"
        
        # 2. 止盈检查
        take_profit_distance = atr * self.take_profit_atr_mult
        if direction == 1:
            if high >= entry_price + take_profit_distance:
                return True, "止盈"
        elif direction == -1:
            if low <= entry_price - take_profit_distance:
                return True, "止盈"
        
        # 3. 最大持仓时间
        holding_periods = index - entry_index
        if holding_periods >= self.max_holding_periods:
            return True, "超时平仓"
        
        # 4. 不平仓
        return False, ""
    
    def update_trailing_stop(self, data: pd.DataFrame, index: int,
                             entry_price: float, direction: int,
                             current_stop: float) -> float:
        """
        更新移动止损
        
        Args:
            data: DataFrame
            index: 当前索引
            entry_price: 入场价格
            direction: 持仓方向
            current_stop: 当前止损价
            
        Returns:
            更新后的止损价
        """
        row = data.iloc[index]
        close = row['close']
        atr = row.get('atr_14', 0)
        trailing_distance = atr * self.trailing_stop_atr_mult
        
        if direction == 1:  # 多头
            # 只有盈利时才移动止损
            if close > entry_price:
                new_stop = close - trailing_distance
                return max(current_stop, new_stop)
        elif direction == -1:  # 空头
            if close < entry_price:
                new_stop = close + trailing_distance
                return min(current_stop, new_stop)
        
        return current_stop
    
    def get_strategy_description(self) -> str:
        """返回策略的文字描述"""
        desc = f"""
## {self.name} 策略说明

### 模型集成
- XGBoost 权重: {self.xgb_weight}
- LSTM 权重: {self.lstm_weight}
- 融合方式: 加权投票

### 入场规则
- 做多: 融合概率 > {self.long_threshold}
- 做空: 融合概率 < {self.short_threshold}
- 波动率过滤: 当前波动率需高于历史 {self.min_volatility_percentile}% 分位

### 出场规则
- 止损: {self.stop_loss_atr_mult} × ATR
- 止盈: {self.take_profit_atr_mult} × ATR
- 移动止损: {self.trailing_stop_atr_mult} × ATR（盈利后启动）
- 最大持仓: {self.max_holding_periods} 个周期

### 仓位管理
- 基于 ATR 动态计算仓位
- 每笔交易风险: 2% 资金
- 最大单笔仓位: 50%
"""
        return desc
