"""
特征工程模块 - FeatureEngineer

构建机器学习模型所需的多维度特征集：
- 技术指标特征（MA, EMA, RSI, MACD, 布林带, ATR, ADX, CCI, Williams %R）
- 动量特征（多周期收益率、价格动量、成交量动量）
- 波动率特征（历史波动率、Parkinson、Garman-Klass）
- 时间特征（小时/星期/月份周期编码）
- 滞后特征和滚动窗口统计特征
- 预测标签定义

作者: Quant_BTC 项目
"""

import os
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


class FeatureEngineer:
    """
    特征工程器
    
    将原始 OHLCV 数据转换为机器学习模型可用的特征矩阵。
    
    使用示例：
        fe = FeatureEngineer()
        df_features = fe.build_features(df_ohlcv)
        X, y = fe.prepare_ml_data(df_features)
    """
    
    def __init__(self, scaler_type: str = 'standard', models_dir: str = 'models'):
        """
        Args:
            scaler_type: 标准化方法，'standard' 或 'minmax'
            models_dir: 模型/scaler 保存目录
        """
        self.scaler_type = scaler_type
        self.models_dir = models_dir
        self.scaler = None
        self.feature_names = []
        os.makedirs(models_dir, exist_ok=True)
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建完整特征集
        
        Args:
            df: 原始 OHLCV DataFrame（索引为 timestamp）
            
        Returns:
            包含所有特征的 DataFrame
        """
        print("🔧 开始构建特征...")
        df_feat = df.copy()
        
        # 确保只保留 OHLCV 列
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df_feat = df_feat[ohlcv_cols]
        
        # 1. 技术指标特征
        print("  📊 计算技术指标特征...")
        df_feat = self._add_technical_indicators(df_feat)
        
        # 2. 动量特征
        print("  🚀 计算动量特征...")
        df_feat = self._add_momentum_features(df_feat)
        
        # 3. 波动率特征
        print("  📈 计算波动率特征...")
        df_feat = self._add_volatility_features(df_feat)
        
        # 4. 时间特征
        print("  🕐 计算时间特征...")
        df_feat = self._add_time_features(df_feat)
        
        # 5. 滞后特征和滚动窗口统计
        print("  🔄 计算滞后特征和滚动统计...")
        df_feat = self._add_lag_features(df_feat)
        df_feat = self._add_rolling_stats(df_feat)
        
        # 6. 定义预测标签
        print("  🎯 定义预测标签...")
        df_feat = self._add_labels(df_feat)
        
        # 7. 删除含 NaN 的行（由于滚动窗口等计算产生）
        n_before = len(df_feat)
        df_feat.dropna(inplace=True)
        n_after = len(df_feat)
        print(f"  🧹 删除 {n_before - n_after} 行含 NaN 数据，剩余 {n_after} 行")
        
        # 记录特征名称（排除 OHLCV 和标签列）
        exclude_cols = ohlcv_cols + ['label_direction', 'label_magnitude', 'future_return']
        self.feature_names = [col for col in df_feat.columns if col not in exclude_cols]
        
        print(f"  ✅ 特征构建完成，共 {len(self.feature_names)} 个特征")
        
        return df_feat
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征"""
        
        # 移动平均线 (MA)
        for period in [7, 14, 30, 60]:
            df[f'ma_{period}'] = ta.sma(df['close'], length=period)
        
        # 指数移动平均线 (EMA)
        for period in [12, 26]:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        
        # RSI (相对强弱指标)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['macd'] = macd.iloc[:, 0]        # MACD线
            df['macd_signal'] = macd.iloc[:, 1]  # 信号线
            df['macd_hist'] = macd.iloc[:, 2]    # MACD柱
        
        # 布林带
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None:
            df['bb_upper'] = bbands.iloc[:, 2]   # 上轨
            df['bb_middle'] = bbands.iloc[:, 1]  # 中轨
            df['bb_lower'] = bbands.iloc[:, 0]   # 下轨
            # 布林带宽度和位置
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (平均真实波幅)
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_7'] = ta.atr(df['high'], df['low'], df['close'], length=7)
        
        # ADX (平均趋向指标)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx.iloc[:, 0]
            df['dmp'] = adx.iloc[:, 1]  # +DI
            df['dmn'] = adx.iloc[:, 2]  # -DI
        
        # CCI (商品通道指标)
        df['cci_14'] = ta.cci(df['high'], df['low'], df['close'], length=14)
        
        # Williams %R
        df['willr_14'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # 成交量相关
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # 价格相对于均线的位置
        df['price_to_ma7'] = df['close'] / df['ma_7'] - 1
        df['price_to_ma30'] = df['close'] / df['ma_30'] - 1
        df['price_to_ma60'] = df['close'] / df['ma_60'] - 1
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量特征"""
        
        # 多周期收益率
        for period in [1, 3, 7, 14, 30]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # 价格动量（当前价格与N周期前的比值）
        for period in [6, 12, 24]:  # 1天、2天、4天
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # 成交量动量
        for period in [6, 12, 24]:
            df[f'volume_momentum_{period}'] = df['volume'] / df['volume'].shift(period) - 1
        
        # 价格加速度（动量的变化率）
        df['price_acceleration'] = df['return_1'] - df['return_1'].shift(1)
        
        # 高低价比率
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # 收盘价在当日范围中的位置
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['close_position'] = df['close_position'].replace([np.inf, -np.inf], 0.5)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征"""
        
        # 历史波动率（收益率标准差）
        for window in [12, 24, 72]:  # 2天、4天、12天
            df[f'hist_vol_{window}'] = df['close'].pct_change().rolling(window).std()
        
        # Parkinson 波动率（基于高低价）
        for window in [12, 24]:
            log_hl = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(window).mean()
            )
        
        # Garman-Klass 波动率
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        df['gk_vol_24'] = np.sqrt(gk.rolling(24).mean())
        
        # ATR 比率（ATR / 收盘价）
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # 波动率变化率
        df['vol_change'] = df['hist_vol_24'] / df['hist_vol_24'].shift(6) - 1
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算时间特征（正弦余弦周期编码）"""
        
        # 提取时间组件
        hour = df.index.hour
        day_of_week = df.index.dayofweek
        month = df.index.month
        
        # 正弦余弦编码（保持周期性）
        # 小时编码（24小时周期）
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # 星期编码（7天周期）
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # 月份编码（12月周期）
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算滞后特征"""
        
        # 收益率滞后
        returns = df['close'].pct_change()
        for lag in range(1, 6):
            df[f'return_lag_{lag}'] = returns.shift(lag)
        
        # 成交量滞后
        for lag in [1, 2, 3]:
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # RSI 滞后
        if 'rsi_14' in df.columns:
            for lag in [1, 2, 3]:
                df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
        
        return df
    
    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算滚动窗口统计特征"""
        
        returns = df['close'].pct_change()
        
        for window in [12, 24, 48]:  # 2天、4天、8天
            # 滚动均值
            df[f'return_mean_{window}'] = returns.rolling(window).mean()
            # 滚动标准差
            df[f'return_std_{window}'] = returns.rolling(window).std()
            # 滚动偏度
            df[f'return_skew_{window}'] = returns.rolling(window).skew()
            # 滚动峰度
            df[f'return_kurt_{window}'] = returns.rolling(window).kurt()
            # 滚动最大值
            df[f'return_max_{window}'] = returns.rolling(window).max()
            # 滚动最小值
            df[f'return_min_{window}'] = returns.rolling(window).min()
        
        # 成交量滚动统计
        for window in [12, 24]:
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
        
        return df
    
    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        定义预测标签
        
        - label_direction: 二分类（1=涨, 0=跌）
        - label_magnitude: 三分类（2=大涨, 1=小涨/横盘, 0=大跌）
        - future_return: 下一周期的实际收益率
        """
        # 下一周期收益率
        df['future_return'] = df['close'].pct_change().shift(-1)
        
        # 二分类标签：涨=1，跌=0
        df['label_direction'] = (df['future_return'] > 0).astype(int)
        
        # 三分类标签：基于收益率分位数
        # 大涨(>0.5%) = 2, 横盘(-0.5%~0.5%) = 1, 大跌(<-0.5%) = 0
        conditions = [
            df['future_return'] > 0.005,   # 大涨
            df['future_return'] < -0.005,  # 大跌
        ]
        choices = [2, 0]
        df['label_magnitude'] = np.select(conditions, choices, default=1)
        
        return df
    
    def prepare_ml_data(
        self, 
        df: pd.DataFrame, 
        label_col: str = 'label_direction',
        scale: bool = True,
        save_scaler: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备机器学习数据（特征矩阵 X 和标签 y）
        
        Args:
            df: 包含特征和标签的 DataFrame
            label_col: 标签列名
            scale: 是否进行标准化
            save_scaler: 是否保存 scaler
            
        Returns:
            (X, y) 元组
        """
        # 获取特征列
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                       'label_direction', 'label_magnitude', 'future_return',
                       'is_outlier']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df[feature_cols].copy()
        y = df[label_col].copy()
        
        # 标准化
        if scale:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=feature_cols,
                index=X.index
            )
            
            if save_scaler:
                scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
                joblib.dump(self.scaler, scaler_path)
                print(f"  💾 Scaler 已保存至: {scaler_path}")
            
            return X_scaled, y
        
        return X, y
    
    def get_correlation_matrix(self, df: pd.DataFrame, threshold: float = 0.85) -> Tuple[pd.DataFrame, List]:
        """
        计算特征相关性矩阵，识别冗余特征
        
        Args:
            df: 特征 DataFrame
            threshold: 相关性阈值
            
        Returns:
            (相关性矩阵, 冗余特征对列表)
        """
        # 只计算数值特征的相关性
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                       'label_direction', 'label_magnitude', 'future_return',
                       'is_outlier']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        corr_matrix = df[feature_cols].corr()
        
        # 找出高相关性特征对
        redundant_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    redundant_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if redundant_pairs:
            print(f"  ⚠️ 发现 {len(redundant_pairs)} 对高相关性特征（>{threshold}）：")
            for pair in redundant_pairs[:10]:  # 只显示前10对
                print(f"    {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")
            if len(redundant_pairs) > 10:
                print(f"    ... 还有 {len(redundant_pairs)-10} 对")
        else:
            print(f"  ✅ 无高相关性特征对（阈值 {threshold}）")
        
        return corr_matrix, redundant_pairs
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, 
                                 max_features: int = 30) -> go.Figure:
        """
        绘制特征相关性热力图
        
        Args:
            corr_matrix: 相关性矩阵
            max_features: 最多显示的特征数量
            
        Returns:
            Plotly Figure 对象
        """
        # 如果特征太多，只显示前 max_features 个
        if len(corr_matrix) > max_features:
            corr_matrix = corr_matrix.iloc[:max_features, :max_features]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 7},
        ))
        
        fig.update_layout(
            title='特征相关性热力图',
            height=800,
            width=900,
            template='plotly_dark'
        )
        
        return fig
    
    def plot_feature_importance_preview(self, X: pd.DataFrame, y: pd.Series) -> go.Figure:
        """
        使用简单的相关性方法预览特征重要性
        
        Args:
            X: 特征矩阵
            y: 标签
            
        Returns:
            Plotly Figure 对象
        """
        # 计算每个特征与标签的相关性（绝对值）
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # 取前30个
        top_features = correlations.head(30)
        
        fig = go.Figure(data=go.Bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='特征与标签相关性（Top 30）',
            xaxis_title='|相关系数|',
            yaxis_title='特征名称',
            height=600,
            template='plotly_dark',
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def split_time_series(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Tuple:
        """
        按时间顺序划分训练集、验证集、测试集
        
        严格按时间顺序划分，防止未来数据泄露。
        
        Args:
            X: 特征矩阵
            y: 标签
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        print(f"📊 数据集划分（时间序列顺序）：")
        print(f"  训练集: {len(X_train):,} 条 ({X_train.index[0]} ~ {X_train.index[-1]})")
        print(f"  验证集: {len(X_val):,} 条 ({X_val.index[0]} ~ {X_val.index[-1]})")
        print(f"  测试集: {len(X_test):,} 条 ({X_test.index[0]} ~ {X_test.index[-1]})")
        print(f"  标签分布 (训练集): {dict(y_train.value_counts().sort_index())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
