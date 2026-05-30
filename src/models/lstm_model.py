"""
LSTM 模型 - LSTMModel

基于 LSTM 长短期记忆网络的时序预测模型。
通过滑动窗口构造序列输入，捕捉价格序列中的长期依赖关系。

作者: Quant_BTC 项目
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.base_model import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM 时序分类模型
    
    用于预测下一周期 BTC 价格方向（涨/跌）。
    通过滑动窗口将特征序列转换为 3D 输入 (samples, timesteps, features)。
    
    特点：
    - 捕捉时间序列中的长期依赖关系
    - 适合处理非线性时序模式
    - Dropout + BatchNorm 防止过拟合
    - EarlyStopping 自动停止训练
    
    使用示例：
        model = LSTMModel(window_size=24)
        model.train(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
    """
    
    def __init__(
        self, 
        window_size: int = 24,
        models_dir: str = 'models',
        lstm_units: list = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100
    ):
        """
        Args:
            window_size: 滑动窗口大小（默认24个4h周期=4天）
            models_dir: 模型保存目录
            lstm_units: LSTM 层单元数列表（默认 [128, 64]）
            dropout_rate: Dropout 比率
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 最大训练轮数
        """
        super().__init__(name='LSTM', models_dir=models_dir)
        
        self.window_size = window_size
        self.lstm_units = lstm_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_features = None
        self.history = None
    
    def _build_model(self, n_features: int) -> Sequential:
        """
        构建 LSTM 网络架构
        
        架构：Input → LSTM(128) → Dropout → BatchNorm → 
              LSTM(64) → Dropout → Dense(32) → Output(1, sigmoid)
        
        Args:
            n_features: 输入特征数量
            
        Returns:
            编译好的 Keras 模型
        """
        model = Sequential([
            Input(shape=(self.window_size, n_features)),
            
            # 第一层 LSTM
            LSTM(self.lstm_units[0], return_sequences=True),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            # 第二层 LSTM
            LSTM(self.lstm_units[1], return_sequences=False),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            # 全连接层
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate / 2),
            
            # 输出层（二分类）
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple:
        """
        使用滑动窗口创建序列数据
        
        将 2D 数据 (samples, features) 转换为 3D (samples, timesteps, features)
        
        Args:
            X: 特征矩阵
            y: 标签（可选）
            
        Returns:
            (X_seq, y_seq) 或 (X_seq,) 如果 y 为 None
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        X_seq = []
        y_seq = []
        
        for i in range(self.window_size, len(X_values)):
            X_seq.append(X_values[i - self.window_size:i])
            if y is not None:
                y_values = y.values if isinstance(y, pd.Series) else y
                y_seq.append(y_values[i])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq,
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        训练 LSTM 模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练历史字典
        """
        print(f"🚀 开始训练 {self.name} 模型...")
        self.n_features = X_train.shape[1]
        print(f"   窗口大小: {self.window_size}, 特征数: {self.n_features}")
        
        # 创建序列数据
        print("   📐 构造滑动窗口序列...")
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train)
        print(f"   训练序列形状: {X_train_seq.shape}")
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
            print(f"   验证序列形状: {X_val_seq.shape}")
        
        # 构建模型
        self.model = self._build_model(self.n_features)
        self.model.summary()
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
        ]
        
        # 模型检查点
        checkpoint_path = os.path.join(self.models_dir, 'lstm_checkpoint.keras')
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True,
                verbose=0
            )
        )
        
        # 训练
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # 记录训练历史
        history_dict = {
            'train_loss': self.history.history['loss'],
            'train_accuracy': self.history.history['accuracy'],
        }
        if validation_data:
            history_dict['val_loss'] = self.history.history['val_loss']
            history_dict['val_accuracy'] = self.history.history['val_accuracy']
        
        # 计算 AUC
        train_proba = self.model.predict(X_train_seq, verbose=0).flatten()
        train_auc = roc_auc_score(y_train_seq, train_proba)
        history_dict['train_auc'] = train_auc
        
        print(f"\n   ✅ 训练完成！")
        print(f"   训练集 AUC: {train_auc:.4f}")
        
        if validation_data:
            val_proba = self.model.predict(X_val_seq, verbose=0).flatten()
            val_auc = roc_auc_score(y_val_seq, val_proba)
            history_dict['val_auc'] = val_auc
            print(f"   验证集 AUC: {val_auc:.4f}")
        
        self.training_history = history_dict
        return history_dict
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_seq = self._create_sequences(X)[0]
        proba = self.model.predict(X_seq, verbose=0).flatten()
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类（涨）概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_seq = self._create_sequences(X)[0]
        return self.model.predict(X_seq, verbose=0).flatten()
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        评估模型性能（重写基类方法以处理序列化）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_seq, y_seq = self._create_sequences(X, y)
        proba = self.model.predict(X_seq, verbose=0).flatten()
        pred = (proba > 0.5).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_seq, pred),
            'precision': precision_score(y_seq, pred, zero_division=0),
            'recall': recall_score(y_seq, pred, zero_division=0),
            'f1': f1_score(y_seq, pred, zero_division=0),
        }
        
        try:
            metrics['auc'] = roc_auc_score(y_seq, proba)
        except ValueError:
            metrics['auc'] = 0.5
        
        return metrics
    
    def save(self, path: Optional[str] = None) -> str:
        """保存模型"""
        if path is None:
            path = os.path.join(self.models_dir, 'lstm_model.keras')
        self.model.save(path)
        print(f"  💾 模型已保存至: {path}")
        return path
    
    def load(self, path: str) -> None:
        """加载模型"""
        self.model = load_model(path)
        self.is_trained = True
        print(f"  📂 模型已加载: {path}")
    
    def plot_training_history(self) -> go.Figure:
        """
        绘制训练/验证损失和准确率曲线
        
        Returns:
            Plotly Figure 对象
        """
        if self.history is None:
            raise ValueError("模型尚未训练，无训练历史")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('损失曲线', '准确率曲线')
        )
        
        epochs = list(range(1, len(self.training_history['train_loss']) + 1))
        
        # 损失曲线
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['train_loss'],
                      mode='lines', name='训练损失',
                      line=dict(color='steelblue')),
            row=1, col=1
        )
        if 'val_loss' in self.training_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=self.training_history['val_loss'],
                          mode='lines', name='验证损失',
                          line=dict(color='coral')),
                row=1, col=1
            )
        
        # 准确率曲线
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['train_accuracy'],
                      mode='lines', name='训练准确率',
                      line=dict(color='steelblue')),
            row=1, col=2
        )
        if 'val_accuracy' in self.training_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=self.training_history['val_accuracy'],
                          mode='lines', name='验证准确率',
                          line=dict(color='coral')),
                row=1, col=2
            )
        
        fig.update_layout(
            title='LSTM 训练历史',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def get_sequence_predictions_aligned(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        获取与原始索引对齐的预测结果
        
        由于滑动窗口会丢失前 window_size 个样本，
        此方法返回对齐后的预测概率和真实标签。
        
        Args:
            X: 特征矩阵
            y: 标签
            
        Returns:
            (aligned_proba, aligned_y) 对齐后的预测概率和真实标签
        """
        X_seq, y_seq = self._create_sequences(X, y)
        proba = self.model.predict(X_seq, verbose=0).flatten()
        
        # 对齐索引（跳过前 window_size 个）
        aligned_index = X.index[self.window_size:]
        aligned_proba = pd.Series(proba, index=aligned_index, name='lstm_proba')
        aligned_y = pd.Series(y_seq, index=aligned_index, name='label')
        
        return aligned_proba, aligned_y
