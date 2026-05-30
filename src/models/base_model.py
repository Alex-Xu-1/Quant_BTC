"""
模型基类 - BaseModel

定义所有 ML 模型的统一接口，支持继承扩展。
新模型只需继承 BaseModel 并实现 train/predict 方法即可。

作者: Quant_BTC 项目
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BaseModel(ABC):
    """
    模型抽象基类
    
    所有 ML 模型必须继承此类并实现以下方法：
    - train(): 训练模型
    - predict(): 预测
    - predict_proba(): 预测概率
    - save(): 保存模型
    - load(): 加载模型
    
    使用示例：
        class MyModel(BaseModel):
            def train(self, X_train, y_train, X_val, y_val):
                ...
            def predict(self, X):
                ...
    """
    
    def __init__(self, name: str = 'BaseModel', models_dir: str = 'models'):
        """
        Args:
            name: 模型名称
            models_dir: 模型保存目录
        """
        self.name = name
        self.models_dir = models_dir
        self.model = None
        self.is_trained = False
        self.training_history = {}
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            
        Returns:
            训练历史/指标字典
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测类别数组
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率数组（正类概率）
        """
        pass
    
    @abstractmethod
    def save(self, path: Optional[str] = None) -> str:
        """
        保存模型到文件
        
        Args:
            path: 保存路径（可选，默认使用 models_dir/name）
            
        Returns:
            保存的文件路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        从文件加载模型
        
        Args:
            path: 模型文件路径
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, average='binary', zero_division=0),
        }
        
        # AUC（需要概率值）
        try:
            metrics['auc'] = roc_auc_score(y, y_proba)
        except ValueError:
            metrics['auc'] = 0.5
        
        return metrics
    
    def print_evaluation(self, X: pd.DataFrame, y: pd.Series, 
                         dataset_name: str = "测试集") -> Dict:
        """
        打印评估结果
        
        Args:
            X: 特征矩阵
            y: 真实标签
            dataset_name: 数据集名称
            
        Returns:
            评估指标字典
        """
        metrics = self.evaluate(X, y)
        
        print(f"\n📊 {self.name} - {dataset_name} 评估结果：")
        print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
        print(f"  F1 分数:            {metrics['f1']:.4f}")
        print(f"  AUC-ROC:            {metrics['auc']:.4f}")
        
        # 分类报告
        y_pred = self.predict(X)
        print(f"\n  分类报告：")
        print(classification_report(y, y_pred, target_names=['跌', '涨']))
        
        return metrics
    
    def plot_roc_curve(self, X: pd.DataFrame, y: pd.Series) -> go.Figure:
        """
        绘制 ROC 曲线
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            Plotly Figure 对象
        """
        from sklearn.metrics import roc_curve
        
        y_proba = self.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        fig = go.Figure()
        
        # ROC 曲线
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'{self.name} (AUC={auc:.4f})',
            line=dict(color='steelblue', width=2)
        ))
        
        # 对角线（随机分类器）
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='随机分类器',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title=f'{self.name} ROC 曲线',
            xaxis_title='假正率 (FPR)',
            yaxis_title='真正率 (TPR)',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def check_overfitting(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          threshold: float = 0.1) -> Dict:
        """
        检查模型是否过拟合
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            threshold: 过拟合判定阈值（AUC差异）
            
        Returns:
            过拟合检查结果
        """
        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)
        
        auc_diff = train_metrics['auc'] - test_metrics['auc']
        is_overfitting = auc_diff > threshold
        
        result = {
            'train_auc': train_metrics['auc'],
            'test_auc': test_metrics['auc'],
            'auc_difference': auc_diff,
            'is_overfitting': is_overfitting,
            'suggestion': ''
        }
        
        if is_overfitting:
            result['suggestion'] = (
                f"⚠️ 检测到过拟合！训练集AUC({train_metrics['auc']:.4f}) "
                f"与测试集AUC({test_metrics['auc']:.4f})差异为 {auc_diff:.4f} > {threshold}\n"
                f"建议：增加正则化强度、减少模型复杂度、增加Dropout、使用更多数据"
            )
            print(result['suggestion'])
        else:
            result['suggestion'] = (
                f"✅ 未检测到明显过拟合。"
                f"训练集AUC: {train_metrics['auc']:.4f}, 测试集AUC: {test_metrics['auc']:.4f}"
            )
            print(result['suggestion'])
        
        return result
