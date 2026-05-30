"""
XGBoost 模型 - XGBModel

基于 XGBoost 梯度提升树的价格方向分类预测模型。
支持 Optuna 贝叶斯超参数优化和 SHAP 特征重要性分析。

作者: Quant_BTC 项目
"""

import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, accuracy_score

from models.base_model import BaseModel


class XGBModel(BaseModel):
    """
    XGBoost 分类模型
    
    用于预测下一周期 BTC 价格方向（涨/跌）。
    
    特点：
    - 擅长处理结构化特征数据
    - 训练速度快，可解释性强
    - 支持 Optuna 自动超参数优化
    - 支持 SHAP 值特征重要性分析
    
    使用示例：
        model = XGBModel()
        model.train(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        model.plot_feature_importance()
    """
    
    def __init__(self, models_dir: str = 'models', params: Optional[Dict] = None):
        """
        Args:
            models_dir: 模型保存目录
            params: XGBoost 超参数字典（可选，不传则使用默认值）
        """
        super().__init__(name='XGBoost', models_dir=models_dir)
        
        # 默认超参数
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
        }
        
        self.feature_names = []
        self.shap_values = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict:
        """
        训练 XGBoost 模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练历史字典
        """
        print(f"🚀 开始训练 {self.name} 模型...")
        print(f"   训练集大小: {X_train.shape}")
        
        self.feature_names = list(X_train.columns)
        
        # 构建模型
        self.model = xgb.XGBClassifier(**self.params)
        
        # 训练参数
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            fit_params['verbose'] = 50  # 每50轮打印一次
        
        # 训练
        self.model.fit(X_train, y_train, **fit_params)
        
        self.is_trained = True
        
        # 记录训练历史
        history = {
            'train_auc': roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1]),
        }
        if X_val is not None:
            history['val_auc'] = roc_auc_score(y_val, self.model.predict_proba(X_val)[:, 1])
        
        self.training_history = history
        
        print(f"   ✅ 训练完成！")
        print(f"   训练集 AUC: {history['train_auc']:.4f}")
        if 'val_auc' in history:
            print(f"   验证集 AUC: {history['val_auc']:.4f}")
        
        return history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测正类（涨）概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: Optional[str] = None) -> str:
        """保存模型"""
        if path is None:
            path = os.path.join(self.models_dir, 'xgboost_model.pkl')
        joblib.dump(self.model, path)
        print(f"  💾 模型已保存至: {path}")
        return path
    
    def load(self, path: str) -> None:
        """加载模型"""
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"  📂 模型已加载: {path}")
    
    def optimize_hyperparams(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        timeout: int = 600
    ) -> Dict:
        """
        使用 Optuna 贝叶斯优化超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            n_trials: 优化轮数
            timeout: 超时时间（秒）
            
        Returns:
            最优超参数字典
        """
        print(f"🔍 开始 Optuna 超参数优化（{n_trials} 轮）...")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
                'random_state': 42,
                'n_jobs': -1,
                'use_label_encoder': False,
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            return auc
        
        # 创建 Optuna study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # 获取最优参数
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
        })
        
        print(f"  ✅ 优化完成！最优 AUC: {study.best_value:.4f}")
        print(f"  最优参数：")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        # 更新模型参数
        self.params = best_params
        
        return best_params
    
    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        train_ratio: float = 0.7
    ) -> Dict:
        """
        Walk-Forward 时间序列交叉验证
        
        Args:
            X: 完整特征矩阵
            y: 完整标签
            n_splits: 分割数
            train_ratio: 每个窗口中训练集的比例
            
        Returns:
            交叉验证结果
        """
        print(f"🔄 Walk-Forward 交叉验证（{n_splits} 折）...")
        
        n = len(X)
        fold_size = n // n_splits
        results = []
        
        for i in range(n_splits):
            # 训练集：从开始到当前折的训练部分
            train_end = int((i + 1) * fold_size * train_ratio)
            # 测试集：当前折的剩余部分
            test_start = train_end
            test_end = min((i + 1) * fold_size, n)
            
            if test_start >= test_end:
                continue
            
            X_train_fold = X.iloc[:train_end]
            y_train_fold = y.iloc[:train_end]
            X_test_fold = X.iloc[test_start:test_end]
            y_test_fold = y.iloc[test_start:test_end]
            
            # 训练临时模型
            temp_model = xgb.XGBClassifier(**self.params)
            temp_model.fit(X_train_fold, y_train_fold, verbose=0)
            
            # 评估
            y_pred_proba = temp_model.predict_proba(X_test_fold)[:, 1]
            y_pred = temp_model.predict(X_test_fold)
            
            fold_result = {
                'fold': i + 1,
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold),
                'auc': roc_auc_score(y_test_fold, y_pred_proba),
                'accuracy': accuracy_score(y_test_fold, y_pred),
                'test_period': f"{X_test_fold.index[0]} ~ {X_test_fold.index[-1]}"
            }
            results.append(fold_result)
            print(f"  Fold {i+1}: AUC={fold_result['auc']:.4f}, "
                  f"Acc={fold_result['accuracy']:.4f}, "
                  f"Period: {fold_result['test_period']}")
        
        # 汇总
        avg_auc = np.mean([r['auc'] for r in results])
        std_auc = np.std([r['auc'] for r in results])
        print(f"\n  📊 平均 AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        
        return {
            'folds': results,
            'avg_auc': avg_auc,
            'std_auc': std_auc
        }
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算 SHAP 值
        
        Args:
            X: 特征矩阵
            
        Returns:
            SHAP 值数组
        """
        print("  🔍 计算 SHAP 值...")
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        return self.shap_values
    
    def plot_feature_importance(self, top_n: int = 20) -> go.Figure:
        """
        绘制特征重要性排序图（基于 XGBoost 内置重要性）
        
        Args:
            top_n: 显示前 N 个重要特征
            
        Returns:
            Plotly Figure 对象
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 获取特征重要性
        importance = self.model.feature_importances_
        feature_importance = pd.Series(importance, index=self.feature_names)
        feature_importance = feature_importance.sort_values(ascending=False).head(top_n)
        
        fig = go.Figure(data=go.Bar(
            x=feature_importance.values,
            y=feature_importance.index,
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title=f'XGBoost 特征重要性 (Top {top_n})',
            xaxis_title='重要性分数',
            yaxis_title='特征',
            height=500,
            template='plotly_dark',
            yaxis=dict(autorange='reversed')
        )
        
        return fig
    
    def plot_shap_summary(self, X: pd.DataFrame, max_display: int = 20) -> None:
        """
        绘制 SHAP 摘要图
        
        Args:
            X: 特征矩阵
            max_display: 最多显示的特征数
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # 使用 matplotlib 绘制 SHAP 图（shap 库原生支持）
        shap.summary_plot(
            self.shap_values, X, 
            max_display=max_display, 
            show=True
        )
