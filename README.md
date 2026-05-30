# 🚀 BTC 机器学习量化交易策略

基于 **XGBoost + LSTM 集成学习** 的比特币量化交易策略研究框架。

## 📋 项目简介

本项目是一个完整的比特币量化交易策略研究 Jupyter Notebook，采用机器学习方法预测 BTC/USDT 价格方向，并通过回测验证策略有效性。

### 核心特性

- 🤖 **双模型集成**：XGBoost（结构化特征） + LSTM（时序模式）加权融合
- 📊 **丰富特征工程**：技术指标、动量因子、波动率因子、时间特征等 50+ 维特征
- 📈 **交互式可视化**：基于 Plotly 的交互式图表，支持缩放、悬停查看
- 🔄 **完整回测引擎**：事件驱动型回测，考虑手续费和滑点
- 🔌 **实盘接口预留**：基于 ccxt 的统一交易接口，支持一键切换实盘

## 📁 项目结构

```
Quant_BTC/
├── README.md                    # 项目说明文档（本文件）
├── requirements.txt             # Python 依赖包
├── .env.example                 # 环境变量模板
├── btc_ml_strategy.ipynb        # 📓 主 Notebook 文件
├── data/                        # 数据存储
│   ├── raw/                     # 原始 K 线数据
│   └── processed/               # 处理后的特征数据
├── models/                      # 训练好的模型文件
├── results/                     # 回测结果输出
└── src/                         # 源代码模块
    ├── __init__.py
    ├── data_fetcher.py          # 数据获取模块
    ├── feature_engineer.py      # 特征工程模块
    ├── strategy.py              # 交易策略模块
    ├── backtester.py            # 回测引擎
    ├── analyzer.py              # 结果分析模块
    ├── exchange.py              # 实盘交易接口
    └── models/                  # ML 模型定义
        ├── __init__.py
        ├── base_model.py        # 模型基类
        ├── xgb_model.py         # XGBoost 模型
        └── lstm_model.py        # LSTM 模型
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo_url>
cd Quant_BTC

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量（可选，实盘交易时需要）

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 Binance API 密钥
```

### 3. 启动 Notebook

```bash
jupyter notebook btc_ml_strategy.ipynb
```

### 4. 按顺序执行 Cell

Notebook 按照以下流程组织，建议从上到下依次执行：

1. 环境初始化与配置
2. 数据获取
3. 数据预处理与探索性分析
4. 特征工程
5. 模型训练与评估（XGBoost）
6. 模型训练与评估（LSTM）
7. 模型集成与策略设计
8. 回测执行
9. 结果分析与投资建议
10. 实盘部署指南（可选）

## 📊 策略概述

### 策略架构

```
数据层：历史K线 → 技术指标 → 特征工程 → 标准化
    ↓
模型层：XGBoost（方向分类） + LSTM（序列预测） → 集成投票
    ↓
策略层：信号生成 → 仓位管理（Kelly公式） → 风控规则
    ↓
执行层：回测引擎 / ccxt实盘接口
```

### 关键参数

| 参数 | 值 |
|------|-----|
| 交易标的 | BTC/USDT |
| 时间周期 | 4h（主策略） |
| 数据范围 | 2020-01 至今 |
| 初始资金 | 10,000 USDT |
| 手续费 | 0.1% |
| 滑点 | 0.05% |

## ⚠️ 风险提示

1. **本项目仅供学习研究使用**，不构成任何投资建议
2. 加密货币市场波动剧烈，历史回测表现不代表未来收益
3. 机器学习模型存在过拟合风险，需定期重新训练
4. 实盘交易前请充分了解风险，建议先使用 Paper Trading 模式验证
5. 请勿将全部资金投入单一策略

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| 数据获取 | ccxt |
| 数据处理 | pandas, numpy |
| 可视化 | plotly, matplotlib |
| 技术指标 | pandas-ta |
| 机器学习 | scikit-learn, xgboost |
| 深度学习 | tensorflow (keras) |
| 模型解释 | shap |
| 超参优化 | optuna |
| 交互组件 | ipywidgets |

## 📝 License

MIT License - 仅供学习研究使用
