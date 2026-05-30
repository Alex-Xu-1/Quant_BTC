"""
实盘交易接口模块 - Exchange

基于 ccxt 的统一交易接口抽象层，支持回测/模拟盘/实盘无缝切换。

功能：
- ExchangeInterface: 交易接口抽象基类
- BinanceExchange: Binance 交易所实现
- PaperTrading: 模拟交易模式（使用实时数据但不实际下单）

安全说明：
- API 密钥通过 .env 文件管理，不硬编码
- 实盘模式需要显式确认

作者: Quant_BTC 项目
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv


@dataclass
class Order:
    """订单数据类"""
    order_id: str
    symbol: str
    side: str               # 'buy' or 'sell'
    order_type: str         # 'market' or 'limit'
    quantity: float
    price: Optional[float]  # limit 订单的价格
    status: str             # 'open', 'filled', 'cancelled'
    filled_price: float = 0.0
    filled_quantity: float = 0.0
    commission: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    side: str               # 'long' or 'short'
    quantity: float
    entry_price: float
    unrealized_pnl: float = 0.0
    timestamp: Optional[datetime] = None


class ExchangeInterface(ABC):
    """
    交易接口抽象基类
    
    定义统一的交易接口，回测引擎和实盘引擎共享此接口。
    切换模式仅需修改配置参数。
    
    使用示例：
        # 实盘
        exchange = BinanceExchange(api_key, secret)
        
        # 模拟盘
        exchange = PaperTrading()
        
        # 统一接口
        order = exchange.place_order('BTC/USDT', 'buy', 'market', 0.001)
        balance = exchange.get_balance()
    """
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: Optional[float] = None) -> Order:
        """
        下单
        
        Args:
            symbol: 交易对（如 'BTC/USDT'）
            side: 方向 ('buy' 或 'sell')
            order_type: 订单类型 ('market' 或 'limit')
            quantity: 数量
            price: 限价单价格（市价单不需要）
            
        Returns:
            Order 对象
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        撤单
        
        Args:
            order_id: 订单ID
            symbol: 交易对
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        查询持仓
        
        Args:
            symbol: 交易对
            
        Returns:
            Position 对象或 None
        """
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict:
        """
        查询余额
        
        Returns:
            余额字典 {'USDT': {'free': ..., 'used': ..., 'total': ...}}
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        获取当前价格
        
        Args:
            symbol: 交易对
            
        Returns:
            当前价格
        """
        pass


class BinanceExchange(ExchangeInterface):
    """
    Binance 交易所实现
    
    基于 ccxt 库封装 Binance API。
    
    ⚠️ 注意：实盘交易有真实资金风险！
    
    使用示例：
        exchange = BinanceExchange()  # 从 .env 加载密钥
        price = exchange.get_current_price('BTC/USDT')
        order = exchange.place_order('BTC/USDT', 'buy', 'market', 0.001)
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None,
                 testnet: bool = False):
        """
        Args:
            api_key: API 密钥（不传则从环境变量加载）
            api_secret: API 密钥（不传则从环境变量加载）
            testnet: 是否使用测试网
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET', '')
        
        if not self.api_key or not self.api_secret:
            print("  ⚠️ 未配置 API 密钥，仅支持公共接口（行情查询）")
        
        # 初始化 ccxt 交易所
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("  🧪 已启用 Binance 测试网模式")
    
    def place_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: Optional[float] = None) -> Order:
        """下单"""
        try:
            if order_type == 'market':
                result = self.exchange.create_market_order(symbol, side, quantity)
            elif order_type == 'limit':
                if price is None:
                    raise ValueError("限价单必须指定价格")
                result = self.exchange.create_limit_order(symbol, side, quantity, price)
            else:
                raise ValueError(f"不支持的订单类型: {order_type}")
            
            order = Order(
                order_id=str(result['id']),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=result['status'],
                filled_price=result.get('average', 0) or 0,
                filled_quantity=result.get('filled', 0) or 0,
                commission=result.get('fee', {}).get('cost', 0) or 0,
                timestamp=datetime.now()
            )
            
            print(f"  ✅ 订单已提交: {side} {quantity} {symbol} @ {order_type}")
            return order
            
        except Exception as e:
            print(f"  ❌ 下单失败: {e}")
            return Order(
                order_id='', symbol=symbol, side=side,
                order_type=order_type, quantity=quantity,
                price=price, status='failed',
                timestamp=datetime.now()
            )
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """撤单"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            print(f"  ✅ 订单已撤销: {order_id}")
            return True
        except Exception as e:
            print(f"  ❌ 撤单失败: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """查询持仓（现货模式下通过余额推断）"""
        try:
            balance = self.exchange.fetch_balance()
            base_currency = symbol.split('/')[0]  # BTC
            
            amount = balance.get(base_currency, {}).get('total', 0)
            if amount > 0:
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                return Position(
                    symbol=symbol,
                    side='long',
                    quantity=amount,
                    entry_price=0,  # 现货无法直接获取入场价
                    unrealized_pnl=0,
                    timestamp=datetime.now()
                )
            return None
        except Exception as e:
            print(f"  ❌ 查询持仓失败: {e}")
            return None
    
    def get_balance(self) -> Dict:
        """查询余额"""
        try:
            balance = self.exchange.fetch_balance()
            # 只返回有余额的币种
            result = {}
            for currency, info in balance.items():
                if isinstance(info, dict) and info.get('total', 0) > 0:
                    result[currency] = {
                        'free': info.get('free', 0),
                        'used': info.get('used', 0),
                        'total': info.get('total', 0)
                    }
            return result
        except Exception as e:
            print(f"  ❌ 查询余额失败: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"  ❌ 获取价格失败: {e}")
            return 0.0


class PaperTrading(ExchangeInterface):
    """
    模拟交易模式（Paper Trading）
    
    使用实时数据但不实际下单，记录虚拟交易。
    适合策略验证阶段使用。
    
    使用示例：
        paper = PaperTrading(initial_balance=10000)
        order = paper.place_order('BTC/USDT', 'buy', 'market', 0.01)
        print(paper.get_balance())
    """
    
    def __init__(self, initial_balance: float = 10000, 
                 commission_rate: float = 0.001):
        """
        Args:
            initial_balance: 初始虚拟余额 (USDT)
            commission_rate: 手续费率
        """
        self.commission_rate = commission_rate
        self.balance = {'USDT': initial_balance}
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_counter = 0
        
        # 用于获取实时价格的公共 API
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        print(f"  📝 Paper Trading 模式已启动")
        print(f"     初始余额: {initial_balance:,.2f} USDT")
        print(f"     手续费率: {commission_rate*100}%")
    
    def place_order(self, symbol: str, side: str, order_type: str,
                    quantity: float, price: Optional[float] = None) -> Order:
        """模拟下单"""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter:06d}"
        
        # 获取当前价格
        current_price = self.get_current_price(symbol)
        if current_price == 0:
            return Order(
                order_id=order_id, symbol=symbol, side=side,
                order_type=order_type, quantity=quantity,
                price=price, status='failed',
                timestamp=datetime.now()
            )
        
        fill_price = price if (order_type == 'limit' and price) else current_price
        
        # 计算手续费
        commission = fill_price * quantity * self.commission_rate
        
        # 更新虚拟余额
        base_currency = symbol.split('/')[0]  # BTC
        quote_currency = symbol.split('/')[1]  # USDT
        
        if side == 'buy':
            cost = fill_price * quantity + commission
            if self.balance.get(quote_currency, 0) < cost:
                print(f"  ⚠️ 余额不足: 需要 {cost:.2f} {quote_currency}")
                return Order(
                    order_id=order_id, symbol=symbol, side=side,
                    order_type=order_type, quantity=quantity,
                    price=price, status='failed',
                    timestamp=datetime.now()
                )
            self.balance[quote_currency] = self.balance.get(quote_currency, 0) - cost
            self.balance[base_currency] = self.balance.get(base_currency, 0) + quantity
        
        elif side == 'sell':
            if self.balance.get(base_currency, 0) < quantity:
                print(f"  ⚠️ 持仓不足: 需要 {quantity} {base_currency}")
                return Order(
                    order_id=order_id, symbol=symbol, side=side,
                    order_type=order_type, quantity=quantity,
                    price=price, status='failed',
                    timestamp=datetime.now()
                )
            revenue = fill_price * quantity - commission
            self.balance[base_currency] = self.balance.get(base_currency, 0) - quantity
            self.balance[quote_currency] = self.balance.get(quote_currency, 0) + revenue
        
        # 创建订单记录
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status='filled',
            filled_price=fill_price,
            filled_quantity=quantity,
            commission=commission,
            timestamp=datetime.now()
        )
        
        self.orders.append(order)
        print(f"  📝 [Paper] {side.upper()} {quantity:.6f} {symbol} @ ${fill_price:,.2f} "
              f"(手续费: ${commission:.2f})")
        
        return order
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """模拟撤单（Paper Trading 中市价单立即成交，无需撤单）"""
        print(f"  📝 [Paper] 撤单: {order_id}")
        return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """查询虚拟持仓"""
        base_currency = symbol.split('/')[0]
        amount = self.balance.get(base_currency, 0)
        
        if amount > 0.00001:  # 忽略极小余额
            return Position(
                symbol=symbol,
                side='long',
                quantity=amount,
                entry_price=0,
                timestamp=datetime.now()
            )
        return None
    
    def get_balance(self) -> Dict:
        """查询虚拟余额"""
        result = {}
        for currency, amount in self.balance.items():
            if amount > 0:
                result[currency] = {
                    'free': amount,
                    'used': 0,
                    'total': amount
                }
        return result
    
    def get_current_price(self, symbol: str) -> float:
        """获取实时价格（使用公共API）"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"  ❌ 获取价格失败: {e}")
            return 0.0
    
    def get_total_value(self, symbol: str = 'BTC/USDT') -> float:
        """计算总资产价值（USDT计价）"""
        total = self.balance.get('USDT', 0)
        
        base_currency = symbol.split('/')[0]
        base_amount = self.balance.get(base_currency, 0)
        
        if base_amount > 0:
            price = self.get_current_price(symbol)
            total += base_amount * price
        
        return total
    
    def get_trade_history(self) -> pd.DataFrame:
        """获取交易历史"""
        if not self.orders:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            '订单ID': o.order_id,
            '时间': o.timestamp,
            '交易对': o.symbol,
            '方向': o.side,
            '类型': o.order_type,
            '数量': o.quantity,
            '成交价': o.filled_price,
            '手续费': o.commission,
            '状态': o.status
        } for o in self.orders])


def create_exchange(config: Dict) -> ExchangeInterface:
    """
    工厂函数：根据配置创建交易接口实例
    
    Args:
        config: 配置字典，需包含 'trading_mode' 键
            - 'backtest': 不创建交易接口（回测使用 Backtester）
            - 'paper': 创建 PaperTrading 实例
            - 'live': 创建 BinanceExchange 实例
            
    Returns:
        ExchangeInterface 实例
    """
    mode = config.get('trading_mode', 'backtest')
    
    if mode == 'paper':
        return PaperTrading(
            initial_balance=config.get('initial_capital', 10000),
            commission_rate=config.get('commission_rate', 0.001)
        )
    elif mode == 'live':
        print("  ⚠️ 警告：即将连接实盘交易！请确认已充分了解风险。")
        return BinanceExchange(
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret')
        )
    else:
        print("  ℹ️ 回测模式，无需创建交易接口")
        return None
