"""
数据获取模块 - DataFetcher

通过 ccxt 库连接 Binance 交易所，获取 BTC/USDT 历史 K 线数据。
支持多周期数据下载、本地缓存、增量更新和异常处理。

作者: Quant_BTC 项目
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional, List

import ccxt
import pandas as pd
from tqdm import tqdm


class DataFetcher:
    """
    数据获取器 - 封装 ccxt Binance 接口
    
    功能：
    - 多周期（1h/4h/1d）K线数据批量下载
    - 本地 CSV 缓存机制，支持增量更新
    - 网络异常自动回退到本地缓存
    
    使用示例：
        fetcher = DataFetcher(data_dir='data/raw')
        df = fetcher.fetch_ohlcv('BTC/USDT', '4h', '2020-01-01')
    """
    
    def __init__(self, data_dir: str = 'data/raw', exchange_id: str = 'binance'):
        """
        初始化数据获取器
        
        Args:
            data_dir: 数据存储目录路径
            exchange_id: 交易所ID（默认 binance）
        """
        self.data_dir = data_dir
        self.exchange_id = exchange_id
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化交易所连接
        self.exchange = self._init_exchange()
        
    def _init_exchange(self) -> ccxt.Exchange:
        """初始化交易所连接（使用公共API，无需密钥）"""
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,  # 启用请求频率限制
            'options': {
                'defaultType': 'spot',  # 现货市场
            }
        })
        return exchange
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        """
        生成缓存文件路径
        
        Args:
            symbol: 交易对（如 'BTC/USDT'）
            timeframe: 时间周期（如 '4h'）
            
        Returns:
            缓存文件的完整路径
        """
        # 将 BTC/USDT 转换为 BTC_USDT
        safe_symbol = symbol.replace('/', '_')
        filename = f"{safe_symbol}_{timeframe}.csv"
        return os.path.join(self.data_dir, filename)
    
    def _load_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        从本地缓存加载数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            
        Returns:
            DataFrame 或 None（缓存不存在时）
        """
        cache_path = self._get_cache_path(symbol, timeframe)
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            print(f"  📂 从本地缓存加载: {cache_path}")
            print(f"     数据范围: {df.index[0]} ~ {df.index[-1]}, 共 {len(df)} 条")
            return df
        return None
    
    def _save_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        保存数据到本地缓存
        
        Args:
            df: 要保存的 DataFrame
            symbol: 交易对
            timeframe: 时间周期
        """
        cache_path = self._get_cache_path(symbol, timeframe)
        df_to_save = df.copy()
        df_to_save.index.name = 'timestamp'
        df_to_save.to_csv(cache_path)
        print(f"  💾 数据已保存至: {cache_path}")
    
    def fetch_ohlcv(
        self, 
        symbol: str = 'BTC/USDT', 
        timeframe: str = '4h',
        start_date: str = '2020-01-01',
        use_cache: bool = True,
        update_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取历史 K 线数据（OHLCV）
        
        通过 ccxt 分页获取从 start_date 至今的完整 K 线数据。
        支持本地缓存和增量更新。
        
        Args:
            symbol: 交易对，默认 'BTC/USDT'
            timeframe: 时间周期，支持 '1h', '4h', '1d'
            start_date: 数据起始日期，格式 'YYYY-MM-DD'
            use_cache: 是否使用本地缓存
            update_cache: 是否更新缓存（增量下载新数据）
            
        Returns:
            包含 OHLCV 数据的 DataFrame，索引为 timestamp
            
        Raises:
            Exception: 网络请求失败且无本地缓存时抛出异常
        """
        print(f"📊 获取 {symbol} {timeframe} K线数据...")
        print(f"   起始日期: {start_date}")
        
        # 尝试加载缓存
        cached_df = None
        if use_cache:
            cached_df = self._load_cache(symbol, timeframe)
        
        # 确定起始时间戳
        if cached_df is not None and update_cache:
            # 增量更新：从缓存最后一条数据的时间开始
            since_ts = int(cached_df.index[-1].timestamp() * 1000)
            print(f"  🔄 增量更新模式，从 {cached_df.index[-1]} 开始")
        elif cached_df is not None and not update_cache:
            # 直接使用缓存
            return cached_df
        else:
            # 全量下载
            since_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        
        # 通过 API 获取数据
        try:
            all_ohlcv = self._fetch_all_ohlcv(symbol, timeframe, since_ts)
        except Exception as e:
            print(f"  ⚠️ API 请求失败: {e}")
            if cached_df is not None:
                print(f"  📂 回退到本地缓存数据")
                return cached_df
            else:
                raise Exception(f"无法获取数据且无本地缓存: {e}")
        
        # 转换为 DataFrame
        if len(all_ohlcv) == 0:
            if cached_df is not None:
                return cached_df
            raise Exception("未获取到任何数据")
        
        new_df = pd.DataFrame(
            all_ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        new_df.set_index('timestamp', inplace=True)
        
        # 合并缓存数据和新数据
        if cached_df is not None and update_cache:
            df = pd.concat([cached_df, new_df])
            df = df[~df.index.duplicated(keep='last')]  # 去重
            df.sort_index(inplace=True)
        else:
            df = new_df
        
        # 保存缓存
        self._save_cache(df, symbol, timeframe)
        
        # 打印统计信息
        self._print_stats(df, symbol, timeframe)
        
        return df
    
    def _fetch_all_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        since: int
    ) -> List:
        """
        分页获取所有 OHLCV 数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            since: 起始时间戳（毫秒）
            
        Returns:
            OHLCV 数据列表
        """
        all_ohlcv = []
        limit = 1000  # 每次请求的最大数据条数
        
        # 计算时间周期对应的毫秒数
        timeframe_ms = self._timeframe_to_ms(timeframe)
        
        # 计算预估总请求次数
        now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        estimated_bars = (now_ts - since) // timeframe_ms
        estimated_requests = max(1, estimated_bars // limit)
        
        print(f"  🌐 开始从交易所下载数据（预估 {estimated_bars} 条）...")
        
        pbar = tqdm(total=estimated_bars, desc="  下载进度", unit="条")
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=limit
                )
            except Exception as e:
                print(f"\n  ⚠️ 请求异常: {e}，等待后重试...")
                time.sleep(5)
                continue
            
            if len(ohlcv) == 0:
                break
            
            all_ohlcv.extend(ohlcv)
            pbar.update(len(ohlcv))
            
            # 更新起始时间为最后一条数据的时间 + 1个周期
            since = ohlcv[-1][0] + timeframe_ms
            
            # 如果获取的数据少于 limit，说明已到最新
            if len(ohlcv) < limit:
                break
            
            # 遵守频率限制
            time.sleep(self.exchange.rateLimit / 1000)
        
        pbar.close()
        print(f"  ✅ 下载完成，共获取 {len(all_ohlcv)} 条数据")
        
        return all_ohlcv
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """
        将时间周期字符串转换为毫秒数
        
        Args:
            timeframe: 时间周期字符串（如 '1h', '4h', '1d'）
            
        Returns:
            对应的毫秒数
        """
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
        }
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers[unit]
    
    def _print_stats(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """打印数据统计信息"""
        print(f"\n📈 {symbol} {timeframe} 数据统计：")
        print(f"  时间范围:  {df.index[0]} ~ {df.index[-1]}")
        print(f"  数据条数:  {len(df):,}")
        print(f"  缺失值统计:")
        missing = df.isnull().sum()
        for col in df.columns:
            if missing[col] > 0:
                print(f"    {col}: {missing[col]} 条缺失")
        if missing.sum() == 0:
            print(f"    无缺失值 ✅")
        print(f"  价格范围:  ${df['low'].min():,.2f} ~ ${df['high'].max():,.2f}")
        print(f"  最新收盘价: ${df['close'].iloc[-1]:,.2f}")
    
    def fetch_multiple_timeframes(
        self,
        symbol: str = 'BTC/USDT',
        timeframes: List[str] = ['1h', '4h', '1d'],
        start_date: str = '2020-01-01'
    ) -> dict:
        """
        获取多个时间周期的数据
        
        Args:
            symbol: 交易对
            timeframes: 时间周期列表
            start_date: 起始日期
            
        Returns:
            字典，key 为时间周期，value 为对应的 DataFrame
        """
        data = {}
        for tf in timeframes:
            print(f"\n{'='*50}")
            data[tf] = self.fetch_ohlcv(symbol, tf, start_date)
        return data


# ============================================
# 便捷函数
# ============================================

def get_btc_data(
    timeframe: str = '4h',
    start_date: str = '2020-01-01',
    data_dir: str = 'data/raw'
) -> pd.DataFrame:
    """
    快速获取 BTC/USDT 数据的便捷函数
    
    Args:
        timeframe: 时间周期，默认 '4h'
        start_date: 起始日期，默认 '2020-01-01'
        data_dir: 数据存储目录
        
    Returns:
        BTC/USDT OHLCV DataFrame
    """
    fetcher = DataFetcher(data_dir=data_dir)
    return fetcher.fetch_ohlcv('BTC/USDT', timeframe, start_date)
