
"""
Phase 1: Simple Trading Bot (Paper Trading ready)

What's included:
- Fetch historical price data (yfinance)
- Compute simple technical indicators: SMA, EMA, RSI
- Simple strategy: MA crossover + RSI filter
- Backtester (walk-forward, basic P&L, Sharpe, drawdown)
- Optional: connect to Alpaca paper trading (disabled by default). If you enable it,
  set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables and set
  `EXECUTE_TRADES = True`.

Notes:
- This script is intentionally simple and educational. Use paper trading only until
  you understand the strategy and risks.
- Dependencies: pandas, numpy, yfinance, pytz, alpaca-trade-api (optional)

Run example:
    python phase1_trading_bot.py --symbol AAPL --start 2022-01-01 --end 2024-12-31

"""

import os
import argparse
from dataclasses import dataclass
from datetime import datetime
import time
import random
import numpy as np
import pandas as pd
import yfinance as yf

# Optional import only used if the user chooses to execute live/paper trades
try:
    import alpaca_trade_api as tradeapi
except Exception:
    tradeapi = None


# ----------------------------- Config & Utilities -----------------------------
@dataclass
class Config:
    symbol: str = "AAPL"
    timeframe: str = "1d"
    start: str = "2022-01-01"
    end: str = None
    short_window: int = 20
    long_window: int = 50
    rsi_period: int = 14
    rsi_upper: int = 70
    rsi_lower: int = 30
    initial_cash: float = 10000.0
    position_size: float = 0.1  # fraction of portfolio per trade
    execute_trades: bool = False  # enable Alpaca paper trading


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 Trading Bot - MA crossover + RSI")
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--execute", action="store_true", help="Enable Alpaca paper trading (needs keys)")
    return p.parse_args()


# ----------------------------- Data & Indicators -----------------------------

def fetch_ohlcv(symbol: str, start: str, end: str = None, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data using yfinance and return a clean dataframe."""
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} {start} {end}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    # Moving averages
    df['sma_short'] = df['close'].rolling(cfg.short_window, min_periods=1).mean()
    df['sma_long'] = df['close'].rolling(cfg.long_window, min_periods=1).mean()
    # Exponential moving average (optional)
    df['ema_short'] = df['close'].ewm(span=cfg.short_window, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=cfg.long_window, adjust=False).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(cfg.rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(cfg.rsi_period, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['rsi'] = 100 - (100 / (1 + rs)).fillna(50)
    # Clean up
    df = df.dropna().copy()
    return df


# ----------------------------- Strategy Signals -----------------------------

def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Generate buy/sell signals.
    Strategy: buy when short SMA crosses above long SMA AND RSI < rsi_upper.
              sell when short SMA crosses below long SMA OR RSI > rsi_upper.
    Signals: 1 = buy, -1 = sell, 0 = hold
    """
    df = df.copy()
    df['signal'] = 0

    # Crossover
    df['prev_sma_short'] = df['sma_short'].shift(1)
    df['prev_sma_long'] = df['sma_long'].shift(1)

    bullish_cross = (df['prev_sma_short'] <= df['prev_sma_long']) & (df['sma_short'] > df['sma_long'])
    bearish_cross = (df['prev_sma_short'] >= df['prev_sma_long']) & (df['sma_short'] < df['sma_long'])

    buy_cond = bullish_cross & (df['rsi'] < cfg.rsi_upper)
    sell_cond = bearish_cross | (df['rsi'] > cfg.rsi_upper)

    df.loc[buy_cond, 'signal'] = 1
    df.loc[sell_cond, 'signal'] = -1

    # Only keep the first signal after a change to avoid repeated signals on consecutive days
    df['signal_change'] = df['signal'].replace(0, np.nan).ffill().fillna(0).diff().fillna(0)
    df['signal'] = df['signal_change'].apply(lambda x: int(x/1) if x != 0 else 0)
    df.drop(columns=['prev_sma_short', 'prev_sma_long', 'signal_change'], inplace=True)
    return df


# ----------------------------- Backtester -----------------------------

def backtest(df: pd.DataFrame, cfg: Config):
    """Very basic backtest: market orders at close on the signal day.
    Position sizing is cfg.position_size fraction of current portfolio.
    """
    cash = cfg.initial_cash
    position = 0.0  # number of shares
    portfolio_values = []
    trades = []
    print(df.head())
    for idx, row in df.iterrows():
        price = row['close'].item()
        signal = row['signal'].item()
        signal= 1 if (random.randint(1, 10) % 2) == 0 else -1
        print(row["signal"].item())
        # Execute buy   
        if signal == 1 and cash > 0:
            allocation = cash * cfg.position_size
            qty = allocation // price
            if qty > 0:
                cost = qty * price
                cash -= cost
                position += qty
                trades.append({'date': idx, 'type': 'BUY', 'price': price, 'qty': int(qty)})

        # Execute sell (close entire position)
        if signal == -1 and position > 0:
            proceeds = position * price
            cash += proceeds
            trades.append({'date': idx, 'type': 'SELL', 'price': price, 'qty': int(position)})
            position = 0

        portfolio_value = cash + position * price
        portfolio_values.append({'date': idx, 'portfolio_value': portfolio_value, 'cash': cash, 'position': position})

    pv = pd.DataFrame(portfolio_values).set_index('date')
    pv['returns'] = pv['portfolio_value'].pct_change().fillna(0)
    total_return = pv['portfolio_value'].iloc[-1] / cfg.initial_cash - 1
    ann_return = (1 + total_return) ** (252.0 / len(pv)) - 1 if len(pv) > 0 else 0
    sharpe = (pv['returns'].mean() / (pv['returns'].std() + 1e-9)) * np.sqrt(252) if pv['returns'].std() > 0 else 0
    drawdown = (pv['portfolio_value'].cummax() - pv['portfolio_value']).max()

    stats = {
        'initial_cash': cfg.initial_cash,
        'final_value': pv['portfolio_value'].iloc[-1] if len(pv) > 0 else cfg.initial_cash,
        'total_return': total_return,
        'annualized_return_est': ann_return,
        'sharpe_est': sharpe,
        'max_drawdown': drawdown,
        'n_trades': len(trades)
    }

    trades_df = pd.DataFrame(trades)
    return pv, trades_df, stats


# ----------------------------- Alpaca Execution (Optional) -----------------------------

def init_alpaca():
    if tradeapi is None:
        raise RuntimeError("alpaca_trade_api not installed. Install with `pip install alpaca-trade-api` to enable live/paper trading.")
    key = os.environ.get('ALPACA_API_KEY')
    secret = os.environ.get('ALPACA_SECRET_KEY')
    base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not key or not secret:
        raise RuntimeError('ALPACA_API_KEY and ALPACA_SECRET_KEY env vars must be set to enable Alpaca')

    api = tradeapi.REST(key, secret, base_url=base_url)
    return api


def execute_signals_with_alpaca(df: pd.DataFrame, cfg: Config):
    """Iterate through df and place market orders on signals in paper account. This is
    a naive implementation for demonstration. In production you must add error handling,
    rate-limiting, order confirmations, and careful position sizing.
    """
    api = init_alpaca()

    for idx, row in df.iterrows():
        signal = row['signal']
        price = row['close']

        if signal == 1:
            # Buy position_size fraction of buying power
            account = api.get_account()
            buying_power = float(account.cash)
            allocation = buying_power * cfg.position_size
            qty = int(allocation // price)
            if qty <= 0:
                print(f"{idx.date()}: Not enough cash to buy")
                continue
            try:
                order = api.submit_order(symbol=cfg.symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
                print(f"{idx.date()}: Submitted BUY order qty={qty} for {cfg.symbol}")
            except Exception as e:
                print(f"Alpaca buy error: {e}")

        if signal == -1:
            # Sell full position if exists
            try:
                positions = api.list_positions()
                pos = next((p for p in positions if p.symbol == cfg.symbol), None)
                if pos:
                    qty = int(float(pos.qty))
                    order = api.submit_order(symbol=cfg.symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
                    print(f"{idx.date()}: Submitted SELL order qty={qty} for {cfg.symbol}")
            except Exception as e:
                print(f"Alpaca sell error: {e}")

        # Respect API rate limits (paper trading), wait a bit
        time.sleep(0.5)


# ----------------------------- Main Entrypoint -----------------------------

def main():
    args = parse_args()
    cfg = Config(symbol=args.symbol, start=args.start, end=args.end, execute_trades=args.execute)

    print(f"Fetching data for {cfg.symbol} from {cfg.start} to {cfg.end or 'now'}...")
    df = fetch_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.timeframe)
    print(f"Downloaded {len(df)} rows")

    df = compute_indicators(df, cfg)
    df = generate_signals(df, cfg)

    print("Running backtest...")
    pv, trades_df, stats = backtest(df, cfg)

    print("Backtest stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print('\nRecent trades:')
    if not trades_df.empty:
        print(trades_df.tail(10).to_string(index=False))
    else:
        print("  No trades generated by strategy")

    if cfg.execute_trades:
        print('\nEXECUTE mode enabled: connecting to Alpaca (paper)')
        try:
            execute_signals_with_alpaca(df, cfg)
        except Exception as e:
            print(f"Error running Alpaca execution: {e}")


if __name__ == '__main__':
    main()
