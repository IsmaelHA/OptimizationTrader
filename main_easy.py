import pandas as pd
import numpy as np
import talib
import yfinance as yf
from ib_insync import *

# -------------------------------
# CONFIGURATION
# -------------------------------
SYMBOL = "AAPL"              # Stock symbol
CASH = 100000                # Starting cash
SHORT_WINDOW = 20            # Short-term SMA
LONG_WINDOW = 50             # Long-term SMA
RSI_PERIOD = 14              # RSI lookback
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# -------------------------------
# CONNECT TO INTERACTIVE BROKERS
# -------------------------------
# Make sure TWS or IB Gateway is running with API enabled
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Paper trading default port is 7497

contract = Stock(SYMBOL, 'SMART', 'USD')

# -------------------------------
# FETCH HISTORICAL DATA (BACKTEST)
# -------------------------------
df = yf.download(SYMBOL, start="2022-01-01", end="2024-01-01")
df['SMA_Short'] = df['Close'].rolling(SHORT_WINDOW).mean()
df['SMA_Long'] = df['Close'].rolling(LONG_WINDOW).mean()
df['RSI'] = talib.RSI(df['Close'], timeperiod=RSI_PERIOD)

# Signal logic
df['Signal'] = 0
df['Signal'] = np.where((df['SMA_Short'] > df['SMA_Long']) & (df['RSI'] < RSI_OVERBOUGHT), 1, df['Signal'])
df['Signal'] = np.where((df['SMA_Short'] < df['SMA_Long']) & (df['RSI'] > RSI_OVERSOLD), -1, df['Signal'])

# Backtest simulation
df['Position'] = df['Signal'].shift(1).fillna(0)
df['Returns'] = df['Close'].pct_change()
df['Strategy'] = df['Position'] * df['Returns']
cumulative_returns = (1 + df['Strategy']).cumprod()

print("Backtest results:")
print("Total return:", cumulative_returns.iloc[-1] - 1)

# -------------------------------
# PLACE TRADES ON IBKR (LIVE PAPER)
# -------------------------------
latest_signal = df['Signal'].iloc[-1]

if latest_signal == 1:
    print("BUY signal detected")
    order = MarketOrder("BUY", 10)  # example: buy 10 shares
    trade = ib.placeOrder(contract, order)
elif latest_signal == -1:
    print("SELL signal detected")
    order = MarketOrder("SELL", 10)  # example: sell 10 shares
    trade = ib.placeOrder(contract, order)
else:
    print("No trade signal")

ib.disconnect()
