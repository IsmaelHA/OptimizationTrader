# SentientTrader

> **SentientTrader** is an AI-driven trading framework that evolves from simple technical strategies to advanced deep learning and reinforcement learning models for research and educational purposes.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Phases](#phases)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [Disclaimer](#disclaimer)

---

## Project Overview

SentientTrader is designed to combine classical technical analysis with AI models such as RNNs, LLMs, and Reinforcement Learning to create a modular and extensible trading bot framework.

The project aims to provide hands-on experience in:

* Time series forecasting
* Natural language processing for sentiment analysis
* Reinforcement learning for dynamic trading strategies
* Backtesting and paper trading

---

## Phases

1. **Phase 1:** Simple MA Crossover + RSI strategy with backtesting and optional Alpaca paper trading.
2. **Phase 2:** Implement RNN/LSTM models to forecast short-term price movements.
3. **Phase 3:** Integrate LLM-based sentiment analysis from news and social media.
4. **Phase 4:** Reinforcement Learning agents for position sizing and portfolio optimization.
5. **Phase 5:** Hybrid ensemble models combining TA, RNN, LLM, and RL outputs for smarter decision-making.

---

## Features

* Market data ingestion from Yahoo Finance, Alpaca, Interactive Brokers.
* Modular design for easy integration of AI models.
* Backtesting engine with performance metrics (Sharpe ratio, max drawdown, CAGR).
* Paper trading support via Alpaca API.
* Future-ready for dashboards and real-time monitoring.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SentientTrader.git
cd SentientTrader

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run Phase 1 strategy example
python phase1_trading_bot.py --symbol AAPL --start 2022-01-01 --end 2024-12-31

# Optional: Enable paper trading (Alpaca API keys required)
python phase1_trading_bot.py --symbol AAPL --execute
```

> Future phases will include scripts for RNN forecasting, LLM sentiment analysis, and RL trading agents.

---

## Roadmap

* [x] Phase 1: MA Crossover + RSI backtesting
* [ ] Phase 2: RNN/LSTM forecasting
* [ ] Phase 3: LLM sentiment integration
* [ ] Phase 4: Reinforcement Learning agent
* [ ] Phase 5: Hybrid ensemble strategy
* [ ] Real-time monitoring dashboard
* [ ] Cloud deployment for paper/live trading

---

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Create a Pull Request.

---

## Disclaimer

This project is for **educational and research purposes only**. Trading involves substantial risk of loss. Do not use the code for live trading without thorough testing and understanding of risks. The author is **not responsible for any financial losses**.
