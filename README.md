# TimesFM Crypto Trading Agent

## Overview

This project is an LLM-driven crypto trading bot for any token/USDC pair on a single specified network (e.g., Ethereum, Base, Solana, etc.), using TimesFM for forecasting, Gaia/IO LLMs for adaptive parameter selection, and Recall for trade execution and logging. It features robust logging, drawdown protection, and modular configuration.

**Note:** The bot is not cross-chain. You can configure it to trade any token supported by your chosen network by setting `TOKEN_ADDRESS` and `USDC_ADDRESS` in `config.py`.

---

## Key Features

- **Adaptive LLM-Driven Strategy**: Uses Gaia and IO LLMs to select optimal trading indicators and parameters based on recent market data and real trade history.
- **Forecasting with TimesFM**: Employs Google's TimesFM model for high-quality price forecasting and signal generation.
- **On-Chain Trading**: Executes real trades via the Recall API.
- **Robust Logging**: All trades are logged in `trade_log.csv` with timestamp, action, price, amount, PnL, and position (all numbers rounded for clarity).
- **Clear Console UI**: Color-coded, human-friendly output for all signals, trades, and balance updates.
- **Centralized Configuration**: All parameters (trading, time, API, tokens, etc.) are managed in `config.py`.

---

## How the Agent Works

### 1. Data Fetching & Preprocessing
- Fetches recent OHLCV data for SOL/USDC from Binance.
- Computes a wide set of technical indicators (see `all_covariates` in code).

### 2. LLM-Driven Indicator & Parameter Selection
- At startup, requests only indicator selection from Gaia/IO LLMs.
- After enough trade history is available, periodically (every `METRIC_WINDOW_MINUTES`) requests both indicators and trading parameters (sensitivity, stop-loss, take-profit) from the LLMs.
- The LLM prompt includes:
  - Latest OHLCV and indicator values
  - All available indicators
  - Current trading metrics (volatility, rolling MAE, trade count, switches, PnL, drawdown)
  - **Recent trade history** (timestamp, action, price, amount, pnl, position) with a format explanation
- If the LLM response cannot be parsed, falls back to config defaults.

### 3. Forecasting & Signal Generation
- Uses TimesFM to forecast future prices and generate trading signals (long/short/hold) based on the selected indicators and parameters.

### 4. Trade Execution
- Executes trades via the Recall API, always using real, up-to-date balances.
- After every trade, fetches and prints the new balances and total portfolio value in USDC.

### 5. Logging & Transparency
- Every successful trade is logged in `trade_log.csv` with 6 fields:
  - timestamp, action, price (2 decimals), amount (1 decimal), pnl (2 decimals), position
- Console output is color-coded (green for buys, red for sells, yellow for switches, etc.) for clarity.


---

## Configuration

Edit `config.py` to set your API keys, trading parameters, and token addresses. The bot can be configured for any token/USDC pair on a single network:

- `TOKEN_ADDRESS` (the token you want to trade, e.g., an ERC-20 or SPL token address)
- `USDC_ADDRESS` (the USDC contract address for your network)
- Trading parameters (SENSITIVITY, STOP_LOSS_PCT, etc.)
- Drawdown protection, logging, and more

**To change the traded token or network:**
- Set `TOKEN_ADDRESS` and `USDC_ADDRESS` to the correct contract addresses for your chosen network.
- Ensure your API keys and endpoints are compatible with the selected network.

**The bot does not perform cross-chain trading.** It operates on a single network at a time, as specified in your configuration.

---

## Setup & Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set environment variables** for API keys and token addresses (see code for details).
3. **Run the agent**:
   ```bash
   python timesfm_trading_agent.py
   ```

---

## File Structure

- `timesfm_trading_agent.py` — Main trading loop and logic
- `gaia_api.py` — Gaia LLM API integration
- `io_api.py` — IO LLM API integration
- `recall_api.py` — Recall trading API integration
- `config.py` — Centralized configuration
- `trade_log.csv` — Trade log (auto-generated)
- `requirements.txt` — Python dependencies

---

## Notes
- The agent is designed for research and educational purposes. Use with caution and at your own risk.
- For best results, ensure your API keys and token addresses are set correctly and securely.
- The agent can be extended to support other pairs, exchanges, or strategies by modifying the config and code.


---

## License
MIT 