import os
import requests

IO_API_KEY = os.getenv("IO_API_KEY")
IO_API_URL = "https://api.intelligence.io.solutions/api/v1/workflows/run"

def io_fallback(ohlcv=None, indicators=None, all_covariates=None, volatility=None, rolling_mae=None, trade_count=None, min_sens=0.0002, max_sens=0.002, min_sl=0.005, max_sl=0.03, min_tp=0.01, max_tp=0.05, pnl_last_hour=0, avg_pnl_last_hour=0, switches_last_hour=0, max_drawdown_last_hour=0, TRADE_PERIOD_MINUTES=60, METRIC_WINDOW_MINUTES=60, trade_history=None):
    """
    Fallback to IO LLM for indicator/parameter selection. Forms a prompt identical to Gaia.
    Optionally includes recent trade history for context if provided.
    """
    # 1. Only indicators
    if ohlcv is not None and indicators is not None and all_covariates is not None and volatility is None:
        prompt = (
            "Given the following OHLCV and indicator values, select exactly 3 most important indicators for trading SOL/USDC right now.\n"
            "Respond with a comma-separated list of exactly 3 indicator names from this list only:\n"
            f"{', '.join(all_covariates)}\n"
            f"OHLCV: {ohlcv}\nIndicators: {indicators}\n"
            f"- Trading decision period: {TRADE_PERIOD_MINUTES} minutes\n"
            "\nIMPORTANT:\n"
            "- Respond ONLY with the indicator names, separated by commas.\n"
            "- DO NOT add any explanation, description, or extra text.\n"
            "- Example of correct response: rsi_14, macd, adx_14\n"
            "- Example of incorrect response: 'The most important indicators are rsi_14, macd, adx_14 because...'\n"
            "Your answer:"
        )
    # 2. Indicators + parameters
    else:
        prompt = (
            f"Given the following OHLCV and indicator values, select exactly 3 most important indicators for trading SOL/USDC right now, and recommend SENSITIVITY, STOP_LOSS, TAKE_PROFIT values.\n"
            f"Respond in the following format (no explanations!):\n"
            f"<indicator1>, <indicator2>, <indicator3>, <sensitivity>, <stop_loss>, <take_profit>\n"
            f"Example (do NOT copy values, use your own based on the data above): rsi_14, macd, adx_14, 0.0005, 0.01, 0.02\n"
            f"OHLCV: {ohlcv}\n"
            f"Indicators: {indicators}\n"
            f"Available indicators: {', '.join(all_covariates) if all_covariates else ''}\n"
            f"- Trading decision period: {TRADE_PERIOD_MINUTES} minutes\n"
            f"\nCurrent metrics:\n"
            f"- Volatility: {volatility:.4f}\n"
            f"- Rolling MAE: {rolling_mae:.4f}\n"
            f"- Trades ({METRIC_WINDOW_MINUTES}m): {trade_count}\n"
            f"- Switch-trades ({METRIC_WINDOW_MINUTES}m): {switches_last_hour}\n"
            f"- Realized PnL ({METRIC_WINDOW_MINUTES}m): {pnl_last_hour:.2f}\n"
            f"- Average PnL per trade: {avg_pnl_last_hour:.2f}\n"
            f"- Maximum drawdown ({METRIC_WINDOW_MINUTES}m): {max_drawdown_last_hour:.2f}\n"
            f"- Trading decision period: {TRADE_PERIOD_MINUTES} minutes\n"
        )
        # Add trade history if provided
        if trade_history and len(str(trade_history).strip()) > 0:
            prompt += ("\nRecent trade history (timestamp, action, price, amount, pnl, position):\n"
                       "Each line is: timestamp, action, price, amount, pnl, position.\n"
                       f"{trade_history}\n")
        prompt += (
            "\nIMPORTANT:\n"
            "- Respond ONLY in the specified format.\n"
            "- DO NOT copy the example values. Use the provided data to select appropriate values.\n"
            "- If there have been few or no trades recently, consider lowering the sensitivity to increase trading frequency.\n"
            "- DO NOT add any explanation, description, or extra text.\n"
            "Your answer:"
        )
    headers = {
        "Authorization": f"Bearer {IO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": "You are an aggressive intraday (day) trader and crypto trading expert.\n" + prompt,
        "agent_names": ["custom_agent"],
        "args": {
            "type": "custom",
            "name": "fallback",
            "objective": "Fallback objective",
            "instructions": "Return result"
        }
    }
    try:
        resp = requests.post(IO_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("output") or data, "io"
    except Exception as e:
        print(f"[IO API error] {e}")
        return None, "io" 