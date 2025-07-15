import os
import requests
from io_api import io_fallback
from config import SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT

GAIA_API_KEY = os.getenv("GAIA_API_KEY")
GAIA_API_URL = "https://llama3b.gaia.domains/v1/chat/completions"

def gaia_select_indicators(ohlcv, indicators, all_covariates, trade_history=None):
    prompt = (
        "Given the following OHLCV and indicator values, select exactly 3 most important indicators for trading SOL/USDC right now.\n"
        "The Trading decision period (in minutes) determines how frequently trading decisions are made and should influence your recommended indicators.\n"
        "Respond with a comma-separated list of exactly 3 indicator names from this list only:\n"
        f"{', '.join(all_covariates)}\n"
        f"OHLCV: {ohlcv}\nIndicators: {indicators}\n"
        f"- Trading decision period: {{TRADE_PERIOD_MINUTES}} minutes\n"
        "\nIMPORTANT:\n"
        "- Respond ONLY with the indicator names, separated by commas.\n"
        "- DO NOT add any explanation, description, or extra text.\n"
        "- Example of correct response: rsi_14, macd, adx_14\n"
        "- Example of incorrect response: 'The most important indicators are rsi_14, macd, adx_14 because...'\n"
        "Your answer:"
    )
    payload = {
        "model": "llama3b",
        "messages": [
            {"role": "system", "content": "You are an aggressive intraday (day) trader and crypto trading expert."},
            {"role": "user", "content": prompt}
        ]
    }
    headers = {
        "Authorization": f"Bearer {GAIA_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(GAIA_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        selected = [x.strip() for x in content.split(",") if x.strip() in all_covariates]
        return (selected[:3] if len(selected) >= 3 else all_covariates[:3], 'gaia')
    except requests.exceptions.HTTPError as e:
            print(f"[GAIA] Error: {e}, falling back to IO...")
            return io_fallback(ohlcv, indicators, all_covariates, trade_history=trade_history)
    except Exception as e:
        print(f"[GAIA] Unexpected error: {e}, falling back to IO...")
        return io_fallback(ohlcv, indicators, all_covariates, trade_history=trade_history)

def parse_io_result(io_result, all_covariates, default_params):
    if isinstance(io_result, str):
        parts = [x.strip() for x in io_result.split(',')]
        inds = [x for x in parts[:3] if x in all_covariates]
        try:
            params = [float(x) for x in parts[3:6]]
        except Exception:
            params = default_params
        if len(inds) != 3:
            inds = all_covariates[:3]
        if len(params) != 3:
            params = default_params
        return inds, params, 'io'
    else:
        return all_covariates[:3], default_params, 'io'

def gaia_select_indicators_and_params(ohlcv, indicators, all_covariates, volatility, rolling_mae, trade_count, min_sens=0.0002, max_sens=0.002, min_sl=0.005, max_sl=0.03, min_tp=0.01, max_tp=0.05, pnl_last_hour=0, avg_pnl_last_hour=0, switches_last_hour=0, max_drawdown_last_hour=0, TRADE_PERIOD_MINUTES=60, METRIC_WINDOW_MINUTES=60, trade_history=None):
    """
    Select optimal indicators and parameters using Gaia LLM, with optional trade history for context.
    """
    # Build prompt
    prompt = (
        f"Given the following OHLCV and indicator values, select exactly 3 most important indicators for trading SOL/USDC right now, and recommend SENSITIVITY, STOP_LOSS, TAKE_PROFIT values.\n"
        f"Respond in the following format (no explanations!):\n"
        f"<indicator1>, <indicator2>, <indicator3>, <sensitivity>, <stop_loss>, <take_profit>\n"
        f"Example: rsi_14, macd, adx_14, 0.001, 0.01, 0.02\n"
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
    if trade_history and len(trade_history.strip()) > 0:
        prompt += ("\nRecent trade history (timestamp, action, price, amount, pnl, position):\n"
                   "Each line is: timestamp, action, price, amount, pnl, position.\n"
                   f"{trade_history}\n")
    prompt += (
        "\nIMPORTANT:\n"
        "- Respond ONLY in the specified format.\n"
        "- DO NOT add any explanation, description, or extra text.\n"
        "Your answer:"
    )
    headers = {
        "Authorization": f"Bearer {GAIA_API_KEY}",
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
        resp = requests.post(GAIA_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parts = [x.strip() for x in content.split(",")]
        selected = [x for x in parts[:3] if x in all_covariates]
        values = []
        for x in parts[3:6]:
            try:
                values.append(float(x))
            except:
                pass
        if len(selected) == 3 and len(values) == 3:
            return (selected, values, 'gaia')
    except requests.exceptions.HTTPError as e:
        if hasattr(e.response, 'status_code') and e.response.status_code in (401, 402):
            print("[GAIA] Auth error or payment required. Falling back to IO...")
            io_result, _ = io_fallback(ohlcv, indicators, all_covariates, volatility, rolling_mae, trade_count, min_sens, max_sens, min_sl, max_sl, min_tp, max_tp, pnl_last_hour, avg_pnl_last_hour, switches_last_hour, max_drawdown_last_hour, TRADE_PERIOD_MINUTES, METRIC_WINDOW_MINUTES, trade_history=trade_history)
            if isinstance(io_result, str):
                parts = [x.strip() for x in io_result.split(',')]
                inds = [x for x in parts[:3] if x in all_covariates]
                try:
                    params = [float(x) for x in parts[3:6]]
                except Exception:
                    params = [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT]
                if len(inds) != 3:
                    inds = all_covariates[:3]
                if len(params) != 3:
                    params = [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT]
                return inds, params, 'io'
            else:
                return all_covariates[:3], [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT], 'io'
    except Exception as e:
        print(f"[GAIA] Error: {e}, falling back to IO...")
        io_result, _ = io_fallback(ohlcv, indicators, all_covariates, volatility, rolling_mae, trade_count, min_sens, max_sens, min_sl, max_sl, min_tp, max_tp, pnl_last_hour, avg_pnl_last_hour, switches_last_hour, max_drawdown_last_hour, TRADE_PERIOD_MINUTES, METRIC_WINDOW_MINUTES, trade_history=trade_history)
        if isinstance(io_result, str):
            parts = [x.strip() for x in io_result.split(',')]
            inds = [x for x in parts[:3] if x in all_covariates]
            try:
                params = [float(x) for x in parts[3:6]]
            except Exception:
                params = [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT]
            if len(inds) != 3:
                inds = all_covariates[:3]
            if len(params) != 3:
                params = [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT]
            return inds, params, 'io'
        else:
            return all_covariates[:3], [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT], 'io'
    # Fallback default
    return all_covariates[:3], [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT], 'default' 