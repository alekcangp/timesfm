import os
import time
import numpy as np
import pandas as pd
from binance.client import Client
import ta
import csv
from datetime import datetime, timedelta
import timesfm
from colorama import init, Fore, Style
import collections
from dotenv import load_dotenv
load_dotenv()
from gaia_api import gaia_select_indicators, gaia_select_indicators_and_params
from config import SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT, SYMBOL, INTERVAL, LOOKBACK, LOOKAHEAD, TRADE_PERIOD_MINUTES, METRIC_WINDOW_MINUTES, TRADE_LOG, MAX_RUNTIME, MIN_TRADE_SIZE, POSITION_SIZE_PCT, MAX_DRAWDOWN_PCT
from io_api import io_fallback
from recall_api import fetch_all_balances, fetch_recall_balance, execute_trade, TOKEN_ADDRESS, USDC_ADDRESS

init(autoreset=True)

client = Client()

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend='torch',
        per_core_batch_size=32,
        horizon_len=128,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=50,
        model_dims=1280,
        use_positional_embedding=False,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)

all_covariates = [
    'adx_14', 'obv', 'returns', 'psar', 'williams_r', 'ultimate_osc',
    'donchian_high', 'donchian_low', 'cmf', 'zscore', 'rsi_14', 'macd', 'bb_high', 'bb_low'
]

def fetch_klines(symbol, interval, lookback, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback, requests_params={'timeout': 30})
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['open'] = df['open'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            # Technical indicators
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            # Additional indicators
            df['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['cci_14'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=14).cci()
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            # Lag features
            df['close_lag1'] = df['close'].shift(1)
            df['rsi_lag1'] = df['rsi_14'].shift(1)
            df['macd_lag1'] = df['macd'].shift(1)
            df['bb_high_lag1'] = df['bb_high'].shift(1)
            df['bb_low_lag1'] = df['bb_low'].shift(1)
            # TimesFM covariates
            df['adx_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['returns'] = df['close'].pct_change()
            df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
            df['ultimate_osc'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
            donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=20)
            df['donchian_high'] = donchian.donchian_channel_hband()
            df['donchian_low'] = donchian.donchian_channel_lband()
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
            df['zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
            df = df.dropna()
            return df
        except Exception as e:
            print(f"[fetch_klines] Error: {e}. Attempt {attempt+1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

def print_balance(token_balance, usdc_balance, current_price):
    total_value = usdc_balance + token_balance * current_price
    print(Fore.CYAN + f"[BALANCE] SOL: {token_balance:.4f}, USDC: {usdc_balance:.2f}, TOTAL: {total_value:.2f} USDC" + Style.RESET_ALL)

def log_trade(action, price, amount, pnl, position):
    file_exists = os.path.isfile(TRADE_LOG)
    with open(TRADE_LOG, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'action', 'price', 'amount', 'pnl', 'position'])
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            action,
            round(price, 2),
            round(amount, 1),
            round(pnl, 2),
            position
        ])

def normalize_series(series):
    if len(series) == 0:
        return series, 0.0, 1.0
    mean = np.mean(series) if len(series) > 0 else 0.0
    std = np.std(series) if len(series) > 0 else 1.0
    if std == 0:
        std = 1.0
    return (series - mean) / (std + 1e-8), mean, std

def denormalize_value(normed_value, mean, std):
    return normed_value * (std + 1e-8) + mean

def predict_signal_timesfm_with_covariates(df, lookback, lookahead, covariate_names):
    context = df.iloc[-(lookback + lookahead):].copy()
    target_series_raw = context['close'].values.astype(np.float32)
    target_series, target_mean, target_std = normalize_series(target_series_raw)
    dynamic_numerical_covariates = {}
    for name in covariate_names:
        cov_raw = context[name].values.astype(np.float32)
        cov_normed, _, _ = normalize_series(cov_raw)
        dynamic_numerical_covariates[name] = [cov_normed]
    point_forecast, _ = tfm.forecast_with_covariates(
        [target_series[:lookback]],
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        freq=[0],
    )
    next_price_normed = point_forecast[0][lookahead-1]
    next_price = denormalize_value(next_price_normed, target_mean, target_std)
    current_price = df['close'].iloc[-1]
    if next_price > current_price * (1 + SENSITIVITY):
        return 'long', next_price
    elif next_price < current_price * (1 - SENSITIVITY):
        return 'short', next_price
    else:
        return 'hold', next_price

def get_recent_trade_history(trade_timestamps, trade_log, window_minutes):
    """Return list of recent trades (dicts) for the last window_minutes."""
    now = datetime.now()
    window_seconds = window_minutes * 60
    recent_trades = []
    for i, ts in enumerate(trade_timestamps):
        if (now - ts).total_seconds() <= window_seconds:
            trade = trade_log[i]
            # trade_log: [timestamp, action, price, amount, pnl, position]
            recent_trades.append({
                'timestamp': trade[0],
                'action': trade[1],
                'price': float(trade[2]),
                'amount': round(float(trade[3]), 1),
                'pnl': float(trade[4]),
                'position': trade[5]
            })
    return recent_trades

def update_parameters_and_indicators(metrics, ohlcv, indicators, all_covariates, lookback, lookahead, tfm, trade_timestamps=None, trade_log=None, metric_window_minutes=None):
    """Update trading parameters and indicators using LLMs. Metrics are passed as a single argument."""
    # Prepare trade history for LLM prompt
    trade_history = []
    if trade_timestamps is not None and trade_log is not None and metric_window_minutes is not None:
        trade_history = get_recent_trade_history(trade_timestamps, trade_log, metric_window_minutes)
    # Format trade history for prompt
    trade_history_str = "\n".join([
        f"{t['timestamp']}, {t['action']}, {t['price']:.2f}, {t['amount']:.1f}, {t['pnl']:.2f}, {t['position']}"
        for t in trade_history
    ])
    # Add trade_history_str to prompt for LLM (Gaia/IO)
    # (Assume gaia_select_indicators_and_params/IO fallback can accept extra arg 'trade_history')
    covariate_names, params, param_source = gaia_select_indicators_and_params(
        ohlcv, indicators, all_covariates,
        metrics['volatility'], metrics['rolling_mae'], metrics['trades'],
        min_sens=0.0002, max_sens=0.002, min_sl=0.005, max_sl=0.03, min_tp=0.01, max_tp=0.05,
        pnl_last_hour=metrics['pnl'], avg_pnl_last_hour=metrics['avg_pnl'],
        switches_last_hour=metrics['switches'], max_drawdown_last_hour=metrics['drawdown'],
        TRADE_PERIOD_MINUTES=TRADE_PERIOD_MINUTES, METRIC_WINDOW_MINUTES=METRIC_WINDOW_MINUTES,
        trade_history=trade_history_str
    )
    SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT = params
    # Print source for indicators
    if param_source == 'gaia':
        print(f"[GAIA] Updated indicators: {covariate_names}")
    elif param_source == 'io':
        print(f"[IO] Updated indicators: {covariate_names}")
    else:
        print(f"[DEFAULT] Using default indicators: {covariate_names}")
    # Print source for parameters
    if param_source == 'gaia':
        print(f"[GAIA] Updated parameters: SENSITIVITY={SENSITIVITY}, STOP_LOSS={STOP_LOSS_PCT}, TAKE_PROFIT={TAKE_PROFIT_PCT}")
    elif param_source == 'io':
        print(f"[IO] Updated parameters: SENSITIVITY={SENSITIVITY}, STOP_LOSS={STOP_LOSS_PCT}, TAKE_PROFIT={TAKE_PROFIT_PCT}")
    else:
        print(f"[DEFAULT] Using default parameters: SENSITIVITY={SENSITIVITY}, STOP_LOSS={STOP_LOSS_PCT}, TAKE_PROFIT={TAKE_PROFIT_PCT}")
    print(f"[GAIA] Trades: {metrics['trades']}, Switches: {metrics['switches']}, Drawdown: {metrics['drawdown']:.2f}, PnL: {metrics['pnl']:.2f}")
    return covariate_names, params, param_source

def compute_metrics(df_full, model_mae_timestamps, trade_timestamps, trade_pnls, switch_timestamps, pnl_history, metric_window_minutes):
    """
    Computes metrics for LLM: volatility, rolling_mae, trades, pnl, switches, drawdown for the last metric_window_minutes.
    """
    returns = df_full['close'].pct_change().dropna()
    volatility = returns[-60:].std() if len(returns) >= 60 else returns.std()
    now = datetime.now()
    window_seconds = metric_window_minutes * 60
    # Rolling MAE
    recent_maes = [mae for ts, mae in model_mae_timestamps if (now - ts).total_seconds() <= window_seconds]
    rolling_mae_val = np.mean(recent_maes) if recent_maes else 0.0
    # Trades, pnl, avg pnl
    recent_indices = [i for i, t in enumerate(trade_timestamps) if (now - t).total_seconds() <= window_seconds]
    trades_last_window = len(recent_indices)
    pnl_last_window = sum(trade_pnls[i] for i in recent_indices) if recent_indices else 0.0
    avg_pnl_last_window = pnl_last_window / trades_last_window if trades_last_window > 0 else 0.0
    # Switches
    switches_last_window = sum(1 for t in switch_timestamps if (now - t).total_seconds() <= window_seconds)
    # Drawdown
    recent_pnls = [pnl_history[i] for i, t in enumerate(trade_timestamps) if (now - t).total_seconds() <= window_seconds]
    max_drawdown_window = 0.0
    if recent_pnls:
        peak = recent_pnls[0]
        dd = 0.0
        for p in recent_pnls:
            if p > peak:
                peak = p
            dd = min(dd, p - peak)
        max_drawdown_window = abs(dd)
    return dict(
        volatility=volatility,
        rolling_mae=rolling_mae_val,
        trades=trades_last_window,
        pnl=pnl_last_window,
        avg_pnl=avg_pnl_last_window,
        switches=switches_last_window,
        drawdown=max_drawdown_window
    )

def fetch_and_print_balance(df_full, token_balance, usdc_balance, current_price):
    """Fetch and print current SOL and USDC balances."""
    ohlcv = df_full[['open', 'high', 'low', 'close', 'volume']].tail(1).to_dict('records')[0]
    indicators = {name: float(df_full[name].iloc[-1]) for name in all_covariates}
    print_balance(token_balance, usdc_balance, current_price)
    return ohlcv, indicators

def run_forecast_and_trade(df, lookback, lookahead, covariate_names, tfm, SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT, position, entry_price, entry_amount, realized_pnl, trades, trade_timestamps, trade_pnls, pnl_history, switch_timestamps, balances, current_price, first_forecast, model_maes, naive_maes, model_accs, model_mae_timestamps, rolling_naive_mae, rolling_acc, state):
    """Run TimesFM forecast and execute trading logic."""
    token_balance = balances.get(TOKEN_ADDRESS, 0)
    usdc_balance = balances.get(USDC_ADDRESS, 0)
    if first_forecast:
        # --- Metrics ---
        context = df.iloc[-(LOOKBACK + LOOKAHEAD):]
        target_series_raw = context['close'].values.astype(np.float32)
        target_series, target_mean, target_std = normalize_series(target_series_raw)
        dynamic_numerical_covariates = {}
        for name in covariate_names:
            cov_raw = context[name].values.astype(np.float32)
            cov_normed, _, _ = normalize_series(cov_raw)
            dynamic_numerical_covariates[name] = [cov_normed]
        point_forecast, _ = tfm.forecast_with_covariates(
            [target_series[:LOOKBACK]],
            dynamic_numerical_covariates=dynamic_numerical_covariates,
            freq=[0],
        )
        y_true = target_series[LOOKBACK:]
        y_pred = point_forecast[0]
        model_mae_val = mae(y_true, y_pred)
        naive_mae_val = naive_mae(target_series)
        acc_val = direction_accuracy(y_true, y_pred)
        model_maes.append(model_mae_val)
        naive_maes.append(naive_mae_val)
        model_accs.append(acc_val)
        model_mae_timestamps.append((datetime.now(), model_mae_val))
        rolling_naive_mae.append(naive_mae_val)
        rolling_acc.append(acc_val)
        # [ROLLING_10] print removed: rolling_model_mae no longer exists (now time-based MAE)
    signal, next_price = predict_signal_timesfm_with_covariates(df, LOOKBACK, LOOKAHEAD, covariate_names)
    if first_forecast:
        first_forecast = False
    # Section header for new trade
    print(Fore.BLUE + Style.BRIGHT + "\n========== NEW TRADE ==========" + Style.RESET_ALL)
    print(f"[SIGNAL] {signal.upper()} at {current_price} (Forecast: {next_price:.2f})")
    pnl = 0
    # Debug print for state before trade logic
    # print(f"[DEBUG] Signal: {signal}, Position: {position}, Entry Amount: {entry_amount}, SOL: {token_balance}, USDC: {usdc_balance}")
    # Hybrid logic:
    if position == 'flat':
        if signal == 'long' and usdc_balance > 1:
            if state.get('entry_confirm') == 'long':
                sol_to_buy = max(MIN_TRADE_SIZE, usdc_balance * POSITION_SIZE_PCT / current_price)
                usdc_needed = sol_to_buy * current_price
                result = execute_trade(USDC_ADDRESS, TOKEN_ADDRESS, usdc_needed, reason="long entry")
                if not result.get('success'):
                    print(Fore.RED + f"[ERROR] Failed to execute long entry trade: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
                if result.get('success'):
                    print(f"{Fore.GREEN}[TRADE] Buying {sol_to_buy:.4f} SOL with {usdc_needed:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
                    position = 'long'
                    entry_price = current_price
                    entry_amount = sol_to_buy
                    time.sleep(2)
                    balances = fetch_all_balances()
                    token_balance = balances.get(TOKEN_ADDRESS, 0)
                    usdc_balance = balances.get(USDC_ADDRESS, 0)
                    log_trade('buy', current_price, sol_to_buy, pnl, position)
                    trades += 1
                    trade_timestamps.append(datetime.now())
                    trade_pnls.append(pnl)
                    pnl_history.append(realized_pnl)
                    total_usdc = usdc_balance + token_balance * current_price
                    print(Fore.CYAN + f"[BALANCE] SOL: {token_balance:.4f}, USDC: {usdc_balance:.2f}, TOTAL: {total_usdc:.2f} USDC" + Style.RESET_ALL)
                    state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'buy', round(current_price, 2), round(sol_to_buy, 1), round(pnl, 2), position])
                state['entry_confirm'] = None
            else:
                state['entry_confirm'] = 'long'
                return position, entry_price, entry_amount, realized_pnl, trades, trade_timestamps, trade_pnls, pnl_history, switch_timestamps, token_balance, usdc_balance, first_forecast, model_maes, naive_maes, model_accs, model_mae_timestamps, rolling_naive_mae, rolling_acc, state
        elif signal == 'short' and token_balance > MIN_TRADE_SIZE:
            if state.get('entry_confirm') == 'short':
                amount = max(MIN_TRADE_SIZE, token_balance * POSITION_SIZE_PCT)
                result = execute_trade(TOKEN_ADDRESS, USDC_ADDRESS, amount, reason="short entry")
                if not result.get('success'):
                    print(Fore.RED + f"[ERROR] Failed to execute short entry trade: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
                if result.get('success'):
                    usdc_received = amount * current_price
                    print(f"{Fore.RED}[TRADE] Selling {amount:.4f} SOL for {usdc_received:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
                    position = 'short'
                    entry_price = current_price
                    entry_amount = amount
                    time.sleep(2)
                    balances = fetch_all_balances()
                    token_balance = balances.get(TOKEN_ADDRESS, 0)
                    usdc_balance = balances.get(USDC_ADDRESS, 0)
                    log_trade('sell', current_price, amount, pnl, position)
                    trades += 1
                    trade_timestamps.append(datetime.now())
                    trade_pnls.append(pnl)
                    pnl_history.append(realized_pnl)
                    total_usdc = usdc_balance + token_balance * current_price
                    print(Fore.CYAN + f"[BALANCE] SOL: {token_balance:.4f}, USDC: {usdc_balance:.2f}, TOTAL: {total_usdc:.2f} USDC" + Style.RESET_ALL)
                    state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'sell', round(current_price, 2), round(amount, 1), round(pnl, 2), position])
                state['entry_confirm'] = None
            else:
                state['entry_confirm'] = 'short'
                return position, entry_price, entry_amount, realized_pnl, trades, trade_timestamps, trade_pnls, pnl_history, switch_timestamps, token_balance, usdc_balance, first_forecast, model_maes, naive_maes, model_accs, model_mae_timestamps, rolling_naive_mae, rolling_acc, state
        else:
            state['entry_confirm'] = None

    # 2. Switch confirmation logic
    elif position == 'long' and signal == 'short' and entry_amount > 0 and token_balance > MIN_TRADE_SIZE:
        if state.get('switch_confirm') == 'short':
            # Two consecutive short signals, perform switch
            print(Fore.YELLOW + Style.BRIGHT + "[SWITCH] Signal changed: closing long, opening short." + Style.RESET_ALL)
            usdc_received = entry_amount * current_price
            print(f"{Fore.RED}[TRADE] Selling {entry_amount:.4f} SOL for {usdc_received:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
            result = execute_trade(TOKEN_ADDRESS, USDC_ADDRESS, entry_amount, reason="signal switch")
            if not result.get('success'):
                print(Fore.RED + f"[ERROR] Failed to execute switch (long to short): {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
            if result.get('success'):
                # Calculate and log PnL for closing long
                pnl = (current_price - entry_price) * entry_amount
                realized_pnl += pnl
                log_trade('sell', current_price, entry_amount, pnl, 'flat')
                trades += 1
                trade_timestamps.append(datetime.now())
                trade_pnls.append(pnl)
                pnl_history.append(realized_pnl)
                state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'sell', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), 'flat'])
                # After closing long, open short
                amount = max(MIN_TRADE_SIZE, token_balance * POSITION_SIZE_PCT)
                result2 = execute_trade(TOKEN_ADDRESS, USDC_ADDRESS, amount, reason="short entry after switch")
                if not result2.get('success'):
                    print(Fore.RED + f"[ERROR] Failed to open short after switch: {result2.get('error', 'Unknown error')}" + Style.RESET_ALL)
                if result2.get('success'):
                    usdc_received2 = amount * current_price
                    print(f"{Fore.RED}[TRADE] Selling {amount:.4f} SOL for {usdc_received2:.2f} USDC at {current_price:.2f} (after switch){Style.RESET_ALL}")
                    position = 'short'
                    entry_price = current_price
                    entry_amount = amount
                    time.sleep(2)
                    balances = fetch_all_balances()
                    token_balance = balances.get(TOKEN_ADDRESS, 0)
                    usdc_balance = balances.get(USDC_ADDRESS, 0)
                    log_trade('sell', current_price, amount, 0, position)
                    state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'sell', round(current_price, 2), round(amount, 1), 0.00, position])
                    total_usdc = usdc_balance + token_balance * current_price
                    print(Fore.CYAN + f"[BALANCE] SOL: {token_balance:.4f}, USDC: {usdc_balance:.2f}, TOTAL: {total_usdc:.2f} USDC" + Style.RESET_ALL)
                    # After switch, reset confirmation
                    state['switch_confirm'] = None
            # After switch, reset confirmation
            state['switch_confirm'] = None
        else:
            state['switch_confirm'] = 'short'
            # print("[INFO] Switch to short requires confirmation. Waiting for second short signal.")
            state['last_signal'] = signal
            return position, entry_price, entry_amount, realized_pnl, trades, trade_timestamps, trade_pnls, pnl_history, switch_timestamps, token_balance, usdc_balance, first_forecast, model_maes, naive_maes, model_accs, model_mae_timestamps, rolling_naive_mae, rolling_acc, state
    elif position == 'short' and signal == 'long' and entry_amount > 0 and usdc_balance > 1:
        if state.get('switch_confirm') == 'long':
            # Two consecutive long signals, perform switch
            print(Fore.YELLOW + Style.BRIGHT + "[SWITCH] Signal changed: closing short, opening long." + Style.RESET_ALL)
            sol_to_buy = entry_amount
            usdc_needed = sol_to_buy * current_price
            print(f"{Fore.GREEN}[TRADE] Buying {sol_to_buy:.4f} SOL with {usdc_needed:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
            result = execute_trade(USDC_ADDRESS, TOKEN_ADDRESS, usdc_needed, reason="signal switch")
            if not result.get('success'):
                print(Fore.RED + f"[ERROR] Failed to execute switch (short to long): {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
            if result.get('success'):
                # Calculate and log PnL for closing short
                pnl = (entry_price - current_price) * entry_amount
                realized_pnl += pnl
                log_trade('buy', current_price, entry_amount, pnl, 'flat')
                trades += 1
                trade_timestamps.append(datetime.now())
                trade_pnls.append(pnl)
                pnl_history.append(realized_pnl)
                state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'buy', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), 'flat'])
                # After closing short, open long
                sol_to_buy2 = max(MIN_TRADE_SIZE, usdc_balance * POSITION_SIZE_PCT / current_price)
                usdc_needed2 = sol_to_buy2 * current_price
                result2 = execute_trade(USDC_ADDRESS, TOKEN_ADDRESS, usdc_needed2, reason="long entry after switch")
                if not result2.get('success'):
                    print(Fore.RED + f"[ERROR] Failed to open long after switch: {result2.get('error', 'Unknown error')}" + Style.RESET_ALL)
                if result2.get('success'):
                    print(f"{Fore.GREEN}[TRADE] Buying {sol_to_buy2:.4f} SOL with {usdc_needed2:.2f} USDC at {current_price:.2f} (after switch){Style.RESET_ALL}")
                    position = 'long'
                    entry_price = current_price
                    entry_amount = sol_to_buy2
                    time.sleep(2)
                    balances = fetch_all_balances()
                    token_balance = balances.get(TOKEN_ADDRESS, 0)
                    usdc_balance = balances.get(USDC_ADDRESS, 0)
                    log_trade('buy', current_price, sol_to_buy2, 0, position)
                    state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'buy', round(current_price, 2), round(sol_to_buy2, 1), 0.00, position])
                    total_usdc = usdc_balance + token_balance * current_price
                    print(Fore.CYAN + f"[BALANCE] SOL: {token_balance:.4f}, USDC: {usdc_balance:.2f}, TOTAL: {total_usdc:.2f} USDC" + Style.RESET_ALL)
                    # After switch, reset confirmation
                    state['switch_confirm'] = None
            # After switch, reset confirmation
            state['switch_confirm'] = None
        else:
            state['switch_confirm'] = 'long'
            # print("[INFO] Switch to long requires confirmation. Waiting for second long signal.")
            state['last_signal'] = signal
            return position, entry_price, entry_amount, realized_pnl, trades, trade_timestamps, trade_pnls, pnl_history, switch_timestamps, token_balance, usdc_balance, first_forecast, model_maes, naive_maes, model_accs, model_mae_timestamps, rolling_naive_mae, rolling_acc, state
    else:
        # Reset confirmation if signal changes or not a reversal
        if state.get('switch_confirm') is not None and signal != state.get('switch_confirm'):
            state['switch_confirm'] = None
    state['last_signal'] = signal

    # --- Exit logic ---
    if position == 'long':
        # print(f"[DEBUG] (Exit) Signal: {signal}, Position: {position}, Entry Amount: {entry_amount}, SOL: {token_balance}, USDC: {usdc_balance}")
        # Check that the position is open and entry_price is not None before arithmetic
        if entry_price is not None and current_price <= entry_price * (1 - STOP_LOSS_PCT) and entry_amount > 0:
            print(Fore.MAGENTA + "[EXIT] Stop-loss triggered: closing long position." + Style.RESET_ALL)
            result = execute_trade(TOKEN_ADDRESS, USDC_ADDRESS, entry_amount, reason="stop-loss")
            if not result.get('success'):
                print(Fore.RED + f"[ERROR] Failed to execute long stop-loss exit: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
            if result.get('success'):
                usdc_received = entry_amount * current_price
                print(f"{Fore.RED}[TRADE] Selling {entry_amount:.4f} SOL for {usdc_received:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
                pnl = (current_price - entry_price) * entry_amount
                realized_pnl += pnl
                log_trade('sell', current_price, entry_amount, pnl, 'flat')
                position = 'flat'
                entry_price = None  # Reset only after a full exit
                entry_amount = 0
                time.sleep(2)
                balances = fetch_all_balances()
                token_balance_real = balances.get(TOKEN_ADDRESS, 0)
                usdc_balance_real = balances.get(USDC_ADDRESS, 0)
                print_balance(token_balance_real, usdc_balance_real, current_price)
                state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'sell', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), position])
        elif entry_price is not None and current_price >= entry_price * (1 + TAKE_PROFIT_PCT) and entry_amount > 0:
            print(Fore.CYAN + "[EXIT] Take-profit triggered: closing long position." + Style.RESET_ALL)
            result = execute_trade(TOKEN_ADDRESS, USDC_ADDRESS, entry_amount, reason="take-profit")
            if not result.get('success'):
                print(Fore.RED + f"[ERROR] Failed to execute long take-profit exit: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
            if result.get('success'):
                usdc_received = entry_amount * current_price
                print(f"{Fore.RED}[TRADE] Selling {entry_amount:.4f} SOL for {usdc_received:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
                pnl = (current_price - entry_price) * entry_amount
                realized_pnl += pnl
                log_trade('sell', current_price, entry_amount, pnl, 'flat')
                position = 'flat'
                entry_price = None  # Reset only after a full exit
                entry_amount = 0
                time.sleep(2)
                balances = fetch_all_balances()
                token_balance_real = balances.get(TOKEN_ADDRESS, 0)
                usdc_balance_real = balances.get(USDC_ADDRESS, 0)
                print_balance(token_balance_real, usdc_balance_real, current_price)
                state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'sell', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), position])
        elif position == 'short':
            # print(f"[DEBUG] (Exit) Signal: {signal}, Position: {position}, Entry Amount: {entry_amount}, SOL: {token_balance}, USDC: {usdc_balance}")
            # Check that the position is open and entry_price is not None before arithmetic
            if entry_price is not None and current_price >= entry_price * (1 + STOP_LOSS_PCT) and entry_amount > 0:
                print(Fore.MAGENTA + "[EXIT] Stop-loss triggered: closing short position." + Style.RESET_ALL)
                usdc_needed = entry_amount * current_price
                result = execute_trade(USDC_ADDRESS, TOKEN_ADDRESS, usdc_needed, reason="stop-loss")
                if not result.get('success'):
                    print(Fore.RED + f"[ERROR] Failed to execute short stop-loss exit: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
                if result.get('success'):
                    print(f"{Fore.GREEN}[TRADE] Buying {entry_amount:.4f} SOL with {usdc_needed:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
                    pnl = (entry_price - current_price) * entry_amount
                    realized_pnl += pnl
                    log_trade('buy', current_price, entry_amount, pnl, 'flat')
                    position = 'flat'
                    entry_price = None  # Reset only after a full exit
                    entry_amount = 0
                    time.sleep(2)
                    balances = fetch_all_balances()
                    token_balance_real = balances.get(TOKEN_ADDRESS, 0)
                    usdc_balance_real = balances.get(USDC_ADDRESS, 0)
                    print_balance(token_balance_real, usdc_balance_real, current_price)
                    state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'buy', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), position])
            elif entry_price is not None and current_price <= entry_price * (1 - TAKE_PROFIT_PCT) and entry_amount > 0:
                print(Fore.CYAN + "[EXIT] Take-profit triggered: closing short position." + Style.RESET_ALL)
                usdc_needed = entry_amount * current_price
                result = execute_trade(USDC_ADDRESS, TOKEN_ADDRESS, usdc_needed, reason="take-profit")
                if not result.get('success'):
                    print(Fore.RED + f"[ERROR] Failed to execute short take-profit exit: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
                if result.get('success'):
                    print(f"{Fore.GREEN}[TRADE] Buying {entry_amount:.4f} SOL with {usdc_needed:.2f} USDC at {current_price:.2f}{Style.RESET_ALL}")
                    pnl = (entry_price - current_price) * entry_amount
                    realized_pnl += pnl
                    log_trade('buy', current_price, entry_amount, pnl, 'flat')
                    position = 'flat'
                    entry_price = None  # Reset only after a full exit
                    entry_amount = 0
                    time.sleep(2)
                    balances = fetch_all_balances()
                    token_balance_real = balances.get(TOKEN_ADDRESS, 0)
                    usdc_balance_real = balances.get(USDC_ADDRESS, 0)
                    print_balance(token_balance_real, usdc_balance_real, current_price)
                    state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'buy', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), position])

    # --- Close on HOLD logic ---
    if signal == 'hold' and position != 'flat' and entry_amount > 0:
        # print(f"[DEBUG] (Close on HOLD) Signal: {signal}, Position: {position}, Entry Amount: {entry_amount}, SOL: {token_balance}, USDC: {usdc_balance}")
        if position == 'long':
            result = execute_trade(TOKEN_ADDRESS, USDC_ADDRESS, entry_amount, reason="hold exit")
            if not result.get('success'):
                print(Fore.RED + f"[ERROR] Failed to close long position on HOLD: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
            if result.get('success'):
                usdc_received = entry_amount * current_price
                print(f"{Fore.RED}[TRADE] Selling {entry_amount:.4f} SOL for {usdc_received:.2f} USDC at {current_price:.2f} (HOLD exit){Style.RESET_ALL}")
                pnl = (current_price - entry_price) * entry_amount
                realized_pnl += pnl
                log_trade('sell', current_price, entry_amount, pnl, 'flat')
                position = 'flat'
                entry_price = None
                entry_amount = 0
                time.sleep(2)
                balances = fetch_all_balances()
                token_balance_real = balances.get(TOKEN_ADDRESS, 0)
                usdc_balance_real = balances.get(USDC_ADDRESS, 0)
                print_balance(token_balance_real, usdc_balance_real, current_price)
                state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'sell', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), position])
        elif position == 'short':
            usdc_needed = entry_amount * current_price
            result = execute_trade(USDC_ADDRESS, TOKEN_ADDRESS, usdc_needed, reason="hold exit")
            if not result.get('success'):
                print(Fore.RED + f"[ERROR] Failed to close short position on HOLD: {result.get('error', 'Unknown error')}" + Style.RESET_ALL)
            if result.get('success'):
                print(f"{Fore.GREEN}[TRADE] Buying {entry_amount:.4f} SOL with {usdc_needed:.2f} USDC at {current_price:.2f} (HOLD exit){Style.RESET_ALL}")
                pnl = (entry_price - current_price) * entry_amount
                realized_pnl += pnl
                log_trade('buy', current_price, entry_amount, pnl, 'flat')
                position = 'flat'
                entry_price = None
                entry_amount = 0
                time.sleep(2)
                balances = fetch_all_balances()
                token_balance_real = balances.get(TOKEN_ADDRESS, 0)
                usdc_balance_real = balances.get(USDC_ADDRESS, 0)
                print_balance(token_balance_real, usdc_balance_real, current_price)
                state['trade_log'].append([time.strftime('%Y-%m-%d %H:%M:%S'), 'buy', round(current_price, 2), round(entry_amount, 1), round(pnl, 2), position])

    # Always print PnL at the end of trade logic
    if realized_pnl > 0:
        pnl_str = Fore.GREEN + f"{realized_pnl:.2f}" + Style.RESET_ALL
    elif realized_pnl < 0:
        pnl_str = Fore.RED + f"{realized_pnl:.2f}" + Style.RESET_ALL
    else:
        pnl_str = f"{realized_pnl:.2f}"
    print(f"[PNL] Realized PnL: {pnl_str} USDC | Trades: {trades}")
    print(Fore.BLUE + Style.BRIGHT + "========== END OF TRADE ==========" + Style.RESET_ALL)
    return position, entry_price, entry_amount, realized_pnl, trades, trade_timestamps, trade_pnls, pnl_history, switch_timestamps, token_balance, usdc_balance, first_forecast, model_maes, naive_maes, model_accs, model_mae_timestamps, rolling_naive_mae, rolling_acc, state

def print_pnl_and_stats(model_maes, naive_maes, model_accs, realized_pnl, trades, elapsed_minutes=None):
    """Print PnL and trading statistics with the entire summary line in magenta."""
    summary_color = Fore.MAGENTA + Style.BRIGHT
    reset = Style.RESET_ALL
    if elapsed_minutes is not None:
        print(f"{summary_color}[SUMMARY] {elapsed_minutes:.1f} min trading complete. Realized PnL: {realized_pnl:.2f} USDC | Total trades: {trades}{reset}")
    else:
        print(f"{summary_color}[SUMMARY] Trading period complete. Realized PnL: {realized_pnl:.2f} USDC | Total trades: {trades}{reset}")

    if model_maes and naive_maes and model_accs:
        print(f"{summary_color}[SUMMARY] Avg Model MAE={np.mean(model_maes):.4f}, Avg Naive MAE={np.mean(naive_maes):.4f}, Avg Direction Acc={np.mean(model_accs):.3f}{reset}")
    else:
        print(f"{summary_color}[SUMMARY] Not enough data for MAE/accuracy stats.{reset}")

def mae(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    return np.mean(np.abs(y_true - y_pred))

def direction_accuracy(y_true, y_pred):
    # Count how many times the direction of movement matched
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    return np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_true[:-1]))

def naive_mae(y):
    # MAE of naive model: price will not change
    if len(y) < 2:
        return 0.0
    return np.mean(np.abs(y[1:] - y[:-1]))

# For manual analysis you can import:
# from indicator_analysis import permutation_importance

def parse_covariate_names(llm_response, all_covariates):
    """Parses LLM response string into a list of indicator names, filters only valid names."""
    if isinstance(llm_response, list):
        names = llm_response
    else:
        # Try splitting on comma first
        names = [x.strip() for x in str(llm_response).split(',')]
        # If only one element and it contains whitespace, try splitting on whitespace
        if len(names) == 1 and ' ' in names[0]:
            names = [x.strip() for x in names[0].split()]
    valid = [name for name in names if name in all_covariates]
    if not valid:
        print(f"[LLM PARSE WARNING] Could not extract valid indicators from LLM response: {llm_response}")
        valid = all_covariates[:3]
    return valid

def parse_covariates_and_params(llm_response, all_covariates):
    """Parses combined LLM response: <ind1>, <ind2>, <ind3>, <sens>, <sl>, <tp>.
    Returns (list of indicators, list of parameters)."""
    if isinstance(llm_response, (list, tuple)) and len(llm_response) == 3 and isinstance(llm_response[0], list):
        # Already parsed (selected, params, source)
        return llm_response[0], llm_response[1]
    parts = [x.strip() for x in str(llm_response).split(',')]
    inds = [name for name in parts[:3] if name in all_covariates]
    try:
        params = [float(x) for x in parts[3:6]]
    except Exception:
        params = [0.0005, 0.01, 0.02]
    if len(inds) != 3:
        print(f"[LLM PARSE WARNING] Could not extract 3 valid indicators from LLM response: {llm_response}")
        inds = all_covariates[:3]
    if len(params) != 3:
        print(f"[LLM PARSE WARNING] Could not extract 3 valid params from LLM response: {llm_response}")
        params = [0.0005, 0.01, 0.02]
    return inds, params

def init_state(optimal_covariates):
    """Initializes state and variables for the main loop."""
    state = dict(
        position='flat',
        entry_price=None,
        entry_amount=0,
        realized_pnl=0,
        trades=0,
        first_forecast=True,
        covariate_names=optimal_covariates,
        model_maes=[],
        naive_maes=[],
        model_accs=[],
        model_mae_timestamps=[],
        rolling_naive_mae=collections.deque(maxlen=10),
        rolling_acc=collections.deque(maxlen=10),
        last_gaia_update=datetime.now(),
        trade_timestamps=[],
        trade_pnls=[],
        switch_timestamps=[],
        pnl_history=[],
        trade_log=[],  # In-memory trade log for LLM window
        last_signal=None,  # For switch confirmation
        switch_confirm=None,  # For switch confirmation
        entry_confirm=None,   # For entry confirmation
    )
    return state

def main():
    """Main trading loop (refactored)."""
    global SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT
    print("[INFO] Fetching initial data...")
    df_full = fetch_klines(SYMBOL, INTERVAL, LOOKBACK + 50)
    if df_full is None or df_full.empty:
        print(Fore.RED + '[ERROR] Failed to fetch initial data. Exiting.' + Style.RESET_ALL)
        return
    ohlcv = df_full[['open', 'high', 'low', 'close', 'volume']].tail(1).to_dict('records')[0]
    indicators = {name: float(df_full[name].iloc[-1]) for name in all_covariates}
    # Prepare empty trade history string for initial call
    trade_history_str = ""
    optimal_covariates, source = gaia_select_indicators(ohlcv, indicators, all_covariates, trade_history=trade_history_str)
    if source == 'gaia':
        print("[INFO] Getting indicator recommendations from Gaia LLM...")
        print("[INFO] Gaia recommends using indicators:", optimal_covariates)
    elif source == 'io':
        print("[INFO] IO recommends using indicators:", optimal_covariates)
    else:
        print("[INFO] Using default indicators:", optimal_covariates)
    covariate_names = parse_covariate_names(optimal_covariates, all_covariates)
    # Use config params at startup
    params = [SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT]
    state = init_state(covariate_names)
    start_time = datetime.now()
    balances = fetch_all_balances()
    token_balance = balances.get(TOKEN_ADDRESS, 0)
    usdc_balance = balances.get(USDC_ADDRESS, 0)
    current_price = 0
    first_metrics_update = True
    # --- Drawdown protection ---
    initial_total_value = usdc_balance + token_balance * (df_full['close'].iloc[-1] if not df_full.empty else 0)
    max_total_value = initial_total_value
    while (datetime.now() - start_time).total_seconds() < MAX_RUNTIME * 60:
        # --- Drawdown check ---
        try:
            balances = fetch_all_balances()
            token_balance = balances.get(TOKEN_ADDRESS, 0)
            usdc_balance = balances.get(USDC_ADDRESS, 0)
            df_tmp = fetch_klines(SYMBOL, INTERVAL, max(LOOKBACK, 20))
            current_price = df_tmp['close'].iloc[-1]
            total_value = usdc_balance + token_balance * current_price
        except Exception as e:
            print(f"[ERROR] Failed to fetch balances or price: {e}. Skipping drawdown check this iteration.")
            time.sleep(TRADE_PERIOD_MINUTES * 60)
            continue
        if total_value > 0:
            if total_value > max_total_value:
                max_total_value = total_value
            drawdown_pct = (max_total_value - total_value) / max_total_value if max_total_value > 0 else 0
            if drawdown_pct >= MAX_DRAWDOWN_PCT:
                print(Fore.RED + Style.BRIGHT + f"[STOP] Max drawdown reached: {drawdown_pct*100:.2f}% (limit: {MAX_DRAWDOWN_PCT*100:.2f}%). Trading stopped." + Style.RESET_ALL)
                break
        else:
            print("[WARNING] total_value is zero, skipping drawdown check this iteration.")
        # --- Update indicators and parameters every METRIC_WINDOW_MINUTES ---
        if (datetime.now() - state['last_gaia_update']).total_seconds() >= METRIC_WINDOW_MINUTES * 60:
            df_full = fetch_klines(SYMBOL, INTERVAL, LOOKBACK + 50)
            if df_full is None or df_full.empty:
                print(Fore.RED + '[ERROR] Failed to fetch data during trading loop. Skipping iteration.' + Style.RESET_ALL)
                time.sleep(TRADE_PERIOD_MINUTES * 60)
                continue
            ohlcv, indicators = fetch_and_print_balance(df_full, token_balance, usdc_balance, current_price)
            metrics = compute_metrics(
                df_full, state['model_mae_timestamps'], state['trade_timestamps'],
                state['trade_pnls'], state['switch_timestamps'], state['pnl_history'], METRIC_WINDOW_MINUTES
            )
            # Only after first metrics window, update both indicators and params from LLM
            state['covariate_names'], params, param_source = update_parameters_and_indicators(
                metrics, ohlcv, indicators, all_covariates, LOOKBACK, LOOKAHEAD, tfm,
                trade_timestamps=state['trade_timestamps'], trade_log=state['trade_log'], metric_window_minutes=METRIC_WINDOW_MINUTES
            )
            SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT = params
            state['last_gaia_update'] = datetime.now()
            elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
            print_pnl_and_stats(state['model_maes'], state['naive_maes'], state['model_accs'], state['realized_pnl'], state['trades'], elapsed_minutes)
        df = fetch_klines(SYMBOL, INTERVAL, LOOKBACK + 50)
        if df is None or df.empty:
            print(Fore.RED + '[ERROR] Failed to fetch data during trading loop. Skipping iteration.' + Style.RESET_ALL)
            time.sleep(TRADE_PERIOD_MINUTES * 60)
            continue
        current_price = df['close'].iloc[-1]
        balances = fetch_all_balances()
        token_balance = balances.get(TOKEN_ADDRESS, 0)
        usdc_balance = balances.get(USDC_ADDRESS, 0)
        # --- Run forecast and trade logic ---
        (state['position'], state['entry_price'], state['entry_amount'], state['realized_pnl'], state['trades'],
         state['trade_timestamps'], state['trade_pnls'], state['pnl_history'], state['switch_timestamps'],
         token_balance, usdc_balance, state['first_forecast'], state['model_maes'], state['naive_maes'],
         state['model_accs'], state['model_mae_timestamps'], state['rolling_naive_mae'], state['rolling_acc'], state) = run_forecast_and_trade(
            df, LOOKBACK, LOOKAHEAD, state['covariate_names'], tfm, SENSITIVITY, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
            state['position'], state['entry_price'], state['entry_amount'], state['realized_pnl'], state['trades'],
            state['trade_timestamps'], state['trade_pnls'], state['pnl_history'], state['switch_timestamps'],
            balances, current_price, state['first_forecast'], state['model_maes'], state['naive_maes'],
            state['model_accs'], state['model_mae_timestamps'], state['rolling_naive_mae'], state['rolling_acc'], state
        )
        time.sleep(TRADE_PERIOD_MINUTES * 60)

if __name__ == '__main__':
    main() 