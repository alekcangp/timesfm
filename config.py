# === Trading Bot Global Configuration ===

# Trading parameters
SENSITIVITY = 0.001  # Default sensitivity for signal threshold
STOP_LOSS_PCT = 0.005  # Default stop-loss percentage
TAKE_PROFIT_PCT = 0.02  # Default take-profit percentage
POSITION_SIZE_PCT = 0.1  # Fraction of balance to use per trade (updated from CONFIG)
MIN_TRADE_SIZE = 0.01  # Minimum trade size in SOL

# Symbol and data settings
SYMBOL = 'SOLUSDC'  # Trading pair symbol
INTERVAL = '5m'  # Candle interval (use string if Client is not imported)
LOOKBACK = 100  # Number of candles to look back for indicators
LOOKAHEAD = 2   # Number of candles to look ahead for some logic

# Time settings
TRADE_PERIOD_MINUTES = 10  # How often to make trading decisions (updated from CONFIG)
METRIC_WINDOW_MINUTES = 60  # Window for metrics (PnL, drawdown, etc.)
MAX_RUNTIME = 60 * 24  # Max runtime in minutes (from CONFIG)

# Logging
TRADE_LOG = 'trade_log.csv'  # Path to trade log file

# IO/GAIA API settings
MIN_SENS = 0.0002  # Minimum sensitivity for LLM prompt
MAX_SENS = 0.002   # Maximum sensitivity for LLM prompt
MIN_SL = 0.005     # Minimum stop-loss for LLM prompt
MAX_SL = 0.03      # Maximum stop-loss for LLM prompt
MIN_TP = 0.01      # Minimum take-profit for LLM prompt
MAX_TP = 0.05      # Maximum take-profit for LLM prompt

# Prompt/LLM agent settings
GAIA_API_URL = "https://llama3b.gaia.domains/v1/chat/completions"
IO_API_URL = "https://api.intelligence.io.solutions/api/v1/workflows/run"

# Recall API settings
RECALL_BASE_URL = "https://api.sandbox.competitions.recall.network"

# Token addresses (should be set via environment variables in production)
SOL_ADDRESS = "So11111111111111111111111111111111111111112"  # Set via os.getenv('SOL_ADDRESS') in code
USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Set via os.getenv('USDC_ADDRESS') in code

# Slippage tolerance for trades
SLIPPAGE_TOLERANCE = "0.2"  # (e.g., 0.2%)

# Drawdown protection
MAX_DRAWDOWN_PCT = 0.2  # Maximum allowed portfolio drawdown (fraction, e.g. 0.2 = 20%). Trading stops if exceeded.

# Add any other global config parameters here as needed 