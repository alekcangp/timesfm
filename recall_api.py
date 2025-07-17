import os
import requests
from config import TOKEN_ADDRESS, USDC_ADDRESS, SLIPPAGE_TOLERANCE

RECALL_API_KEY = os.getenv('RECALL_API_KEY')
RECALL_BASE_URL = "https://api.sandbox.competitions.recall.network"


def fetch_all_balances():
    url = f"{RECALL_BASE_URL}/api/agent/balances"
    headers = {"Authorization": f"Bearer {RECALL_API_KEY}", "accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if not data.get('success'):
            print(f"[Recall Balance Error] API call not successful: {data}")
            return {}
        balances = {bal['tokenAddress']: float(bal.get('amount', 0)) for bal in data.get('balances', [])}
        return balances
    except Exception as e:
        print(f"[Recall Balance Error] {e}")
        return {}

# For backward compatibility:
def fetch_recall_balance(token_address):
    balances = fetch_all_balances()
    return balances.get(token_address, 0)

def execute_trade(from_token, to_token, amount, reason="AI trade"):
    url = f"{RECALL_BASE_URL}/api/trade/execute"
    headers = {
        "Authorization": f"Bearer {RECALL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "fromToken": from_token,
        "toToken": to_token,
        "amount": str(amount),
        "reason": reason,
        "slippageTolerance": SLIPPAGE_TOLERANCE
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        return resp.json()
    except Exception as e:
        print(f"[Recall Trade Error] {e}")
        return {"error": str(e)} 