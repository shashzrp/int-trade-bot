from dotenv import load_dotenv
import os

load_dotenv()
from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
    paper=True,
)

acct = client.get_account()
print(f"Account ID:     {acct.account_number}")
print(f"Status:         {acct.status}")
print(f"Equity:         ${acct.equity}")
print(f"Buying power:   ${acct.buying_power}")
print(f"Pattern day trader: {acct.pattern_day_trader}")