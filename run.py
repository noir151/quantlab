from core.backtest import BacktestEngine
from strategies.sma_cross import SMACross
from data.data_generator import generate_prices

data = generate_prices()

strategy = SMACross()
engine = BacktestEngine(data, strategy)

value, trades = engine.run()

print("Final value:", value)
print("Number of trades:", len(trades))
print("First trades:", trades[:5])
