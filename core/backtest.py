import pandas as pd


class BacktestEngine:
    """
    Minimal backtesting engine (Phase 2):
    - Generates signals from strategy
    - Executes simple full-in/full-out trades
    - Tracks equity curve (portfolio value per bar)
    """

    def __init__(self, data: pd.DataFrame, strategy, initial_cash: float = 10000.0):
        self.data = data.reset_index(drop=True)
        self.strategy = strategy
        self.initial_cash = float(initial_cash)

        self.cash = float(initial_cash)
        self.position_qty = 0.0  # number of units held
        self.trades = []         # list of dicts

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + self.position_qty * float(price))

    def run(self):
        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        signals = self.strategy.generate(self.data)
        if len(signals) != len(self.data):
            raise ValueError("Strategy signals must match data length")

        equity = []
        for i in range(len(self.data)):
            price = float(self.data.loc[i, "close"])
            signal = float(signals.loc[i])

            # BUY: enter full position
            if signal == 1 and self.position_qty == 0:
                qty = self.cash / price if price > 0 else 0.0
                self.position_qty = qty
                self.cash = 0.0
                self.trades.append({"side": "BUY", "idx": i, "price": price, "qty": qty})

            # SELL: exit full position
            elif signal == -1 and self.position_qty > 0:
                self.cash = self.position_qty * price
                self.trades.append({"side": "SELL", "idx": i, "price": price, "qty": self.position_qty})
                self.position_qty = 0.0

            equity.append(self._portfolio_value(price))

        equity_curve = pd.Series(equity, name="equity")
        final_value = float(equity_curve.iloc[-1])
        return final_value, self.trades, equity_curve
