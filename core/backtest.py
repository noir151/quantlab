import pandas as pd
import numpy as np


class BacktestEngine:

    def __init__(self, data: pd.DataFrame, strategy, initial_cash=10000):
        self.data = data
        self.strategy = strategy
        self.cash = initial_cash
        self.position = 0
        self.trades = []

    def run(self):
        signals = self.strategy.generate(self.data)

        for i, row in self.data.iterrows():
            price = row["close"]
            signal = signals.loc[i]

            if signal == 1 and self.position == 0:
                self.position = self.cash / price
                self.cash = 0
                self.trades.append(("BUY", i, price))

            elif signal == -1 and self.position > 0:
                self.cash = self.position * price
                self.position = 0
                self.trades.append(("SELL", i, price))

        final_value = self.cash + self.position * self.data.iloc[-1]["close"]
        return final_value, self.trades
