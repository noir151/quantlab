import pandas as pd

class SMACross:

    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow

    def generate(self, df: pd.DataFrame):
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()

        signal = (fast_ma > slow_ma).astype(int)
        signal = signal.diff().fillna(0)

        return signal
