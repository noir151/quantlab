import pandas as pd
from typing import Optional

from core.execution import ExecutionModel, Fill


class BacktestEngine:
    """
    Phase 3 backtest engine:
    - Strategy emits signals
    - Execution model simulates fills with slippage/fees/latency
    - Tracks equity curve
    - Records detailed trade fills
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        initial_cash: float = 10000.0,
        execution: Optional[ExecutionModel] = None,
    ):
        self.data = data.reset_index(drop=True)
        self.strategy = strategy
        self.initial_cash = float(initial_cash)

        self.exec = execution or ExecutionModel()

        self.cash = float(initial_cash)
        self.position_qty = 0.0  # units held
        self.trades = []         # list of dicts (fills)

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + self.position_qty * float(price))

    def _price_at(self, idx: int) -> float:
        return float(self.data.loc[idx, "close"])

    def _apply_fill(self, fill: Fill):
        """
        Update portfolio state based on fill.
        """
        if fill.side == "BUY":
            cost = fill.qty * fill.fill_price + fill.fee
            # In this engine we do "all-in" sizing so cost ~ cash
            self.cash -= cost
            self.position_qty += fill.qty
        elif fill.side == "SELL":
            proceeds = fill.qty * fill.fill_price - fill.fee
            self.cash += proceeds
            self.position_qty -= fill.qty
        else:
            raise ValueError("Invalid fill side")

        # store trade fill as plain dict for JSON friendliness
        self.trades.append({
            "side": fill.side,
            "idx_signal": fill.idx_signal,
            "idx_fill": fill.idx_fill,
            "requested_price": fill.requested_price,
            "fill_price": fill.fill_price,
            "qty": fill.qty,
            "fee": fill.fee,
            "slippage_cost": fill.slippage_cost,
        })

    def run(self):
        if "close" not in self.data.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        signals = self.strategy.generate(self.data)
        if len(signals) != len(self.data):
            raise ValueError("Strategy signals must match data length")

        equity = []
        pending_order = None  # {"side":..., "signal_idx":..., "signal_price":..., "fill_idx":...}

        for i in range(len(self.data)):
            price = self._price_at(i)

            # 1) Check pending order fill
            if pending_order is not None and i >= pending_order["fill_idx"]:
                fill = self.exec.market_fill(
                    side=pending_order["side"],
                    qty=pending_order["qty"],
                    signal_idx=pending_order["signal_idx"],
                    signal_price=pending_order["signal_price"],
                    fill_idx=i,
                    fill_price_raw=price,
                )
                self._apply_fill(fill)
                pending_order = None

            # 2) Generate new orders if none pending
            signal = float(signals.loc[i])

            # Full-in / full-out logic (kept simple intentionally)
            if pending_order is None:
                # Enter
                if signal == 1 and self.position_qty == 0:
                    signal_price = price
                    fill_idx = min(i + self.exec.latency_bars, len(self.data) - 1)

                    # all-in sizing: buy with available cash, keep fees in mind roughly
                    # We size conservatively by ignoring fee in sizing; cash may go slightly negative if fee exists.
                    qty = (self.cash / signal_price) if signal_price > 0 else 0.0

                    pending_order = {
                        "side": "BUY",
                        "qty": float(qty),
                        "signal_idx": i,
                        "signal_price": float(signal_price),
                        "fill_idx": int(fill_idx),
                    }

                # Exit
                elif signal == -1 and self.position_qty > 0:
                    signal_price = price
                    fill_idx = min(i + self.exec.latency_bars, len(self.data) - 1)

                    pending_order = {
                        "side": "SELL",
                        "qty": float(self.position_qty),
                        "signal_idx": i,
                        "signal_price": float(signal_price),
                        "fill_idx": int(fill_idx),
                    }

            equity.append(self._portfolio_value(price))

        equity_curve = pd.Series(equity, name="equity")
        final_value = float(equity_curve.iloc[-1])
        return final_value, self.trades, equity_curve