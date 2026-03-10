from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Fill:
    side: str                 # "BUY" or "SELL"
    qty: float
    requested_price: float    # price when signal fired
    fill_price: float         # price after slippage
    fee: float
    slippage_cost: float
    idx_signal: int
    idx_fill: int
    meta: Dict[str, Any]


class ExecutionModel:
    """
    Simple execution simulator:
    - slippage in basis points (bps)
    - fees: per-trade flat fee + bps on notional
    - latency in bars (fill occurs N bars later)
    """

    def __init__(
        self,
        slippage_bps: float = 2.0,
        fee_bps: float = 0.0,
        fee_fixed: float = 0.0,
        latency_bars: int = 0,
    ):
        self.slippage_bps = float(slippage_bps)
        self.fee_bps = float(fee_bps)
        self.fee_fixed = float(fee_fixed)
        self.latency_bars = int(latency_bars)

    def _apply_slippage(self, side: str, price: float) -> float:
        """
        BUY pays up (worse price), SELL receives down (worse price).
        slippage_bps = 10 means 0.10% price impact
        """
        impact = self.slippage_bps / 10000.0
        if side == "BUY":
            return price * (1.0 + impact)
        if side == "SELL":
            return price * (1.0 - impact)
        raise ValueError("side must be BUY or SELL")

    def _calc_fee(self, notional: float) -> float:
        return self.fee_fixed + (self.fee_bps / 10000.0) * notional

    def market_fill(
        self,
        side: str,
        qty: float,
        signal_idx: int,
        signal_price: float,
        fill_idx: int,
        fill_price_raw: float,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Fill:
        """
        Create a fill at fill_price_raw (bar close), apply slippage+fees.
        """
        fill_price = self._apply_slippage(side, float(fill_price_raw))
        notional = float(qty) * fill_price
        fee = self._calc_fee(notional)

        # slippage cost is difference between raw price and executed price
        if side == "BUY":
            slippage_cost = (fill_price - float(fill_price_raw)) * float(qty)
        else:
            slippage_cost = (float(fill_price_raw) - fill_price) * float(qty)

        return Fill(
            side=side,
            qty=float(qty),
            requested_price=float(signal_price),
            fill_price=float(fill_price),
            fee=float(fee),
            slippage_cost=float(slippage_cost),
            idx_signal=int(signal_idx),
            idx_fill=int(fill_idx),
            meta=extra_meta or {},
        )