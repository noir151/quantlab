import numpy as np
import pandas as pd


def equity_to_returns(equity: pd.Series) -> pd.Series:
    """Percent returns from an equity curve."""
    equity = equity.astype(float)
    rets = equity.pct_change().fillna(0.0)
    return rets


def max_drawdown(equity: pd.Series) -> dict:
    """
    Max drawdown based on equity curve.
    Returns dict with drawdown value and start/end indexes.
    """
    equity = equity.astype(float)
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0

    min_dd = dd.min()
    end = dd.idxmin()
    start = equity.loc[:end].idxmax() if len(equity.loc[:end]) else equity.index[0]

    return {
        "max_drawdown": float(min_dd),
        "dd_start": start,
        "dd_end": end,
    }


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    - returns: periodic returns (e.g., daily)
    - risk_free_rate: annual risk free rate (e.g., 0.02 for 2%)
    """
    returns = returns.astype(float)

    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period

    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0

    return float(np.sqrt(periods_per_year) * (excess.mean() / std))


def trade_stats(trades: list) -> dict:
    """
    trades: list of dicts with keys:
      - side: "BUY" or "SELL"
      - idx: index/time
      - price: float
      - qty: float
      - value: float (portfolio value at trade time, optional)
    Assumes BUY then SELL pairs.
    """
    # Extract round-trips: BUY then next SELL
    round_trips = []
    buy = None
    for t in trades:
        if t["side"] == "BUY":
            buy = t
        elif t["side"] == "SELL" and buy is not None:
            round_trips.append((buy, t))
            buy = None

    if not round_trips:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return_per_trade": 0.0,
        }

    pnls = []
    rets = []
    for b, s in round_trips:
        # PnL in currency
        pnl = (s["price"] - b["price"]) * b["qty"]
        pnls.append(pnl)

        # Return per trade in pct based on entry value
        entry_value = b["price"] * b["qty"]
        trade_ret = (pnl / entry_value) if entry_value else 0.0
        rets.append(trade_ret)

    pnls = np.array(pnls, dtype=float)
    rets = np.array(rets, dtype=float)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = float((pnls > 0).mean())
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "trades": int(len(pnls)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return_per_trade": float(rets.mean()),
    }


def summarize_performance(equity: pd.Series, trades: list, periods_per_year: int = 252) -> dict:
    """
    Computes core performance metrics.
    """
    returns = equity_to_returns(equity)

    dd = max_drawdown(equity)
    sr = sharpe_ratio(returns, periods_per_year=periods_per_year)

    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if len(equity) else 0.0
    vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) else 0.0

    tstats = trade_stats(trades)

    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "annualized_volatility": vol,
        "sharpe": float(sr),
        "max_drawdown": dd["max_drawdown"],
        "drawdown_start": dd["dd_start"],
        "drawdown_end": dd["dd_end"],
        **tstats,
    }
