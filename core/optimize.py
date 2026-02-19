from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any
import pandas as pd

from core.backtest import BacktestEngine
from core.metrics import summarize_performance


@dataclass
class OptimizationResult:
    params: Dict[str, Any]
    final_equity: float
    total_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    win_rate: float
    profit_factor: float
    avg_return_per_trade: float


def grid_search(
    data: pd.DataFrame,
    strategy_factory: Callable[..., Any],
    grid: Dict[str, List[Any]],
    initial_cash: float = 10000.0,
    periods_per_year: int = 252,
    sort_by: str = "sharpe",
    descending: bool = True,
) -> Tuple[pd.DataFrame, OptimizationResult]:
    """
    Brute-force grid search optimizer.

    strategy_factory: function/class that returns a strategy object, e.g. SMACross
    grid: {"fast": [5,10], "slow":[20,30]}
    sort_by: one of DataFrame columns: sharpe, total_return, max_drawdown, etc.
    """
    keys = list(grid.keys())
    if not keys:
        raise ValueError("Grid cannot be empty")

    # built cartesian product without itertools to keep it beginner-friendly
    combos = [{}]
    for k in keys:
        new_combos = []
        for c in combos:
            for v in grid[k]:
                cc = dict(c)
                cc[k] = v
                new_combos.append(cc)
        combos = new_combos

    results: List[OptimizationResult] = []

    for params in combos:
        # Rule enforced for, if both fast/slow exist, enforce fast < slow (common SMA constraint)
        if "fast" in params and "slow" in params:
            if params["fast"] >= params["slow"]:
                continue

        strategy = strategy_factory(**params)
        engine = BacktestEngine(data, strategy, initial_cash=initial_cash)

        final_value, trades, equity = engine.run()
        stats = summarize_performance(equity, trades, periods_per_year=periods_per_year)

        results.append(
            OptimizationResult(
                params=params,
                final_equity=float(stats["final_equity"]),
                total_return=float(stats["total_return"]),
                sharpe=float(stats["sharpe"]),
                max_drawdown=float(stats["max_drawdown"]),
                trades=int(stats["trades"]),
                win_rate=float(stats["win_rate"]),
                profit_factor=float(stats["profit_factor"]) if stats["profit_factor"] != float("inf") else float("inf"),
                avg_return_per_trade=float(stats["avg_return_per_trade"]),
            )
        )

    if not results:
        raise RuntimeError("No valid parameter combinations produced results")

    # Convert to DataFrame
    rows = []
    for r in results:
        row = {
            **r.params,
            "final_equity": r.final_equity,
            "total_return": r.total_return,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "trades": r.trades,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "avg_return_per_trade": r.avg_return_per_trade,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if sort_by not in df.columns:
        raise ValueError(f"sort_by must be one of: {list(df.columns)}")

    df = df.sort_values(by=sort_by, ascending=not descending).reset_index(drop=True)

    best_row = df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in keys if k in best_row}

    best = OptimizationResult(
        params=best_params,
        final_equity=float(best_row["final_equity"]),
        total_return=float(best_row["total_return"]),
        sharpe=float(best_row["sharpe"]),
        max_drawdown=float(best_row["max_drawdown"]),
        trades=int(best_row["trades"]),
        win_rate=float(best_row["win_rate"]),
        profit_factor=float(best_row["profit_factor"]),
        avg_return_per_trade=float(best_row["avg_return_per_trade"]),
    )

    return df, best
