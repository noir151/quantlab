from core.optimize import grid_search
from strategies.sma_cross import SMACross
from data.data_generator import generate_prices


def main():
    # Use synthetic data for now; later we'll load real OHLCV
    data = generate_prices(n=2000)

    grid = {
        "fast": [5, 8, 10, 12, 15, 20],
        "slow": [20, 30, 40, 50, 60, 80],
    }

    results, best = grid_search(
        data=data,
        strategy_factory=SMACross,
        grid=grid,
        initial_cash=10000,
        periods_per_year=252,
        sort_by="sharpe",
        descending=True,
    )

    print("\n=== QuantLab Optimizer (Grid Search) ===")
    print("Top 10 parameter sets by Sharpe:\n")
    cols = ["fast", "slow", "sharpe", "total_return", "max_drawdown", "trades", "win_rate", "final_equity"]
    print(results[cols].head(10).to_string(index=False))

    print("\nBest params:", best.params)
    print(f"Best Sharpe: {best.sharpe:.3f}")
    print(f"Total return: {best.total_return*100:.2f}%")
    print(f"Max drawdown: {best.max_drawdown*100:.2f}%")
    print(f"Final equity: £{best.final_equity:.2f}")

    # Optional: save to CSV for your repo / Power BI / analysis
    results.to_csv("optimizer_results.csv", index=False)
    print("\nSaved: optimizer_results.csv")


if __name__ == "__main__":
    main()
