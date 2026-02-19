from core.backtest import BacktestEngine
from core.metrics import summarize_performance
from strategies.sma_cross import SMACross
from data.data_generator import generate_prices


def main():
    data = generate_prices(n=1500)

    strategy = SMACross(fast=10, slow=30)
    engine = BacktestEngine(data, strategy, initial_cash=10000)

    final_value, trades, equity = engine.run()
    stats = summarize_performance(equity, trades, periods_per_year=252)

    print("\n=== QuantLab Phase 2: Backtest Summary ===")
    print(f"Final equity:        £{stats['final_equity']:.2f}")
    print(f"Total return:        {stats['total_return']*100:.2f}%")
    print(f"Annualized vol:      {stats['annualized_volatility']*100:.2f}%")
    print(f"Sharpe ratio:        {stats['sharpe']:.3f}")
    print(f"Max drawdown:        {stats['max_drawdown']*100:.2f}%")
    print(f"Trades (round trips): {stats['trades']}")
    print(f"Win rate:            {stats['win_rate']*100:.1f}%")
    pf = stats["profit_factor"]
    print(f"Profit factor:       {pf:.2f}" if pf != float("inf") else "Profit factor:       inf")
    print(f"Avg return/trade:    {stats['avg_return_per_trade']*100:.2f}%")
    print("Drawdown window:     ", stats["drawdown_start"], "→", stats["drawdown_end"])
    print("First trades:        ", trades[:4])


if __name__ == "__main__":
    main()
