from core.backtest import BacktestEngine
from core.metrics import summarize_performance
from core.execution import ExecutionModel
from strategies.sma_cross import SMACross
from data.data_generator import generate_prices


def main():
    data = generate_prices(n=2000)

    strategy = SMACross(fast=10, slow=30)

    execution = ExecutionModel(
        slippage_bps=5.0,     # 5 bps = 0.05%
        fee_bps=1.0,          # 1 bps fee on notional
        fee_fixed=0.10,       # 10p per trade
        latency_bars=1,       # fill 1 bar later
    )

    engine = BacktestEngine(data, strategy, initial_cash=10000, execution=execution)
    final_value, trades, equity = engine.run()

    stats = summarize_performance(equity, trades, periods_per_year=252)

    print("\n=== QuantLab Phase 3: Execution-Aware Backtest ===")
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

    # Show execution details
    if trades:
        print("\nFirst fills:")
        for t in trades[:4]:
            print(t)


if __name__ == "__main__":
    main()