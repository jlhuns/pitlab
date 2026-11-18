# pitlab

**The backtesting stack that can't lie, including the parts that use AI.**

Backtests usually flatter strategies. Data leaks future information, fills come out
better than real ones, and dozens of strategy variants get tried until one looks good
by luck. LLMs make it worse: a model trained on later news already "knows" what
happened during the period you're testing. pitlab is a set of small, independent
Python packages that close each of these holes.

| Package | Purpose | Status |
|---|---|---|
| [`honeststats`](packages/honeststats) | Metrics with confidence intervals, walk-forward testing, deflated Sharpe, probability of backtest overfitting | 🚧 in progress |
| `pitstore` | Point-in-time data: every record knows when it became available | planned |
| `replay` | Event-driven market replay where strategies can only see the past | planned |
| `leakcheck` | Detects whether an LLM remembers the period you're testing on | planned |
| `riskguard` | Out-of-process risk limits and kill switch for humans and agents | planned |
| `brokerbridge` | One order API for paper trading and live brokers | planned |
| `newsalpha`, `agentdesk`, `tradelens` | LLM news scoring, agent evaluation, trade journaling | later |

See [ROADMAP.md](ROADMAP.md). Weekly challenge specs live in [docs/challenges](docs/challenges).

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## License

MIT
