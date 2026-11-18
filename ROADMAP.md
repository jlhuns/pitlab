# pitlab roadmap

One feature per week. A challenge posts every Monday, gets reviewed Friday, and is
checked off here when its pull request merges. Every 4th week is lighter: docs,
cleanup and a release. The weekly automation reads this file to pick the next
challenge, so keep the format: `- [ ] Week NN · package · title — detail`.

## Phase 1 · `honeststats` (honest statistics)
- [ ] Week 01 · honeststats · Core performance metrics — sharpe, sortino, max_drawdown (with start/trough/recovery), cagr, calmar, hit_rate, profit_factor; strict edge-case rules; property tests
- [ ] Week 02 · honeststats · Bootstrap confidence intervals — stationary block bootstrap (Politis-Romano) for any metric; reproducible seeds; CI coverage test on synthetic data
- [ ] Week 03 · honeststats · Walk-forward harness — rolling and anchored train/test splits with purging and an embargo; a fit/score strategy protocol
- [ ] Week 04 · honeststats · Release week — docs site or rich README, examples, changelog, publish v0.1.0 to PyPI via trusted publishing
- [ ] Week 05 · honeststats · Deflated Sharpe ratio — Bailey and López de Prado; probabilistic Sharpe; correction for the number of trials tried
- [ ] Week 06 · honeststats · Probability of backtest overfitting — CSCV method; logit distribution; summary plot data
- [ ] Week 07 · honeststats · Report — one call builds a text/HTML report with metrics, CIs, deflated Sharpe and PBO
- [ ] Week 08 · honeststats · Release week — v0.2.0, benchmarks, blog-style example notebook

## Phase 2 · `pitstore` (point-in-time data)
- [ ] Week 09 · pitstore · Bitemporal data model — each record carries valid_time (what it describes) and known_time (when it became available); Parquet storage
- [ ] Week 10 · pitstore · As-of queries — `as_of(t)` returns only what was known at t; a property test proving lookahead is impossible
- [ ] Week 11 · pitstore · Revisions and corrections — restated fundamentals and corrected bars; full history kept
- [ ] Week 12 · pitstore · Release week — loaders for CSV and Yahoo-style bars, docs, v0.1.0

## Phase 3 · `replay` (event-driven market replay)
- [ ] Week 13 · replay · Event clock and strategy interface — strategies only receive the past, driven by pitstore
- [ ] Week 14 · replay · Fill model — slippage, fees, partial fills, latency
- [ ] Week 15 · replay · Portfolio and accounting — positions, cash, P&L, corporate actions
- [ ] Week 16 · replay · Release week — honeststats integration, end-to-end example, v0.1.0

## Phase 4 · `leakcheck` (LLM contamination detection)
- [ ] Week 17 · leakcheck · Knowledge probes — test whether a model "remembers" events after a given date
- [ ] Week 18 · leakcheck · Cutoff estimation — estimate a model's effective knowledge cutoff from probes
- [ ] Week 19 · leakcheck · Anonymizer — mask tickers, names and dates in prompts, with a reversible mapping
- [ ] Week 20 · leakcheck · Release week — write-up, v0.1.0

## Phase 5 · `riskguard` + `brokerbridge` (safe execution)
- [ ] Week 21 · riskguard · Policy engine — max position, max daily loss, banned symbols, rate limits
- [ ] Week 22 · riskguard · Out-of-process enforcement and kill switch
- [ ] Week 23 · brokerbridge · Unified order API with a paper-trading backend
- [ ] Week 24 · Release week — riskguard v0.1.0, brokerbridge v0.1.0

## Later phases
- `newsalpha` · LLM news scoring routed through leakcheck and pitstore
- `agentdesk` · evaluation harness for AI trading agents (replay + riskguard + honeststats)
- `tradelens` · trade journal and behavior analytics
