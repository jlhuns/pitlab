# Week 01 — Core performance metrics

**Package:** `honeststats` · **Module:** `honeststats.metrics` · **Difficulty:** ●○○○○
**Started:** Wed Nov 18 (mid-week). Fri Nov 20 is a progress check; the full review is Fri Nov 27.

## Why this matters

Every later feature in `honeststats` (bootstrap CIs, walk-forward, deflated Sharpe, PBO)
consumes these numbers, so a silent bug here poisons everything downstream. Most
libraries return a plausible-looking number for degenerate input (constant returns,
no losing trades, a -120% "return"). pitlab's promise is a stack that can't lie, so the
rule here is: bad input fails loudly, undefined metrics return `nan` *and* warn.

## Background

All functions take **periodic simple returns** `r_t = P_t / P_{t-1} - 1` (`0.01` = +1%)
as a 1-D array and `periods_per_year` (252 daily, 52 weekly, 12 monthly).

**Sharpe ratio.** `sharpe = mean(r - rf) / std(r - rf, ddof=1) * sqrt(periods_per_year)`.
`rf` is the *per-period* risk-free rate (not annual). `ddof=1` because we estimate the
variance from a sample; the population estimator (`ddof=0`) is biased low and flatters the
ratio, most visibly on short samples. Annualizing by `sqrt(periods)` assumes returns are
roughly i.i.d.: mean scales with `T`, std with `sqrt(T)`, so the ratio scales with
`sqrt(T)`. That assumption fails under autocorrelation (smoothed or illiquid returns
inflate the annualized figure), which is one reason Week 02 adds bootstrap confidence
intervals. Sharpe (1994) defines the ex-post ratio as average differential return over the
standard deviation of that differential.

**Sortino ratio.** `sortino = mean(r - target) / DD * sqrt(periods_per_year)` where the
downside deviation is `DD = sqrt(mean(min(r - target, 0)^2))`. The mean is over **all**
observations (not just the losing ones) and uses no `ddof` correction: it is a
second *lower partial moment* about the target, not a sample variance. Upside volatility
is not penalized. Reference: Sortino and Price (1994).

**Equity curve and drawdown.** `equity = cumprod(1 + r)`, `dd = equity / running_max - 1`.
**Deviation from a naive `cumprod`:** prepend the initial capital `1.0`, so the curve has
`n + 1` points. Otherwise a loss in the very first period is invisible (the running max
would start *after* the loss). This changes nothing else (`equity[-1]` is identical).
`Drawdown.start`, `.trough` and `.recovery` are indices into this `n + 1` point curve:
index `0` is "before the first return", index `k` is "after return `k - 1`".
`start` is the last curve index `<= trough` at the running max (the peak); `trough` is
the first index of the minimum; `recovery` is the first index after the trough with
`equity >= peak` (`None` if it never gets back). No drawdown at all returns
`Drawdown(0.0, 0, 0, 0)`.

**CAGR.** `cagr = equity[-1] ** (periods_per_year / n) - 1`. Compound, not arithmetic.

**Calmar.** `calmar = cagr / abs(max_drawdown(r).value)`. No drawdown means the ratio
is undefined: return `nan` and warn.

**Hit rate** is the fraction of observations with `r > 0` over *all* observations (a flat
period is not a win).

**Profit factor** is `sum(gains) / abs(sum(losses))`. No-loss case (you must document it in
the docstring): gains but no losses returns `math.inf` with no warning; no gains and no losses
(all zeros) returns `nan` with a `UserWarning`; losses only returns `0.0`.

References (verified via web search; page ranges vary between catalogues, DOIs are the
authoritative pointer, so check them before citing in formal work):

- Sharpe, W. F. (1994). "The Sharpe Ratio". *The Journal of Portfolio Management*, 21(1),
  49–58. DOI 10.3905/jpm.1994.409501. Reprint: <https://web.stanford.edu/~wfsharpe/art/sr/sr.htm>
- Sortino, F. A. and Price, L. N. (1994). "Performance Measurement in a Downside Risk
  Framework". *The Journal of Investing*, 3(3). DOI 10.3905/joi.3.3.59.

## API to implement

File: `packages/honeststats/src/honeststats/metrics.py` (`mypy --strict` clean).

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

ReturnsLike: TypeAlias = Sequence[float] | npt.NDArray[np.floating]
NanPolicy = Literal["raise", "omit"]

@dataclass(frozen=True)
class Drawdown:
    value: float          # <= 0, e.g. -0.28 for a 28% drawdown
    start: int            # curve index of the peak
    trough: int           # curve index of the lowest point
    recovery: int | None  # curve index where the peak was regained, else None

def sharpe(r: ReturnsLike, rf: float = 0.0, periods_per_year: int = 252,
           nan_policy: NanPolicy = "raise") -> float: ...
def sortino(r: ReturnsLike, target: float = 0.0, periods_per_year: int = 252,
            nan_policy: NanPolicy = "raise") -> float: ...
def max_drawdown(r: ReturnsLike, nan_policy: NanPolicy = "raise") -> Drawdown: ...
def cagr(r: ReturnsLike, periods_per_year: int = 252,
         nan_policy: NanPolicy = "raise") -> float: ...
def calmar(r: ReturnsLike, periods_per_year: int = 252,
           nan_policy: NanPolicy = "raise") -> float: ...
def hit_rate(r: ReturnsLike, nan_policy: NanPolicy = "raise") -> float: ...
def profit_factor(r: ReturnsLike, nan_policy: NanPolicy = "raise") -> float: ...
```


Put input validation in **one** shared private helper; every public function calls it.

Edge-case rules (all enforced by the tests):

| Situation | Behaviour |
|---|---|
| empty input (after `omit`) | `ValueError` |
| `sharpe` with fewer than 2 observations | `ValueError` (`ddof=1` is undefined) |
| `NaN` and `nan_policy="raise"` | `ValueError` |
| `NaN` and `nan_policy="omit"` | drop NaNs, then compute; invalid `nan_policy` is a `ValueError` |
| `±inf` | `ValueError` (even with `omit`) |
| any value `< -1` | `ValueError` (even with `omit`); exactly `-1` is valid (total ruin) |
| 2-D (or higher) input | `ValueError` |
| list or `ndarray` input | both accepted, same result; never mutate the caller's array |
| zero volatility (`sharpe`), zero downside (`sortino`), no drawdown (`calmar`), all-zero (`profit_factor`) | return `nan` and emit `UserWarning` |

Floating-point trap: `sum([0.01] * 5) / 5 != 0.01` exactly, so the std of a constant series
is about `1e-18`, not `0.0`. A naive `std == 0` check lets `sharpe([0.01] * 5)` return
`~1e15` instead of `nan`. Detect constant input exactly (for example `np.ptp(x) == 0`).
Warnings should use `stacklevel=2` so they point at the caller.

## Acceptance tests

`packages/honeststats/tests/test_week01_metrics.py` is provided and must pass unmodified:
known values for every function (each derived by hand in a comment), the edge-case table,
and hypothesis property tests (drawdown in `[-1, 0]` and matches a brute-force scan, Sharpe
scale invariance, rf as a shift, permutation invariance, Sortino monotonicity, CAGR of
constant growth). Reference values: `sharpe([0.01, 0.02, 0.03, 0.04]) ≈ 30.7409`,
`max_drawdown([0.10, -0.10]).value == -0.10`. The `summary` tests are skipped until you
implement the stretch goal. Run:

```bash
uv run pytest packages/honeststats/tests/test_week01_metrics.py
```

It currently fails at import time because `honeststats.metrics` does not exist yet.

## Definition of done

- [ ] CI green on the PR (ruff check, ruff format, `mypy --strict`, pytest on 3.11–3.13)
- [ ] All provided tests in `test_week01_metrics.py` pass, unmodified
- [ ] At least 3 additional edge-case tests of your own (e.g. very long series, single
      observation for the non-Sharpe metrics, mixed int/float lists, `-1` mid-series)
- [ ] `mypy --strict` clean with full type hints
- [ ] Every public function has a docstring with its formula, units, and edge-case behaviour
- [ ] Clear, atomic commit messages
- [ ] Mark the PR ready for review and tick Week 01 in `ROADMAP.md` when merging

## Stretch goal

Add `summary(r, rf=0.0, target=0.0, periods_per_year=252, nan_policy="raise")` returning a
frozen dataclass `Summary` with fields `n`, `sharpe`, `sortino`, `max_drawdown: Drawdown`,
`cagr`, `calmar`, `hit_rate`, `profit_factor`. Validate and clean the input once, not seven
times. Tests for it are in the file and un-skip automatically. Also export the public API
from `honeststats/__init__.py`.

## Estimated time

6–8 hours.
