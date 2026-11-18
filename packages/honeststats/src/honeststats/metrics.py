
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import warnings

import numpy as np
import numpy.typing as npt


ReturnsLike = Sequence[float] | npt.NDArray[np.floating]
NanPolicy = Literal["raise", "omit"]


@dataclass(frozen=True)
class Drawdown:
    """Details of the maximum drawdown."""

    value: float
    start: int
    trough: int
    recovery: int | None


@dataclass(frozen=True)
class Summary:
    """Summary of common return metrics."""

    n: int
    sharpe: float
    sortino: float
    max_drawdown: Drawdown
    cagr: float
    calmar: float
    hit_rate: float
    profit_factor: float


# ---------------------------------------------------------------------
# Validation and preparation
# ---------------------------------------------------------------------

def _prepare_returns(
    r: ReturnsLike,
    nan_policy: NanPolicy = "raise",
) -> npt.NDArray[np.float64]:
    """
    Convert returns to a one-dimensional float array and validate them.

    NaNs are either rejected or removed according to nan_policy.
    Infinite values and returns below -100% are always rejected.
    """
    if nan_policy not in ("raise", "omit"):
        raise ValueError("nan_policy must be 'raise' or 'omit'")

    try:
        returns = np.asarray(r, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns must contain numeric values") from exc

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")

    if returns.size == 0:
        raise ValueError("returns cannot be empty")

    if np.isinf(returns).any():
        raise ValueError("returns cannot contain infinite values")

    nan_mask = np.isnan(returns)

    if nan_mask.any():
        if nan_policy == "raise":
            raise ValueError("returns cannot contain NaN values")
        returns = returns[~nan_mask]

    if returns.size == 0:
        raise ValueError("returns cannot be empty after omitting NaNs")

    if (returns < -1.0).any():
        raise ValueError("returns cannot be less than -1.0")

    return returns


def _validate_periods(periods_per_year: float) -> None:
    """Validate the annualization frequency."""
    if (
        not np.isscalar(periods_per_year)
        or not np.isfinite(periods_per_year)
        or periods_per_year <= 0
    ):
        raise ValueError("periods_per_year must be a positive finite number")


# ---------------------------------------------------------------------
# Shared calculation helpers
#
# These helpers receive already-validated returns. They do not clean
# the input again, which lets summary() prepare the data only once.
# ---------------------------------------------------------------------

def _annualize_ratio(
    numerator: float,
    denominator: float,
    periods_per_year: float,
) -> float:
    """Annualize a return ratio using the square-root-of-time rule."""
    if denominator == 0.0 or not np.isfinite(denominator):
        warnings.warn(
            "Cannot calculate ratio with a zero or invalid denominator",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return float(
        (numerator / denominator) * np.sqrt(periods_per_year)
    )


def _cumulative_returns(
    returns: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build an equity curve beginning at 1.0."""
    return np.concatenate(
        ([1.0], np.cumprod(1.0 + returns))
    )


def _sharpe_clean(
    returns: npt.NDArray[np.float64],
    rf: float,
    periods_per_year: float,
) -> float:
    """Calculate annualized Sharpe from validated returns."""
    if returns.size < 2:
        raise ValueError("Sharpe ratio requires at least two observations")

    excess_returns = returns - rf
    mean_excess = float(np.mean(excess_returns))
    std_excess = float(np.std(excess_returns, ddof=1))

    return _annualize_ratio(
        mean_excess,
        std_excess,
        periods_per_year,
    )


def _sortino_clean(
    returns: npt.NDArray[np.float64],
    target: float,
    periods_per_year: float,
) -> float:
    """Calculate annualized Sortino from validated returns."""
    excess_returns = returns - target
    downside_returns = np.minimum(excess_returns, 0.0)

    downside_deviation = float(
        np.sqrt(np.mean(downside_returns**2))
    )

    mean_excess = float(np.mean(excess_returns))

    return _annualize_ratio(
        mean_excess,
        downside_deviation,
        periods_per_year,
    )


def _max_drawdown_clean(
    returns: npt.NDArray[np.float64],
) -> Drawdown:
    """
    Calculate maximum drawdown from validated returns.

    Indices refer to the equity curve, including its initial value at
    index 0. The first observation's ending equity is index 1.
    """
    equity = _cumulative_returns(returns)

    running_peaks = np.maximum.accumulate(equity)
    drawdowns = equity / running_peaks - 1.0

    trough = int(np.argmin(drawdowns))
    value = float(drawdowns[trough])

    if value >= 0.0:
        return Drawdown(
            value=0.0,
            start=0,
            trough=0,
            recovery=0,
        )

    # np.argmax returns the first occurrence of the highest equity.
    start = int(np.argmax(equity[: trough + 1]))
    peak_value = float(equity[start])

    recovery_candidates = np.flatnonzero(
        equity[trough + 1 :] >= peak_value
    )

    recovery = (
        int(trough + 1 + recovery_candidates[0])
        if recovery_candidates.size
        else None
    )

    return Drawdown(
        value=value,
        start=start,
        trough=trough,
        recovery=recovery,
    )


def _cagr_clean(
    returns: npt.NDArray[np.float64],
    periods_per_year: float,
) -> float:
    """Calculate CAGR from validated returns."""
    total_growth = float(np.prod(1.0 + returns))
    years = returns.size / periods_per_year

    return float(total_growth ** (1.0 / years) - 1.0)


def _calmar_from_values(
    annualized_return: float,
    drawdown: Drawdown,
) -> float:
    """Calculate Calmar using an already-computed CAGR and drawdown."""
    if drawdown.value == 0.0:
        warnings.warn(
            "Cannot calculate Calmar ratio without a drawdown",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return float(annualized_return / abs(drawdown.value))


def _hit_rate_clean(
    returns: npt.NDArray[np.float64],
) -> float:
    """Calculate the fraction of positive returns."""
    return float(np.mean(returns > 0.0))


def _profit_factor_clean(
    returns: npt.NDArray[np.float64],
) -> float:
    """Calculate gross profits divided by gross losses."""
    gross_profit = float(np.sum(returns[returns > 0.0]))
    gross_loss = float(-np.sum(returns[returns < 0.0]))

    if gross_loss == 0.0:
        if gross_profit > 0.0:
            return float("inf")

        warnings.warn(
            "Cannot calculate profit factor without profits or losses",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return float(gross_profit / gross_loss)


# ---------------------------------------------------------------------
# Public metric functions
#
# Each public function prepares its own input. summary() bypasses
# these wrappers and calls the shared calculation helpers directly.
# ---------------------------------------------------------------------

def sharpe(
    r: ReturnsLike,
    rf: float = 0.0,
    periods_per_year: float = 252,
    nan_policy: NanPolicy = "raise",
) -> float:
    """Calculate annualized Sharpe ratio."""
    returns = _prepare_returns(r, nan_policy)
    _validate_periods(periods_per_year)

    return _sharpe_clean(returns, rf, periods_per_year)


# Backwards-compatible alias.
sharpe_ratio = sharpe


def sortino(
    r: ReturnsLike,
    target: float = 0.0,
    periods_per_year: float = 252,
    nan_policy: NanPolicy = "raise",
) -> float:
    """Calculate annualized Sortino ratio."""
    returns = _prepare_returns(r, nan_policy)
    _validate_periods(periods_per_year)

    return _sortino_clean(returns, target, periods_per_year)


def max_drawdown(
    r: ReturnsLike,
    nan_policy: NanPolicy = "raise",
) -> Drawdown:
    """Calculate maximum drawdown and its equity-curve indices."""
    returns = _prepare_returns(r, nan_policy)

    return _max_drawdown_clean(returns)


def cagr(
    r: ReturnsLike,
    periods_per_year: float = 252,
    nan_policy: NanPolicy = "raise",
) -> float:
    """Calculate compound annual growth rate."""
    returns = _prepare_returns(r, nan_policy)
    _validate_periods(periods_per_year)

    return _cagr_clean(returns, periods_per_year)


def calmar(
    r: ReturnsLike,
    periods_per_year: float = 252,
    nan_policy: NanPolicy = "raise",
) -> float:
    """Calculate Calmar ratio."""
    returns = _prepare_returns(r, nan_policy)
    _validate_periods(periods_per_year)

    annualized_return = _cagr_clean(returns, periods_per_year)
    drawdown = _max_drawdown_clean(returns)

    return _calmar_from_values(annualized_return, drawdown)


def hit_rate(
    r: ReturnsLike,
    nan_policy: NanPolicy = "raise",
) -> float:
    """Calculate the fraction of periods with positive returns."""
    returns = _prepare_returns(r, nan_policy)

    return _hit_rate_clean(returns)


def profit_factor(
    r: ReturnsLike,
    nan_policy: NanPolicy = "raise",
) -> float:
    """Calculate gross profits divided by gross losses."""
    returns = _prepare_returns(r, nan_policy)

    return _profit_factor_clean(returns)


# ---------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------

def summary(
    r: ReturnsLike,
    rf: float = 0.0,
    target: float = 0.0,
    periods_per_year: float = 252,
    nan_policy: NanPolicy = "raise",
) -> Summary:
    """
    Calculate all supported metrics from one prepared returns array.

    Input validation and NaN handling happen once. Each metric then
    operates directly on the prepared array.
    """
    returns = _prepare_returns(r, nan_policy)
    _validate_periods(periods_per_year)

    # Calculate shared metrics once.
    annualized_return = _cagr_clean(
        returns,
        periods_per_year,
    )
    drawdown = _max_drawdown_clean(returns)

    return Summary(
        n=int(returns.size),
        sharpe=_sharpe_clean(
            returns,
            rf,
            periods_per_year,
        ),
        sortino=_sortino_clean(
            returns,
            target,
            periods_per_year,
        ),
        max_drawdown=drawdown,
        cagr=annualized_return,
        calmar=_calmar_from_values(
            annualized_return,
            drawdown,
        ),
        hit_rate=_hit_rate_clean(returns),
        profit_factor=_profit_factor_clean(returns),
    )
