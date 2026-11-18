"""Week 01 acceptance tests for ``honeststats.metrics``.

Every expected number below was derived independently of any implementation with a
pure-Python reference calculation (``math`` only, no numpy); the derivation is in the
comment next to each constant. Conventions the tests pin down (see docs/challenges/week-01.md):

* Drawdown is measured on the equity curve *including* the initial capital of 1.0, so the
  curve has ``len(r) + 1`` points and ``Drawdown`` indices refer to that curve
  (index 0 = before the first return, index ``k`` = after return ``k - 1``).
* With no drawdown at all, ``max_drawdown`` returns ``Drawdown(0.0, 0, 0, 0)``.
* ``profit_factor`` returns ``inf`` when there are gains but no losses.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest
from honeststats import metrics
from honeststats.metrics import (
    Drawdown,
    cagr,
    calmar,
    hit_rate,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
)
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

# A: 4 steadily rising returns.
#   mean = 0.025; deviations = +-0.015, +-0.005; sample var = (2*0.015**2 + 2*0.005**2)/3
#   = 0.0005/3; std = 0.0129099...; sharpe = 0.025/0.0129099 * sqrt(252) = 30.74085...
A = [0.01, 0.02, 0.03, 0.04]
A_SHARPE_252 = 30.7408522978788

# B: mixed returns, used for most known-value tests.
#   equity curve (with initial 1.0): 1, 1.02, 1.0098, 1.040094, 1.01929212, 1.0294850412
B = [0.02, -0.01, 0.03, -0.02, 0.01]
#   sharpe: mean 0.006, sample std 0.0194936..., ratio 0.30779 * sqrt(252) = 4.59322...
B_SHARPE = 4.593220484431882
#   rf = 0.001: excess mean 0.005, same std -> 3.82768...
B_SHARPE_RF = 3.8276837370265673
#   sortino target 0: downside squares 0.01**2 + 0.02**2 = 0.0005; /5 = 0.0001; dd = 0.01
#   -> 0.006/0.01 * sqrt(252) = 9.52470...
B_SORTINO = 9.524704719832526
#   sortino target 0.01: excess = [0.01, -0.02, 0.02, -0.03, 0]; mean = -0.004 (negative!)
#   downside squares 0.0004 + 0.0009 = 0.0013; /5 = 0.00026; dd = 0.0161245...
B_SORTINO_T = -3.9379807398833626
#   gains 0.02 + 0.03 + 0.01 = 0.06; losses 0.01 + 0.02 = 0.03 -> 2.0; 3 of 5 positive -> 0.6
B_PF = 2.0
B_HIT = 0.6
#   peak at curve index 3 (1.040094), trough at index 4: 1.01929212 / 1.040094 - 1 = -0.02;
#   final equity 1.02948 < 1.040094 so never recovered.
B_DD = (-0.02, 3, 4, None)
#   cagr, periods_per_year=252, n=5: 1.0294850412 ** (252 / 5) - 1 = 3.32563...
B_CAGR = 3.325636719291218

# C: a drawdown that recovers.
#   curve: 1, 1.1, 0.88, 0.792, 1.0296, 1.23552, 1.173744
#   peak idx 1 (1.1); trough idx 3 (0.792 / 1.1 - 1 = -0.28); first curve value >= 1.1
#   after the trough is idx 5 (1.23552).
C = [0.10, -0.20, -0.10, 0.30, 0.20, -0.05]
C_DD = (-0.28, 1, 3, 5)
#   periods_per_year = 6 = n  -> cagr = final - 1 = 0.173744; calmar = 0.173744 / 0.28
C_CAGR_6 = 0.173744
C_CALMAR_6 = 0.6205142857142871


def as_tuple(d: Drawdown) -> tuple[float, int, int, int | None]:
    return (d.value, d.start, d.trough, d.recovery)


# --------------------------------------------------------------------------------------
# Known values
# --------------------------------------------------------------------------------------


def test_sharpe_reference_value() -> None:
    assert sharpe(A, periods_per_year=252) == pytest.approx(30.7409, abs=5e-4)
    assert sharpe(A) == pytest.approx(A_SHARPE_252, rel=1e-12)


def test_sharpe_known_values_and_rf() -> None:
    assert sharpe(B) == pytest.approx(B_SHARPE, rel=1e-12)
    assert sharpe(B, rf=0.001) == pytest.approx(B_SHARPE_RF, rel=1e-12)


def test_sharpe_annualization() -> None:
    # sharpe scales with sqrt(periods_per_year): 12 vs 252 -> factor sqrt(12 / 252)
    expected = B_SHARPE * math.sqrt(12 / 252)
    assert sharpe(B, periods_per_year=12) == pytest.approx(expected, rel=1e-12)


def test_sortino_known_values() -> None:
    assert sortino(B) == pytest.approx(B_SORTINO, rel=1e-12)
    assert sortino(B, target=0.01) == pytest.approx(B_SORTINO_T, rel=1e-12)


def test_max_drawdown_known_values() -> None:
    dd = max_drawdown([0.10, -0.10])
    assert dd.value == pytest.approx(-0.10)
    # curve 1, 1.1, 0.99: peak idx 1, trough idx 2, never back to 1.1
    assert (dd.start, dd.trough, dd.recovery) == (1, 2, None)

    b = max_drawdown(B)
    assert b.value == pytest.approx(B_DD[0], rel=1e-9)
    assert as_tuple(b)[1:] == B_DD[1:]

    c = max_drawdown(C)
    assert c.value == pytest.approx(C_DD[0], rel=1e-9)
    assert as_tuple(c)[1:] == C_DD[1:]


def test_max_drawdown_counts_a_loss_in_the_first_period() -> None:
    # Initial capital is part of the curve: 1, 0.8, 0.88 -> -20% from the very start.
    dd = max_drawdown([-0.2, 0.1])
    assert dd.value == pytest.approx(-0.2)
    assert (dd.start, dd.trough, dd.recovery) == (0, 1, None)


def test_max_drawdown_recovers_exactly_to_the_old_peak() -> None:
    # 1, 2, 1, 2: back at the old peak counts as recovery (>=), idx 3.
    dd = max_drawdown([1.0, -0.5, 1.0])
    assert dd.value == pytest.approx(-0.5)
    assert (dd.start, dd.trough, dd.recovery) == (1, 2, 3)


def test_max_drawdown_without_any_drawdown() -> None:
    assert as_tuple(max_drawdown([0.01, 0.02, 0.0])) == (0.0, 0, 0, 0)


def test_max_drawdown_total_ruin() -> None:
    # r == -1 is allowed (equity hits zero and stays there): drawdown is exactly -100%.
    dd = max_drawdown([0.1, -1.0, 0.5])
    assert dd.value == pytest.approx(-1.0)
    assert dd.recovery is None


def test_drawdown_is_frozen_dataclass() -> None:
    dd = max_drawdown(C)
    assert dataclasses.is_dataclass(dd)
    assert [f.name for f in dataclasses.fields(Drawdown)] == [
        "value",
        "start",
        "trough",
        "recovery",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        dd.value = 0.0  # type: ignore[misc]


def test_cagr_known_values() -> None:
    assert cagr(B) == pytest.approx(B_CAGR, rel=1e-9)
    # two +10% periods with 1 period per year: sqrt(1.21) - 1 = 10%; with 2 per year: 21%
    assert cagr([0.1, 0.1], periods_per_year=1) == pytest.approx(0.10, rel=1e-12)
    assert cagr([0.1, 0.1], periods_per_year=2) == pytest.approx(0.21, rel=1e-12)
    # final equity 0 -> -100%
    assert cagr([0.5, -1.0], periods_per_year=2) == pytest.approx(-1.0)


def test_calmar_known_value() -> None:
    assert calmar(C, periods_per_year=6) == pytest.approx(C_CALMAR_6, rel=1e-9)
    assert cagr(C, periods_per_year=6) == pytest.approx(C_CAGR_6, rel=1e-9)


def test_hit_rate_known_values() -> None:
    assert hit_rate(B) == pytest.approx(B_HIT)
    # zeros are not hits; denominator is every observation
    assert hit_rate([0.0, 0.0, 0.01, -0.01]) == pytest.approx(0.25)
    assert hit_rate([0.01, 0.02]) == 1.0


def test_profit_factor_known_values() -> None:
    assert profit_factor(B) == pytest.approx(B_PF)
    assert profit_factor([-0.01, -0.02]) == 0.0  # losses only
    # no losses: documented as +inf (gains / 0), and it must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert profit_factor([0.01, 0.02]) == math.inf


def test_known_value_calls_emit_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sharpe(B)
        sortino(B)
        max_drawdown(B)
        cagr(B)
        calmar(C)
        hit_rate(B)
        profit_factor(B)


def test_accepts_list_and_ndarray_identically() -> None:
    arr = np.array(B)
    assert sharpe(arr) == sharpe(B)
    assert sortino(arr) == sortino(B)
    assert as_tuple(max_drawdown(arr)) == as_tuple(max_drawdown(B))
    assert cagr(arr) == cagr(B)


def test_input_is_not_mutated() -> None:
    arr = np.array([0.01, float("nan"), 0.02, -0.01])
    before = arr.copy()
    sharpe(arr, nan_policy="omit")
    np.testing.assert_array_equal(arr, before)


# --------------------------------------------------------------------------------------
# Edge-case rules
# --------------------------------------------------------------------------------------

ALL_FUNCS = [sharpe, sortino, max_drawdown, cagr, calmar, hit_rate, profit_factor]


@pytest.mark.parametrize("fn", ALL_FUNCS)
def test_empty_input_raises(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn([])
    with pytest.raises(ValueError):
        fn(np.array([], dtype=float))


@pytest.mark.parametrize("fn", ALL_FUNCS)
def test_nan_raises_by_default(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn([0.01, float("nan"), -0.02, 0.03])


@pytest.mark.parametrize("fn", ALL_FUNCS)
def test_infinite_values_raise(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn([0.01, float("inf"), -0.02])


@pytest.mark.parametrize("fn", ALL_FUNCS)
def test_return_below_minus_one_raises(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn([0.01, -1.0001, 0.02])


@pytest.mark.parametrize("fn", ALL_FUNCS)
def test_two_dimensional_input_raises(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn(np.array([[0.01, 0.02], [0.03, -0.01]]))


@pytest.mark.parametrize("fn", ALL_FUNCS)
def test_nan_policy_omit_matches_clean_input(fn) -> None:  # type: ignore[no-untyped-def]
    clean = [0.01, -0.02, 0.03, 0.015, -0.005]
    dirty = [0.01, float("nan"), -0.02, 0.03, float("nan"), 0.015, -0.005]
    got = fn(dirty, nan_policy="omit")
    want = fn(clean)
    if isinstance(want, Drawdown):
        assert as_tuple(got) == as_tuple(want)
    else:
        assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("fn", ALL_FUNCS)
def test_nan_policy_omit_with_nothing_left_raises(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn([float("nan"), float("nan")], nan_policy="omit")


def test_nan_policy_omit_still_rejects_inf_and_below_minus_one() -> None:
    with pytest.raises(ValueError):
        sharpe([0.01, float("nan"), float("inf")], nan_policy="omit")
    with pytest.raises(ValueError):
        sharpe([0.01, float("nan"), -2.0], nan_policy="omit")


def test_invalid_nan_policy_raises() -> None:
    with pytest.raises(ValueError):
        sharpe(B, nan_policy="ignore")  # type: ignore[arg-type]


def test_sharpe_needs_two_observations() -> None:
    with pytest.raises(ValueError):
        sharpe([0.01])


@pytest.mark.parametrize("value", [0.01, 0.0, -0.02])
def test_sharpe_zero_volatility_returns_nan_and_warns(value: float) -> None:
    # Trap: sum(5 * 0.01) / 5 is not exactly 0.01 in floating point, so a naive
    # `std == 0` check misses constant series. Detect constant input robustly.
    with pytest.warns(UserWarning):
        result = sharpe([value] * 5)
    assert math.isnan(result)


def test_sharpe_zero_volatility_after_rf_shift() -> None:
    with pytest.warns(UserWarning):
        assert math.isnan(sharpe([0.01] * 4, rf=0.01))


def test_sortino_zero_downside_returns_nan_and_warns() -> None:
    with pytest.warns(UserWarning):
        assert math.isnan(sortino([0.01, 0.02, 0.0, 0.03]))
    # everything above a higher target is also "no downside"
    with pytest.warns(UserWarning):
        assert math.isnan(sortino(B, target=-0.5))


def test_calmar_without_drawdown_returns_nan_and_warns() -> None:
    with pytest.warns(UserWarning):
        assert math.isnan(calmar([0.01, 0.02, 0.03]))


def test_profit_factor_all_zero_returns_nan_and_warns() -> None:
    with pytest.warns(UserWarning):
        assert math.isnan(profit_factor([0.0, 0.0, 0.0]))


def test_minus_one_is_a_valid_return() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hit_rate([-1.0, 0.5]) == 0.5


# --------------------------------------------------------------------------------------
# Property tests
# --------------------------------------------------------------------------------------

returns = st.lists(
    st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=120,
)
PROP = settings(max_examples=150, deadline=None)


@PROP
@given(returns)
def test_prop_drawdown_bounds_and_ordering(r: list[float]) -> None:
    dd = max_drawdown(r)
    assert -1.0 <= dd.value <= 0.0
    assert 0 <= dd.start <= dd.trough <= len(r)
    if dd.value < 0:
        assert dd.start < dd.trough
        assert dd.recovery is None or dd.trough < dd.recovery <= len(r)
    else:
        assert as_tuple(dd) == (0.0, 0, 0, 0)


@PROP
@given(returns)
def test_prop_drawdown_matches_brute_force(r: list[float]) -> None:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(r))])
    worst = min(curve[j] / curve[: j + 1].max() - 1.0 for j in range(len(curve)))
    assert max_drawdown(r).value == pytest.approx(worst, abs=1e-12)


@PROP
@given(returns, st.floats(min_value=0.1, max_value=5.0))
def test_prop_sharpe_scale_invariant_with_zero_rf(r: list[float], k: float) -> None:
    arr = np.asarray(r)
    assume(np.ptp(arr) > 1e-3)
    scaled = arr * k * 0.2  # keeps |scaled| <= 0.5 so every return stays above -1
    base = arr * 0.2
    assert sharpe(scaled) == pytest.approx(sharpe(base), rel=1e-7)


@PROP
@given(returns, st.floats(min_value=0.0, max_value=0.05))
def test_prop_sharpe_rf_is_a_shift(r: list[float], c: float) -> None:
    arr = np.asarray(r) * 0.5
    assume(np.ptp(arr) > 1e-3)
    assert sharpe(arr + c, rf=c) == pytest.approx(sharpe(arr), rel=1e-7, abs=1e-9)


@PROP
@given(returns, st.randoms(use_true_random=False))
def test_prop_order_free_metrics_ignore_permutation(r: list[float], rnd) -> None:  # type: ignore[no-untyped-def]
    arr = np.asarray(r)
    assume(np.ptp(arr) > 1e-3)
    shuffled = list(r)
    rnd.shuffle(shuffled)
    assert sharpe(shuffled) == pytest.approx(sharpe(arr), rel=1e-7)
    assert hit_rate(shuffled) == hit_rate(arr)
    assert cagr(shuffled) == pytest.approx(cagr(arr), rel=1e-7, abs=1e-12)


@PROP
@given(returns, st.floats(min_value=0.1, max_value=2.0))
def test_prop_sortino_positive_scale_invariant(r: list[float], k: float) -> None:
    arr = np.asarray(r) * 0.25
    assume(np.any(arr < -1e-3))
    assert sortino(arr * k) == pytest.approx(sortino(arr), rel=1e-7)


@PROP
@given(returns, st.floats(min_value=0.001, max_value=0.2))
def test_prop_sortino_rises_when_an_upside_return_grows(r: list[float], bump: float) -> None:
    arr = np.asarray(r, dtype=float)
    assume(np.any(arr < -1e-3))  # need a non-zero downside deviation
    assume(np.any(arr > 0.0))
    i = int(np.argmax(arr))
    improved = arr.copy()
    improved[i] += bump  # still >= 0, so the downside deviation is unchanged
    assert sortino(improved) > sortino(arr)


@PROP
@given(returns)
def test_prop_sortino_is_nan_when_nothing_is_below_target(r: list[float]) -> None:
    arr = np.abs(np.asarray(r))
    with pytest.warns(UserWarning):
        assert math.isnan(sortino(arr))


@PROP
@given(
    st.floats(min_value=-0.05, max_value=0.05),
    st.integers(min_value=1, max_value=300),
    st.integers(min_value=1, max_value=365),
)
def test_prop_cagr_of_constant_growth(g: float, n: int, periods: int) -> None:
    got = cagr([g] * n, periods_per_year=periods)
    assert got == pytest.approx((1.0 + g) ** periods - 1.0, rel=1e-7, abs=1e-12)


@PROP
@given(returns)
def test_prop_hit_rate_and_profit_factor_ranges(r: list[float]) -> None:
    assert 0.0 <= hit_rate(r) <= 1.0
    arr = np.asarray(r)
    if np.any(arr != 0.0):
        assert profit_factor(r) >= 0.0


@PROP
@given(returns)
def test_prop_calmar_sign_follows_growth(r: list[float]) -> None:
    arr = np.asarray(r)
    assume(max_drawdown(arr).value < -1e-9)
    c = calmar(arr)
    growth = cagr(arr)
    assert math.copysign(1.0, c) == math.copysign(1.0, growth) or growth == 0.0


# --------------------------------------------------------------------------------------
# Stretch goal: summary() (skipped until the owner implements it)
# --------------------------------------------------------------------------------------

summary = getattr(metrics, "summary", None)
needs_summary = pytest.mark.skipif(
    summary is None, reason="stretch goal: summary() not implemented"
)


@needs_summary
def test_summary_collects_every_metric() -> None:
    assert summary is not None
    s = summary(C, periods_per_year=6)
    assert dataclasses.is_dataclass(s)
    assert s.sharpe == pytest.approx(sharpe(C, periods_per_year=6))
    assert s.sortino == pytest.approx(sortino(C, periods_per_year=6))
    assert as_tuple(s.max_drawdown) == as_tuple(max_drawdown(C))
    assert s.cagr == pytest.approx(C_CAGR_6)
    assert s.calmar == pytest.approx(C_CALMAR_6)
    assert s.hit_rate == pytest.approx(hit_rate(C))
    assert s.profit_factor == pytest.approx(profit_factor(C))
    assert s.n == len(C)


@needs_summary
def test_summary_is_frozen() -> None:
    assert summary is not None
    s = summary(B)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.sharpe = 0.0  # type: ignore[misc]


@needs_summary
def test_summary_propagates_validation_errors() -> None:
    assert summary is not None
    with pytest.raises(ValueError):
        summary([])


def test_metrics_handle_very_long_series() -> None:
    returns = [0.0001, -0.00005] * 50_000

    assert math.isfinite(sharpe(returns))
    assert math.isfinite(sortino(returns))
    assert math.isfinite(cagr(returns))
    assert math.isfinite(hit_rate(returns))
    assert math.isfinite(profit_factor(returns))
    assert -1.0 <= max_drawdown(returns).value <= 0.0


def test_non_sharpe_metrics_handle_single_observation() -> None:
    returns = [0.10]

    assert cagr(returns) == pytest.approx(1.10**252 - 1)
    assert hit_rate(returns) == pytest.approx(1.0)
    assert profit_factor(returns) == math.inf

    dd = max_drawdown(returns)
    assert as_tuple(dd) == (0.0, 0, 0, 0)

    with pytest.warns(UserWarning):
        assert math.isnan(sortino(returns))

    with pytest.warns(UserWarning):
        assert math.isnan(calmar(returns))


def test_metrics_accept_mixed_integer_and_float_returns() -> None:
    returns = [1, -0.5, 0, 0.25]

    assert hit_rate(returns) == pytest.approx(0.5)
    assert profit_factor(returns) == pytest.approx(2.5)

    dd = max_drawdown(returns)
    assert dd.value == pytest.approx(-0.5)
    assert dd.start == 1
    assert dd.trough == 2
    assert dd.recovery is None

    expected_cagr = (2.0 * 0.5 * 1.0 * 1.25) ** (252 / 4) - 1
    assert cagr(returns) == pytest.approx(expected_cagr)
