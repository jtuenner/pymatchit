# Regression tests for logic bugs found in the 2026-06 review.

import numpy as np
import pandas as pd
import pytest

from pymatchit.core import MatchIt


@pytest.fixture
def sim_data():
    """Simulated data: 200 units, ~60% treated, binary 'site' variable."""
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "educ": rng.randint(8, 18, n),
            "black": rng.binomial(1, 0.3, n),
            "site": rng.randint(0, 2, n),
        }
    )
    ps_true = 1 / (1 + np.exp(-(-3 + 0.05 * df.age + 0.1 * df.educ)))
    df["treat"] = rng.binomial(1, ps_true)
    return df


# ==========================================
# Crash fixes
# ==========================================
def test_optimal_with_tight_caliper_does_not_crash(sim_data):
    """Calipers used to make linear_sum_assignment raise 'cost matrix is infeasible'."""
    m = MatchIt(sim_data, method="optimal", caliper=0.001, random_state=1)
    m.fit("treat ~ age + educ + black")
    # A strict caliper should drop pairs, not crash
    m_loose = MatchIt(sim_data, method="optimal", random_state=1)
    m_loose.fit("treat ~ age + educ + black")
    assert len(m.matched_data) < len(m_loose.matched_data)


def test_optimal_caliper_respected(sim_data):
    caliper = 0.5
    m = MatchIt(sim_data, method="optimal", caliper=caliper, random_state=1)
    m.fit("treat ~ age + educ + black")
    threshold = caliper * m.distance_measure.std()
    pairs = m.matches()
    for row in pairs.itertuples():
        diff = abs(
            m.distance_measure.loc[row.treated_index]
            - m.distance_measure.loc[row.control_index]
        )
        assert diff <= threshold + 1e-12


def test_genetic_with_caliper_and_custom_index(sim_data):
    """Genetic matching used .iloc with index labels: crashed on offset indexes."""
    df = sim_data.copy()
    df.index = df.index + 1000
    m = MatchIt(
        df, method="genetic", caliper=0.25, pop_size=10, max_generations=2, random_state=1
    )
    m.fit("treat ~ age + educ + black")
    assert len(m.matched_data) > 0


# ==========================================
# Genetic matching without replacement
# ==========================================
def test_genetic_without_replacement_matches_all_possible(sim_data):
    """Used to fetch only `ratio` neighbors, leaving most treated units unmatched."""
    m = MatchIt(sim_data, method="genetic", pop_size=10, max_generations=2, random_state=1)
    m.fit("treat ~ age + educ + black")
    n_treated = sim_data.treat.sum()
    n_control = (sim_data.treat == 0).sum()
    matched_treated = (m.matched_data.treat == 1).sum()
    assert matched_treated == min(n_treated, n_control)


# ==========================================
# antiexact
# ==========================================
@pytest.mark.parametrize("method", ["nearest", "optimal"])
def test_antiexact_enforced(sim_data, method):
    m = MatchIt(sim_data, method=method, antiexact=["site"], random_state=1)
    m.fit("treat ~ age + educ + black")
    pairs = m.matches()
    assert len(pairs) > 0
    for row in pairs.itertuples():
        assert (
            sim_data.loc[row.treated_index, "site"]
            != sim_data.loc[row.control_index, "site"]
        )


def test_antiexact_with_replacement(sim_data):
    m = MatchIt(sim_data, method="nearest", replace=True, antiexact=["site"], random_state=1)
    m.fit("treat ~ age + educ + black")
    pairs = m.matches()
    assert len(pairs) > 0
    for row in pairs.itertuples():
        assert (
            sim_data.loc[row.treated_index, "site"]
            != sim_data.loc[row.control_index, "site"]
        )


def test_antiexact_combined_with_exact(sim_data):
    """antiexact must also be enforced inside exact-matching strata."""
    m = MatchIt(sim_data, method="nearest", exact=["black"], antiexact=["site"], random_state=1)
    m.fit("treat ~ age + educ")
    pairs = m.matches()
    assert len(pairs) > 0
    for row in pairs.itertuples():
        assert (
            sim_data.loc[row.treated_index, "black"]
            == sim_data.loc[row.control_index, "black"]
        )
        assert (
            sim_data.loc[row.treated_index, "site"]
            != sim_data.loc[row.control_index, "site"]
        )


def test_antiexact_unsupported_method_raises(sim_data):
    m = MatchIt(sim_data, method="cem", antiexact=["site"])
    with pytest.raises(NotImplementedError, match="antiexact"):
        m.fit("treat ~ age + educ")


# ==========================================
# mahvars
# ==========================================
def test_mahvars_runs_and_estimates_ps(sim_data):
    m = MatchIt(
        sim_data, method="nearest", mahvars=["age", "educ"], caliper=0.5, random_state=1
    )
    m.fit("treat ~ age + educ + black")
    # PS must still be estimated (used for the caliper)
    assert m.propensity_scores is not None
    assert len(m.matched_data) > 0


def test_mahvars_changes_matches(sim_data):
    """Mahalanobis-on-mahvars matching must differ from plain PS matching."""
    m_ps = MatchIt(sim_data, method="nearest", random_state=1)
    m_ps.fit("treat ~ age + educ + black")
    m_mah = MatchIt(sim_data, method="nearest", mahvars=["age"], random_state=1)
    m_mah.fit("treat ~ age + educ + black")
    pairs_ps = m_ps.matches().sort_values("treated_index").reset_index(drop=True)
    pairs_mah = m_mah.matches().sort_values("treated_index").reset_index(drop=True)
    assert not pairs_ps.equals(pairs_mah)


def test_mahvars_optimal_changes_matches(sim_data):
    m_ps = MatchIt(sim_data, method="optimal", random_state=1)
    m_ps.fit("treat ~ age + educ + black")
    m_mah = MatchIt(sim_data, method="optimal", mahvars=["age"], random_state=1)
    m_mah.fit("treat ~ age + educ + black")
    assert m_mah.propensity_scores is not None
    pairs_ps = m_ps.matches().sort_values("treated_index").reset_index(drop=True)
    pairs_mah = m_mah.matches().sort_values("treated_index").reset_index(drop=True)
    assert not pairs_ps.equals(pairs_mah)


def test_mahvars_full_changes_weights(sim_data):
    m_ps = MatchIt(sim_data, method="full", random_state=1)
    m_ps.fit("treat ~ age + educ + black")
    m_mah = MatchIt(sim_data, method="full", mahvars=["age"], random_state=1)
    m_mah.fit("treat ~ age + educ + black")
    assert m_mah.propensity_scores is not None
    assert len(m_mah.matched_data) > 0
    assert not m_ps.weights.equals(m_mah.weights)


def test_mahvars_unsupported_method_raises(sim_data):
    m = MatchIt(sim_data, method="genetic", mahvars=["age"])
    with pytest.raises(NotImplementedError, match="mahvars"):
        m.fit("treat ~ age + educ")


def test_mahvars_with_mahalanobis_distance_raises(sim_data):
    m = MatchIt(sim_data, method="nearest", distance="mahalanobis", mahvars=["age"])
    with pytest.raises(ValueError, match="mahvars"):
        m.fit("treat ~ age + educ")


# ==========================================
# ATC estimand
# ==========================================
def test_atc_nearest_matches_controls_as_focal(sim_data):
    m = MatchIt(sim_data, method="nearest", estimand="ATC", random_state=1)
    m.fit("treat ~ age + educ + black")
    # Keys of matched_indices must be control units
    for key in m.matched_indices:
        assert sim_data.loc[key, "treat"] == 0
    pairs = m.matches()
    assert {"treated_index", "control_index"} <= set(pairs.columns)
    for row in pairs.itertuples():
        assert sim_data.loc[row.control_index, "treat"] == 0
        assert sim_data.loc[row.treated_index, "treat"] == 1
    # Focal (control) units carry weight 1
    md = m.matched_data
    assert (md.loc[md.treat == 0, "weights"] == 1.0).all()


@pytest.mark.parametrize("method", ["exact", "cem", "subclass"])
def test_atc_stratification_methods_produce_weights(sim_data, method):
    """ATC used to fall through the weight branches, returning an empty matched set."""
    df = sim_data.copy()
    df["educ_hi"] = (df.educ > 12).astype(int)
    formula = "treat ~ black + educ_hi" if method in ("exact", "cem") else "treat ~ age + educ"
    m = MatchIt(df, method=method, estimand="ATC")
    m.fit(formula)
    md = m.matched_data
    assert len(md) > 0
    # Focal (control) units have weight 1, treated units fractional weights
    assert (md.loc[md.treat == 0, "weights"] == 1.0).all()
    assert (md.loc[md.treat == 1, "weights"] > 0).all()


def test_ate_with_pair_matching_raises(sim_data):
    for method in ("nearest", "optimal", "genetic"):
        m = MatchIt(sim_data, method=method, estimand="ATE")
        with pytest.raises(ValueError, match="ATE"):
            m.fit("treat ~ age + educ")


@pytest.mark.parametrize("method", ["optimal", "genetic"])
def test_atc_other_pair_matchers(sim_data, method):
    """The ATC focal-group switch must also apply to optimal and genetic matching."""
    kwargs = {"pop_size": 10, "max_generations": 2} if method == "genetic" else {}
    m = MatchIt(sim_data, method=method, estimand="ATC", random_state=1, **kwargs)
    m.fit("treat ~ age + educ + black")
    assert len(m.matched_indices) > 0
    for key, vals in m.matched_indices.items():
        assert sim_data.loc[key, "treat"] == 0
        for v in vals:
            assert sim_data.loc[v, "treat"] == 1
    md = m.matched_data
    assert (md.loc[md.treat == 0, "weights"] == 1.0).all()


# ==========================================
# User-supplied distance
# ==========================================
def test_user_supplied_distance_used_as_is(sim_data):
    rng = np.random.RandomState(3)
    scores = pd.Series(rng.uniform(0, 1, len(sim_data)), index=sim_data.index)
    m = MatchIt(sim_data, distance=scores.values, method="nearest", random_state=1)
    m.fit("treat ~ age + educ")
    np.testing.assert_allclose(m.distance_measure.values, scores.values)


def test_user_supplied_distance_nonprobability(sim_data):
    """Generic distance scores (outside [0,1]) must not be clipped/logit-transformed."""
    rng = np.random.RandomState(4)
    scores = pd.Series(rng.normal(0, 5, len(sim_data)), index=sim_data.index)
    m = MatchIt(sim_data, distance=scores.values, method="nearest", random_state=1)
    m.fit("treat ~ age + educ")
    np.testing.assert_allclose(m.distance_measure.values, scores.values)
    assert len(m.matched_data) > 0


# ==========================================
# Ratio weights with calipers
# ==========================================
def test_ratio_weights_account_for_partial_matches(sim_data):
    """Control weights are 1/k_i per match, where k_i is the treated unit's match count."""
    m = MatchIt(sim_data, method="nearest", ratio=2, caliper=0.2, random_state=1)
    m.fit("treat ~ age + educ + black")
    pairs = m.matches(format="wide")
    expected = {}
    for t_idx, c_list in m.matched_indices.items():
        k = len(c_list)
        for c in c_list:
            expected[c] = expected.get(c, 0.0) + 1.0 / k
    for c_idx, w in expected.items():
        assert m.weights.loc[c_idx] == pytest.approx(w)
    # This configuration produces treated units with only one in-caliper match;
    # their controls must carry full weight 1 (not 1/ratio)
    partial = [t for t, c in m.matched_indices.items() if len(c) == 1]
    assert len(partial) > 0
    for t in partial:
        c = m.matched_indices[t][0]
        assert m.weights.loc[c] == pytest.approx(1.0)


# ==========================================
# Caliper SD consistency with exact strata
# ==========================================
def test_caliper_with_exact_uses_global_sd(sim_data):
    caliper = 0.25
    m = MatchIt(sim_data, method="nearest", caliper=caliper, exact=["site"], random_state=1)
    m.fit("treat ~ age + educ + black")
    threshold = caliper * m.distance_measure.std()
    pairs = m.matches()
    for row in pairs.itertuples():
        diff = abs(
            m.distance_measure.loc[row.treated_index]
            - m.distance_measure.loc[row.control_index]
        )
        assert diff <= threshold + 1e-12


# ==========================================
# m_order='random' must not touch the global RNG
# ==========================================
def test_m_order_random_does_not_reseed_global_rng(sim_data):
    np.random.seed(123)
    m = MatchIt(sim_data, method="nearest", m_order="random", random_state=42)
    m.fit("treat ~ age + educ")
    draw_after_match = np.random.rand()

    np.random.seed(123)
    draw_clean = np.random.rand()
    assert draw_after_match == draw_clean


def test_m_order_random_reproducible(sim_data):
    results = []
    for _ in range(2):
        m = MatchIt(sim_data, method="nearest", m_order="random", random_state=7)
        m.fit("treat ~ age + educ")
        results.append(m.matches().sort_values("treated_index").reset_index(drop=True))
    pd.testing.assert_frame_equal(results[0], results[1])


# ==========================================
# CBPS warm start
# ==========================================
def test_cbps_warm_start_used(sim_data, monkeypatch):
    """beta_init was always discarded (length mismatch) and silently fell back
    to zeros; the optimizer must now start from logistic-regression coefficients."""
    import pymatchit.distance as dist_mod

    captured = {}
    real_minimize = dist_mod.minimize

    def spy(fun, x0, **kw):
        captured["x0"] = np.asarray(x0).copy()
        return real_minimize(fun, x0, **kw)

    monkeypatch.setattr(dist_mod, "minimize", spy)
    ps, _ = dist_mod.estimate_distance(sim_data, "treat ~ age + educ + black", method="cbps")
    assert captured["x0"].any(), "CBPS started from the all-zeros fallback"
    assert ((ps > 0) & (ps < 1)).all()


# ==========================================
# String index support in matches()
# ==========================================
def test_matches_with_string_index(sim_data):
    df = sim_data.copy()
    # object dtype: pyarrow-backed string indexes break patsy itself (upstream issue)
    df.index = pd.Index([f"unit_{i}" for i in range(len(df))], dtype=object)
    m = MatchIt(df, method="nearest", random_state=1)
    m.fit("treat ~ age + educ")
    pairs = m.matches()
    assert len(pairs) > 0
    assert pairs["treated_index"].notna().all()
    assert pairs["treated_index"].str.startswith("unit_").all()
