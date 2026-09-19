# Tests for the deferred improvements: R-consistent link semantics, MILP
# cardinality matching, structured full matching, and weighted KS statistics.

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit as logit_fn
from scipy.stats import ks_2samp

from pymatchit.core import MatchIt
from pymatchit.diagnostics import compute_ks_statistics, weighted_ks_statistic


@pytest.fixture
def sim_data():
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "educ": rng.randint(8, 18, n),
            "black": rng.binomial(1, 0.3, n),
        }
    )
    ps_true = 1 / (1 + np.exp(-(-3 + 0.05 * df.age + 0.1 * df.educ)))
    df["treat"] = rng.binomial(1, ps_true)  # majority treated
    return df


@pytest.fixture
def sim_data_many_controls(sim_data):
    rng = np.random.RandomState(1)
    df = sim_data.copy()
    df["treat"] = rng.binomial(1, 0.25, len(df))  # majority control
    return df


# ==========================================
# Link semantics (R MatchIt behavior)
# ==========================================
def test_link_logit_matches_on_probability(sim_data):
    m = MatchIt(sim_data, method="nearest", link="logit", random_state=1)
    m.fit("treat ~ age + educ")
    np.testing.assert_allclose(m.distance_measure.values, m.propensity_scores.values)


def test_link_linear_logit_matches_on_linear_predictor(sim_data):
    m = MatchIt(sim_data, method="nearest", link="linear.logit", random_state=1)
    m.fit("treat ~ age + educ")
    np.testing.assert_allclose(
        m.distance_measure.values, logit_fn(m.propensity_scores.values), atol=1e-8
    )


def test_link_probit_matches_on_probability(sim_data):
    m = MatchIt(sim_data, method="nearest", link="probit", random_state=1)
    m.fit("treat ~ age + educ")
    np.testing.assert_allclose(m.distance_measure.values, m.propensity_scores.values)
    m_lin = MatchIt(sim_data, method="nearest", link="linear.probit", random_state=1)
    m_lin.fit("treat ~ age + educ")
    # Linear predictor differs from the probability and is unbounded
    assert not np.allclose(m_lin.distance_measure.values, m_lin.propensity_scores.values)


def test_link_logit_ml_method_matches_on_probability(sim_data):
    m = MatchIt(
        sim_data,
        method="nearest",
        distance="randomforest",
        link="logit",
        distance_options={"n_estimators": 10},
        random_state=1,
    )
    m.fit("treat ~ age + educ")
    np.testing.assert_allclose(m.distance_measure.values, m.propensity_scores.values)


# ==========================================
# Cardinality matching (MILP)
# ==========================================
def test_cardinality_att_satisfies_balance(sim_data_many_controls):
    df = sim_data_many_controls
    tol = 0.05
    m = MatchIt(df, method="cardinality", std_tols=tol)
    m.fit("treat ~ age + educ + black")

    covs = ["age", "educ", "black"]
    t_mask = df.treat == 1
    c_sel = (df.treat == 0) & (m.weights > 0)
    pooled = np.sqrt((df.loc[t_mask, covs].var() + df.loc[~t_mask, covs].var()) / 2)
    smd = (df.loc[t_mask, covs].mean() - df.loc[c_sel, covs].mean()).abs() / pooled
    assert (smd <= tol + 1e-6).all()
    # All treated retained with weight 1
    assert (m.weights[t_mask] == 1.0).all()


def test_cardinality_milp_at_least_as_large_as_greedy(sim_data_many_controls):
    from pymatchit.matchers import CardinalityMatcher

    df = sim_data_many_controls
    covs = ["age", "educ", "black"]
    X_t = df.loc[df.treat == 1, covs].values
    X_c = df.loc[df.treat == 0, covs].values
    pooled = np.sqrt((X_t.var(axis=0) + X_c.var(axis=0)) / 2)
    eps = 0.05 * pooled
    target = X_t.mean(axis=0)

    milp_sel = CardinalityMatcher._milp_select(X_c, target, eps, time_limit=60.0)
    greedy_sel = CardinalityMatcher._greedy_select(X_c, target, eps)
    assert milp_sel is not None
    assert milp_sel.sum() >= greedy_sel.sum()


def test_cardinality_atc_mirrors_att(sim_data):
    m = MatchIt(sim_data, method="cardinality", std_tols=0.05, estimand="ATC")
    m.fit("treat ~ age + educ + black")
    md = m.matched_data
    # All controls retained with weight 1; a treated subset selected
    assert (md.loc[md.treat == 0, "weights"] == 1.0).all()
    assert 0 < (md.treat == 1).sum() <= sim_data.treat.sum()


def test_cardinality_ate_balances_both_groups(sim_data):
    tol = 0.1
    m = MatchIt(sim_data, method="cardinality", std_tols=tol, estimand="ATE")
    m.fit("treat ~ age + educ + black")

    df = sim_data
    covs = ["age", "educ", "black"]
    sel_t = (df.treat == 1) & (m.weights > 0)
    sel_c = (df.treat == 0) & (m.weights > 0)
    assert sel_t.sum() > 0 and sel_c.sum() > 0
    pooled = np.sqrt(
        (df.loc[df.treat == 1, covs].var() + df.loc[df.treat == 0, covs].var()) / 2
    )
    smd = (df.loc[sel_t, covs].mean() - df.loc[sel_c, covs].mean()).abs() / pooled
    assert (smd <= tol + 1e-6).all()


# ==========================================
# Full matching
# ==========================================
def test_full_matching_caliper_respected(sim_data):
    caliper = 0.1
    m = MatchIt(sim_data, method="full", caliper=caliper)
    m.fit("treat ~ age + educ + black")
    threshold = caliper * m.distance_measure.std()
    md = m.matched_data
    for _, grp in md.groupby("subclass"):
        d = m.distance_measure.loc[grp.index]
        t_d = d[grp.treat == 1]
        c_d = d[grp.treat == 0]
        for tv in t_d:
            for cv in c_d:
                assert abs(tv - cv) <= threshold + 1e-9


@pytest.mark.parametrize("fixture", ["sim_data", "sim_data_many_controls"])
def test_full_matching_max_controls(fixture, request):
    df = request.getfixturevalue(fixture)
    m = MatchIt(df, method="full", max_controls_per_subclass=3)
    m.fit("treat ~ age + educ + black")
    md = m.matched_data
    sizes = md[md.treat == 0].groupby("subclass").size()
    assert sizes.max() <= 3


@pytest.mark.parametrize("fixture", ["sim_data", "sim_data_many_controls"])
def test_full_matching_min_controls(fixture, request):
    df = request.getfixturevalue(fixture)
    m = MatchIt(df, method="full", min_controls_per_subclass=2)
    m.fit("treat ~ age + educ + black")
    md = m.matched_data
    sizes = md[md.treat == 0].groupby("subclass").size()
    assert sizes.min() >= 2


def test_full_matching_no_caliper_matches_everyone(sim_data):
    m = MatchIt(sim_data, method="full")
    m.fit("treat ~ age + educ + black")
    assert (m.weights > 0).all()


def test_full_matching_subclasses_have_both_groups(sim_data):
    m = MatchIt(sim_data, method="full", caliper=0.2)
    m.fit("treat ~ age + educ + black")
    md = m.matched_data
    for _, grp in md.groupby("subclass"):
        assert (grp.treat == 1).any()
        assert (grp.treat == 0).any()


# ==========================================
# Weighted KS statistics
# ==========================================
def test_weighted_ks_uniform_weights_equals_ks_2samp():
    rng = np.random.RandomState(2)
    x1 = rng.normal(0, 1, 100)
    x2 = rng.normal(0.5, 1, 80)
    wks = weighted_ks_statistic(x1, np.ones(100), x2, np.ones(80))
    assert wks == pytest.approx(ks_2samp(x1, x2).statistic)


def test_weighted_ks_responds_to_weights():
    # Control sample = treated sample plus outliers; downweighting the
    # outliers must shrink the weighted KS toward zero
    x_t = np.array([1.0, 2.0, 3.0, 4.0])
    w_t = np.ones(4)
    x_c = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 100.0])
    w_unif = np.ones(6)
    w_down = np.array([1.0, 1.0, 1.0, 1.0, 1e-9, 1e-9])
    ks_unif = weighted_ks_statistic(x_t, w_t, x_c, w_unif)
    ks_down = weighted_ks_statistic(x_t, w_t, x_c, w_down)
    assert ks_down < ks_unif
    assert ks_down == pytest.approx(0.0, abs=1e-6)


def test_compute_ks_statistics_uses_weights(sim_data):
    m = MatchIt(sim_data, method="full")
    m.fit("treat ~ age + educ + black")

    ks_weighted = compute_ks_statistics(m.data, ["age", "educ"], "treat", m.weights)
    # Compare with deliberately unweighted (all weight-1 among matched)
    flat = (m.weights > 0).astype(float)
    ks_flat = compute_ks_statistics(m.data, ["age", "educ"], "treat", flat)
    assert not ks_weighted["KS (Matched)"].equals(ks_flat["KS (Matched)"])
    assert ks_weighted["KS (Matched)"].notna().all()
    assert ks_weighted["p-value (Matched)"].notna().all()
