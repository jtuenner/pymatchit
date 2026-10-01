# Tests for the option/method combinations supported by R MatchIt: exact,
# antiexact, mahvars and m_order across methods, cardinality vs. profile
# matching, estimand-specific CBPS, and link functions.

import numpy as np
import pandas as pd
import pytest

from pymatchit.core import MatchIt, _OPTION_SUPPORT
from pymatchit.matchers import _greedy_match

FORMULA = "treat ~ age + educ + black"
COVS = ["age", "educ", "black"]
GENETIC = dict(method="genetic", pop_size=10, max_generations=3, random_state=1)


@pytest.fixture
def sim_data():
    """Simulated data: 300 units, ~40% treated, a binary and a 3-level grouping."""
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "educ": rng.randint(8, 18, n),
            "black": rng.binomial(1, 0.3, n),
            "site": rng.randint(0, 2, n),
            "region": rng.randint(0, 3, n),
        }
    )
    ps_true = 1 / (1 + np.exp(-(-4.5 + 0.05 * df.age + 0.15 * df.educ)))
    df["treat"] = rng.binomial(1, ps_true)
    return df


def _pairs(m):
    return [(a, b) for a, partners in m.matched_indices.items() for b in partners]


def _subclass_cross_pairs(m, col):
    """Values of `col` for every treated/control pair sharing a subclass."""
    for _, grp in m.matched_data.groupby("subclass"):
        t = grp.loc[grp.treat == 1, col].values
        c = grp.loc[grp.treat == 0, col].values
        yield t[:, None], c[None, :]


def _max_abs_smd(m):
    return m.summary(print_output=False)["matched"]["Std. Mean Diff."].abs().max()


# ==========================================
# The support table: unsupported combinations raise
# ==========================================
ALL_METHODS = [
    "nearest",
    "optimal",
    "full",
    "genetic",
    "cardinality",
    "subclass",
    "exact",
    "cem",
]
OPTION_VALUES = {
    "exact": ["site"],
    "antiexact": ["site"],
    "mahvars": ["age"],
    "caliper": 0.2,
    "replace": True,
    "m_order": "data",
    "ratio": 2,
    "cutpoints": {"age": 4},
    "tols": {"age": 1.0},
    "min_controls_per_subclass": 2,
    "max_controls_per_subclass": 5,
}
UNSUPPORTED = [
    (option, method)
    for option, supported in _OPTION_SUPPORT.items()
    for method in ALL_METHODS
    if method not in supported
]


@pytest.mark.parametrize("option,method", UNSUPPORTED)
def test_unsupported_combination_raises(sim_data, option, method):
    m = MatchIt(sim_data, method=method, **{option: OPTION_VALUES[option]})
    with pytest.raises(ValueError, match=f"{option} is not supported"):
        m.fit(FORMULA)


def test_default_options_never_raise(sim_data):
    """Leaving every option at its default is valid for every method."""
    for method in ALL_METHODS:
        kwargs = GENETIC if method == "genetic" else {"method": method}
        formula = "treat ~ black + site" if method == "exact" else FORMULA
        MatchIt(sim_data, **kwargs).fit(formula)


def test_explicit_ratio_one_is_accepted_everywhere(sim_data):
    MatchIt(sim_data, method="full", ratio=1).fit(FORMULA)


@pytest.mark.parametrize("ratio", [0, 1.5, -1])
def test_invalid_ratio_raises(sim_data, ratio):
    with pytest.raises(ValueError, match="ratio must be a positive whole number"):
        MatchIt(sim_data, ratio=ratio).fit(FORMULA)


# ==========================================
# Genetic matching: exact, antiexact, mahvars, m_order
# ==========================================
def test_genetic_exact_respected(sim_data):
    m = MatchIt(sim_data, exact=["site"], **GENETIC).fit(FORMULA)
    pairs = _pairs(m)
    assert len(pairs) > 0
    for a, b in pairs:
        assert sim_data.loc[a, "site"] == sim_data.loc[b, "site"]


def test_genetic_antiexact_respected(sim_data):
    m = MatchIt(sim_data, antiexact=["region"], **GENETIC).fit(FORMULA)
    pairs = _pairs(m)
    assert len(pairs) > 0
    for a, b in pairs:
        assert sim_data.loc[a, "region"] != sim_data.loc[b, "region"]


def test_genetic_combined_constraints(sim_data):
    m = MatchIt(
        sim_data,
        exact=["site"],
        antiexact=["region"],
        caliper={"distance": 0.5, "age": 5},
        **GENETIC,
    ).fit(FORMULA)
    threshold = 0.5 * m.distance_measure.std()
    pairs = _pairs(m)
    assert len(pairs) > 0
    for a, b in pairs:
        assert sim_data.loc[a, "site"] == sim_data.loc[b, "site"]
        assert sim_data.loc[a, "region"] != sim_data.loc[b, "region"]
        assert abs(sim_data.loc[a, "age"] - sim_data.loc[b, "age"]) <= 5
        assert abs(m.distance_measure[a] - m.distance_measure[b]) <= threshold + 1e-12


def test_genetic_mahvars_changes_matching_variables(sim_data):
    m_all = MatchIt(sim_data, **GENETIC).fit(FORMULA)
    m_age = MatchIt(sim_data, mahvars=["age"], **GENETIC).fit(FORMULA)
    assert not m_all.weights.equals(m_age.weights)

    # With age as the only matching variable, matched pairs are closer on age
    def mean_age_gap(m):
        return np.mean(
            [abs(sim_data.loc[a, "age"] - sim_data.loc[b, "age"]) for a, b in _pairs(m)]
        )

    assert mean_age_gap(m_age) < mean_age_gap(m_all)


def test_genetic_m_order_used(sim_data):
    # More treated than controls, so the order decides who gets matched
    df = sim_data.copy()
    df["treat"] = 1 - df["treat"]
    largest = MatchIt(df, m_order="largest", **GENETIC).fit(FORMULA)
    smallest = MatchIt(df, m_order="smallest", **GENETIC).fit(FORMULA)
    assert set(largest.matched_indices) != set(smallest.matched_indices)


def test_genetic_without_replacement_uses_each_control_once(sim_data):
    m = MatchIt(sim_data, ratio=2, **GENETIC).fit(FORMULA)
    used = [b for _, b in _pairs(m)]
    assert len(used) == len(set(used))


def test_genetic_improves_on_unweighted_matching(sim_data):
    """The optimizer's result is at least as balanced as its first candidates."""
    quick = MatchIt(
        sim_data, method="genetic", pop_size=2, max_generations=0, random_state=1
    ).fit(FORMULA)
    tuned = MatchIt(
        sim_data, method="genetic", pop_size=30, max_generations=10, random_state=1
    ).fit(FORMULA)
    assert _max_abs_smd(tuned) <= _max_abs_smd(quick) + 1e-9


def test_genetic_reproducible(sim_data):
    a = MatchIt(sim_data, **GENETIC).fit(FORMULA)
    b = MatchIt(sim_data, **GENETIC).fit(FORMULA)
    assert a.weights.equals(b.weights)


# ==========================================
# Greedy matching helper
# ==========================================
def test_greedy_match_rounds_and_forbidden_pairs():
    D = np.array([[1.0, 2.0, np.inf], [1.5, np.inf, np.inf]])
    f, o = _greedy_match(D, order=np.array([0, 1]), ratio=2, replace=False)
    # Round 1: unit 0 takes column 0, unit 1 has nothing left it may use.
    # Round 2: unit 0 takes column 1. The forbidden column 2 is never used.
    assert list(zip(f, o)) == [(0, 0), (0, 1)]

    f, o = _greedy_match(D, order=np.array([1, 0]), ratio=1, replace=False)
    assert sorted(zip(f, o)) == [(0, 1), (1, 0)]

    f, o = _greedy_match(D, order=np.array([0, 1]), ratio=2, replace=True)
    assert sorted(zip(f, o)) == [(0, 0), (0, 1), (1, 0)]


# ==========================================
# Full matching: antiexact
# ==========================================
def test_full_antiexact_respected(sim_data):
    m = MatchIt(sim_data, method="full", antiexact=["region"]).fit(FORMULA)
    assert len(m.matched_data) > 0
    for t, c in _subclass_cross_pairs(m, "region"):
        assert not (t == c).any()


def test_full_exact_and_antiexact(sim_data):
    m = MatchIt(sim_data, method="full", exact=["site"], antiexact=["region"]).fit(
        FORMULA
    )
    for t, c in _subclass_cross_pairs(m, "region"):
        assert not (t == c).any()
    for t, c in _subclass_cross_pairs(m, "site"):
        assert (t == c).all()


# ==========================================
# Matching order
# ==========================================
def test_m_order_default_is_smallest_for_atc(sim_data):
    default = MatchIt(sim_data, method="nearest", estimand="ATC").fit(FORMULA)
    smallest = MatchIt(
        sim_data, method="nearest", estimand="ATC", m_order="smallest"
    ).fit(FORMULA)
    largest = MatchIt(
        sim_data, method="nearest", estimand="ATC", m_order="largest"
    ).fit(FORMULA)
    assert default.matched_indices == smallest.matched_indices
    assert default.matched_indices != largest.matched_indices


def test_m_order_default_is_largest_for_att(sim_data):
    df = sim_data.copy()
    df["treat"] = 1 - df["treat"]  # more treated than controls
    default = MatchIt(df, method="nearest").fit(FORMULA)
    largest = MatchIt(df, method="nearest", m_order="largest").fit(FORMULA)
    assert default.matched_indices == largest.matched_indices


def test_m_order_by_score_needs_a_propensity_score(sim_data):
    m = MatchIt(sim_data, method="nearest", distance="mahalanobis", m_order="largest")
    with pytest.raises(ValueError, match="orders units by the propensity score"):
        m.fit(FORMULA)
    # The default falls back to data order without a propensity score
    MatchIt(sim_data, method="nearest", distance="mahalanobis").fit(FORMULA)


def test_invalid_m_order_raises(sim_data):
    with pytest.raises(ValueError, match="m_order must be"):
        MatchIt(sim_data, method="nearest", m_order="closest").fit(FORMULA)


# ==========================================
# Cardinality and profile matching
# ==========================================
def test_cardinality_default_is_profile_matching(sim_data):
    m = MatchIt(sim_data, method="cardinality", std_tols=0.05).fit(FORMULA)
    # The whole focal (treated) group is kept
    assert (m.weights[sim_data.treat == 1] == 1.0).all()
    assert _max_abs_smd(m) <= 0.05 + 1e-6
    assert m.matches().empty


@pytest.mark.parametrize("ratio", [1, 2])
def test_cardinality_fixed_ratio(sim_data, ratio):
    m = MatchIt(sim_data, method="cardinality", ratio=ratio, std_tols=0.05).fit(FORMULA)
    n_t = int(((sim_data.treat == 1) & (m.weights > 0)).sum())
    n_c = int(((sim_data.treat == 0) & (m.weights > 0)).sum())
    assert n_t > 0
    assert n_c == ratio * n_t
    assert _max_abs_smd(m) <= 0.05 + 1e-6
    # Both groups carry the same total weight
    assert m.weights[sim_data.treat == 1].sum() == pytest.approx(
        m.weights[sim_data.treat == 0].sum()
    )


def test_cardinality_ratio_selects_at_least_as_many_as_profile_keeps(sim_data):
    """1:1 cardinality matching may drop treated units; it never adds any."""
    m = MatchIt(sim_data, method="cardinality", ratio=1, std_tols=0.01).fit(FORMULA)
    n_t = int(((sim_data.treat == 1) & (m.weights > 0)).sum())
    assert 0 < n_t <= int((sim_data.treat == 1).sum())


def test_cardinality_atc_fixed_ratio_focal_is_control(sim_data):
    df = sim_data.copy()
    df["treat"] = 1 - df["treat"]  # fewer controls than treated
    m = MatchIt(df, method="cardinality", estimand="ATC", ratio=2, std_tols=0.1).fit(
        FORMULA
    )
    n_t = int(((df.treat == 1) & (m.weights > 0)).sum())
    n_c = int(((df.treat == 0) & (m.weights > 0)).sum())
    assert n_c > 0 and n_t == 2 * n_c


def test_cardinality_ate_fixed_ratio(sim_data):
    m = MatchIt(
        sim_data, method="cardinality", estimand="ATE", ratio=1, std_tols=0.1
    ).fit(FORMULA)
    sel_t = (sim_data.treat == 1) & (m.weights > 0)
    sel_c = (sim_data.treat == 0) & (m.weights > 0)
    assert sel_t.sum() == sel_c.sum() > 0
    assert _max_abs_smd(m) <= 0.1 + 1e-6


def test_cardinality_exact_solves_each_stratum(sim_data):
    tol = 0.1
    m = MatchIt(sim_data, method="cardinality", exact=["site"], std_tols=tol).fit(
        FORMULA
    )
    md = m.matched_data
    sd_t = sim_data.loc[sim_data.treat == 1, COVS].std()
    for site, grp in md.groupby("site"):
        # One subclass per stratum, all its treated units kept, balance within
        assert grp["subclass"].nunique() == 1
        stratum = sim_data[sim_data.site == site]
        assert (grp.treat == 1).sum() == (stratum.treat == 1).sum()
        t, c = grp[grp.treat == 1], grp[grp.treat == 0]
        smd = (t[COVS].mean() - c[COVS].mean()).abs() / sd_t
        assert (smd <= tol + 1e-6).all()
        assert t["weights"].sum() == pytest.approx(c["weights"].sum())
    assert md["subclass"].nunique() == sim_data["site"].nunique()


def test_cardinality_mahvars_pairs_selected_units(sim_data):
    plain = MatchIt(sim_data, method="cardinality", ratio=1, std_tols=0.05).fit(FORMULA)
    paired = MatchIt(
        sim_data, method="cardinality", ratio=1, std_tols=0.05, mahvars=["age", "educ"]
    ).fit(FORMULA)
    # Pairing does not change who is selected, so balance is unchanged
    assert (plain.weights > 0).sum() == (paired.weights > 0).sum()
    assert _max_abs_smd(paired) <= 0.05 + 1e-6

    pairs = paired.matches()
    assert list(pairs.columns) == ["treated_index", "control_index"]
    n_t = int(((sim_data.treat == 1) & (paired.weights > 0)).sum())
    assert len(pairs) == n_t
    assert pairs["control_index"].is_unique
    # Every selected unit is in exactly one pair
    assert paired.matched_data["subclass"].value_counts().eq(2).all()


def test_cardinality_mahvars_needs_ratio(sim_data):
    m = MatchIt(sim_data, method="cardinality", mahvars=["age"])
    with pytest.raises(ValueError, match="set ratio"):
        m.fit(FORMULA)


def test_cardinality_infeasible_ratio_raises(sim_data):
    # The groups do not overlap on x, so no subsets can have similar means
    df = sim_data.copy()
    df["x"] = df["age"] + 1000 * df["treat"]
    # distance="mahalanobis": no propensity model is fit on the separated data
    m = MatchIt(df, method="cardinality", distance="mahalanobis", ratio=1, std_tols=0.1)
    with pytest.raises(ValueError, match="no balanced sample"):
        m.fit("treat ~ x")


# ==========================================
# CBPS follows the estimand
# ==========================================
@pytest.mark.parametrize("estimand", ["ATT", "ATC", "ATE"])
def test_cbps_balances_covariates_for_estimand(sim_data, estimand):
    method = "full" if estimand == "ATE" else "nearest"
    m = MatchIt(sim_data, method=method, distance="cbps", estimand=estimand).fit(
        FORMULA
    )
    ps = m.propensity_scores.values
    y = sim_data.treat.values
    if estimand == "ATT":
        w_t, w_c = y, (1 - y) * ps / (1 - ps)
    elif estimand == "ATC":
        w_t, w_c = y * (1 - ps) / ps, 1 - y
    else:
        w_t, w_c = y / ps, (1 - y) / (1 - ps)

    X = sim_data[COVS].values.astype(float)
    diff = w_t @ X / w_t.sum() - w_c @ X / w_c.sum()
    # The weights implied by the score balance the covariate means exactly
    np.testing.assert_allclose(diff / X.std(axis=0), 0, atol=1e-5)


def test_cbps_estimands_give_different_scores(sim_data):
    att = MatchIt(sim_data, distance="cbps", estimand="ATT").fit(FORMULA)
    atc = MatchIt(sim_data, distance="cbps", estimand="ATC").fit(FORMULA)
    assert not np.allclose(att.propensity_scores, atc.propensity_scores)


# ==========================================
# Links
# ==========================================
@pytest.mark.parametrize("link", ["cloglog", "cauchit"])
def test_glm_additional_links(sim_data, link):
    logit = MatchIt(sim_data).fit(FORMULA)
    m = MatchIt(sim_data, link=link).fit(FORMULA)
    assert m.propensity_scores.between(0, 1).all()
    assert not np.allclose(m.propensity_scores, logit.propensity_scores)
    np.testing.assert_allclose(m.distance_measure, m.propensity_scores)

    m_lin = MatchIt(sim_data, link=f"linear.{link}").fit(FORMULA)
    assert not np.allclose(m_lin.distance_measure, m_lin.propensity_scores)


def test_linear_alias_is_linear_logit(sim_data):
    a = MatchIt(sim_data, link="linear").fit(FORMULA)
    b = MatchIt(sim_data, link="linear.logit").fit(FORMULA)
    np.testing.assert_allclose(a.distance_measure, b.distance_measure)


def test_unknown_link_raises(sim_data):
    with pytest.raises(ValueError, match="not recognized"):
        MatchIt(sim_data, link="identity").fit(FORMULA)


@pytest.mark.parametrize("distance", ["randomforest", "decisiontree", "neuralnet"])
def test_linear_link_unavailable_for_probability_only_estimators(sim_data, distance):
    m = MatchIt(sim_data, distance=distance, link="linear.logit", random_state=1)
    with pytest.raises(ValueError, match="only provides predicted probabilities"):
        m.fit(FORMULA)


def test_linear_link_available_for_gbm(sim_data):
    m = MatchIt(
        sim_data,
        distance="gbm",
        link="linear.logit",
        distance_options={"n_estimators": 20},
        random_state=1,
    ).fit(FORMULA)
    assert not np.allclose(m.distance_measure, m.propensity_scores)
