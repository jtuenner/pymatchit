# Regression tests for the follow-up fixes to the 2026-06 review: options that
# were still silently ignored, calipers in full/genetic matching, and the
# order in which ratio > 1 matches are assigned.

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pymatchit.core import MatchIt
from pymatchit.matchers import FullMatcher

FORMULA = "treat ~ age + educ + black"


@pytest.fixture
def sim_data():
    """Simulated data: 300 units, ~40% treated, binary 'site' variable."""
    rng = np.random.RandomState(0)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "educ": rng.randint(8, 18, n),
            "black": rng.binomial(1, 0.3, n),
            "site": rng.randint(0, 2, n),
        }
    )
    ps_true = 1 / (1 + np.exp(-(-4.5 + 0.05 * df.age + 0.15 * df.educ)))
    df["treat"] = rng.binomial(1, ps_true)
    return df


# ==========================================
# Options that cannot be honoured are rejected
# ==========================================
@pytest.mark.parametrize("method", ["subclass", "cem", "exact"])
def test_exact_unsupported_methods_raise(sim_data, method):
    m = MatchIt(sim_data, method=method, exact=["site"])
    with pytest.raises(ValueError, match="exact is not supported"):
        m.fit(FORMULA)


@pytest.mark.parametrize("method", ["subclass", "cem", "cardinality", "exact"])
def test_caliper_unsupported_methods_raise(sim_data, method):
    m = MatchIt(sim_data, method=method, caliper=0.2)
    with pytest.raises(ValueError, match="caliper"):
        m.fit(FORMULA)


@pytest.mark.parametrize("method", ["optimal", "full", "cardinality", "cem"])
def test_replace_unsupported_methods_raise(sim_data, method):
    m = MatchIt(sim_data, method=method, replace=True)
    with pytest.raises(ValueError, match="replace is not supported"):
        m.fit(FORMULA)


@pytest.mark.parametrize("distance", ["cbps", "ridge"])
@pytest.mark.parametrize("link", ["probit", "linear.probit"])
def test_probit_link_requires_glm(sim_data, distance, link):
    m = MatchIt(sim_data, distance=distance, link=link, random_state=1)
    with pytest.raises(ValueError, match="only available for distance='glm'"):
        m.fit(FORMULA)


# ==========================================
# Covariate-specific calipers in full and genetic matching
# ==========================================
def test_genetic_covariate_caliper_respected(sim_data):
    m = MatchIt(
        sim_data,
        method="genetic",
        caliper={"age": 2},
        pop_size=10,
        max_generations=2,
        random_state=1,
    )
    m.fit(FORMULA)
    pairs = m.matches()
    assert len(pairs) > 0
    for row in pairs.itertuples():
        gap = abs(
            sim_data.loc[row.treated_index, "age"]
            - sim_data.loc[row.control_index, "age"]
        )
        assert gap <= 2


def test_full_covariate_caliper_respected(sim_data):
    m = MatchIt(sim_data, method="full", caliper={"age": 2})
    m.fit(FORMULA)
    md = m.matched_data
    assert len(md) > 0
    # Every matched unit has an opposite-group partner within the caliper
    for _, grp in md.groupby("subclass"):
        t_age = grp.loc[grp.treat == 1, "age"].values
        c_age = grp.loc[grp.treat == 0, "age"].values
        gaps = np.abs(t_age[:, None] - c_age[None, :])
        assert (gaps.min(axis=1) <= 2).all()
        assert (gaps.min(axis=0) <= 2).all()

    m_free = MatchIt(sim_data, method="full")
    m_free.fit(FORMULA)
    assert len(md) < len(m_free.matched_data)


def test_full_caliper_with_exact_uses_global_sd(sim_data):
    """The caliper width comes from the full sample's distance SD, not each stratum's."""
    seen = []
    original = FullMatcher._match_global

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("dist_std"))
        return original(self, *args, **kwargs)

    with mock.patch.object(FullMatcher, "_match_global", spy):
        m = MatchIt(sim_data, method="full", caliper=0.25, exact=["site"])
        m.fit(FORMULA)

    assert len(seen) == sim_data["site"].nunique()
    for dist_std in seen:
        assert dist_std == pytest.approx(m.distance_measure.std())


# ==========================================
# ratio > 1: matches are assigned in rounds
# ==========================================
def test_ratio_matches_assigned_in_rounds(sim_data):
    n_t = int((sim_data.treat == 1).sum())
    n_c = int((sim_data.treat == 0).sum())
    # Enough controls for one match each, not enough for two each
    assert n_t < n_c < 2 * n_t

    m = MatchIt(sim_data, method="nearest", ratio=2, random_state=1)
    m.fit(FORMULA)
    counts = [len(c) for c in m.matched_indices.values()]
    # Every treated unit gets a first match before anyone gets a second
    assert len(counts) == n_t
    assert sum(counts) == n_c
    assert max(counts) == 2


def test_ratio_one_unchanged_by_rounds(sim_data):
    """Round-based assignment is identical to greedy matching for ratio=1."""
    m = MatchIt(sim_data, method="nearest", random_state=1)
    m.fit(FORMULA)
    used = [c for cs in m.matched_indices.values() for c in cs]
    assert len(used) == len(set(used)) == int((sim_data.treat == 1).sum())
