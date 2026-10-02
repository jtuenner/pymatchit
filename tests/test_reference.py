# Checks the matchers against independent reference implementations: a plain
# greedy matcher for nearest neighbor matching on mahvars with a caliper, and
# exhaustive search and a linear program for optimal full matching.

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import pinv
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

from pymatchit import load_lalonde
from pymatchit.core import MatchIt
from pymatchit.matchers import FullMatcher, _min_cost_edge_cover

FORMULA = "treat ~ age + sev + inc"
LALONDE = "treat ~ age + educ + black + hispan + married + nodegree + re74 + re75"


@pytest.fixture
def sim_data():
    """Simulated data: 260 units, ~30% treated, string index labels."""
    rng = np.random.RandomState(0)
    n = 260
    df = pd.DataFrame(
        {
            "age": rng.normal(45, 11, n),
            "sev": rng.normal(5, 2, n),
            "inc": rng.normal(40, 9, n),
            "site": rng.choice(["a", "b", "c"], n),
        }
    )
    lp = -1.2 + 0.04 * (df.age - 45) + 0.35 * (df.sev - 5) - 0.02 * (df.inc - 40)
    df["treat"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    df.index = [f"u{i}" for i in rng.permutation(n)]
    return df


# ==========================================
# Nearest neighbor matching on mahvars: the caliper is on the propensity score
# ==========================================
def _reference_greedy(D, allowed, order, ratio, replace):
    """Greedy matching written directly from the rules: in `order`, each
    focal unit takes its closest allowed unit that is still free, one match
    per round."""
    Dm = np.where(allowed, D, np.inf)
    pairs = {}
    if replace:
        for i in range(D.shape[0]):
            best = [j for j in np.argsort(Dm[i])[:ratio] if np.isfinite(Dm[i, j])]
            if best:
                pairs[i] = sorted(best)
        return pairs
    free = np.ones(D.shape[1], dtype=bool)
    done = np.zeros(D.shape[0], dtype=bool)
    for _ in range(ratio):
        for i in order:
            if done[i]:
                continue
            d = np.where(free, Dm[i], np.inf)
            j = int(d.argmin())
            if not np.isfinite(d[j]):
                done[i] = True
                continue
            pairs.setdefault(i, []).append(j)
            free[j] = False
    return {i: sorted(js) for i, js in pairs.items()}


@pytest.mark.parametrize(
    "options",
    [
        dict(caliper=0.25),
        dict(caliper=0.25, replace=True),
        dict(caliper=0.5, ratio=2),
        dict(caliper={"distance": 0.4, "inc": 6}),
        dict(caliper=0.3, estimand="ATC"),
        dict(caliper=0.25, m_order="data"),
    ],
)
def test_nearest_mahvars_caliper_matches_reference(sim_data, options):
    mahvars = ["age", "sev"]
    m = MatchIt(sim_data, method="nearest", mahvars=mahvars, **options).fit(FORMULA)

    focal = (sim_data.treat == (0 if options.get("estimand") == "ATC" else 1)).values
    ps = m.distance_measure.values
    X = sim_data[mahvars].values
    D = cdist(X[focal], X[~focal], "mahalanobis", VI=pinv(np.cov(X.T)))

    caliper = options["caliper"]
    if not isinstance(caliper, dict):
        caliper = {"distance": caliper}
    allowed = np.ones(D.shape, dtype=bool)
    for name, width in caliper.items():
        if name == "distance":
            v, width = ps, width * m.distance_measure.std()
        else:
            v = sim_data[name].values
        allowed &= np.abs(v[focal][:, None] - v[~focal][None, :]) <= width

    if options.get("m_order") == "data":
        order = np.arange(focal.sum())
    elif options.get("estimand") == "ATC":
        order = np.argsort(ps[focal])
    else:
        order = np.argsort(ps[focal])[::-1]
    expected = _reference_greedy(
        D, allowed, order, options.get("ratio", 1), options.get("replace", False)
    )

    f_pos = {label: i for i, label in enumerate(sim_data.index[focal])}
    o_pos = {label: i for i, label in enumerate(sim_data.index[~focal])}
    got = {f_pos[k]: sorted(o_pos[c] for c in v) for k, v in m.matched_indices.items()}
    assert got == expected
    # The caliper leaves most units matchable here; it used to be compared
    # with the Mahalanobis distance, which left almost none
    assert len(got) > 0.6 * min(focal.sum(), (~focal).sum())


def test_nearest_mahvars_caliper_with_exact(sim_data):
    caliper = 0.3
    m = MatchIt(
        sim_data,
        method="nearest",
        mahvars=["age", "sev"],
        caliper=caliper,
        exact="site",
    ).fit(FORMULA)
    pairs = m.matches()
    assert len(pairs) > 30
    ps = m.distance_measure
    gap = np.abs(
        ps.loc[pairs.treated_index].values - ps.loc[pairs.control_index].values
    )
    assert (gap <= caliper * ps.std() + 1e-12).all()
    site = sim_data["site"]
    assert (
        site.loc[pairs.treated_index].values == site.loc[pairs.control_index].values
    ).all()


def test_distance_caliper_needs_propensity_score(sim_data):
    m = MatchIt(sim_data, method="nearest", distance="mahalanobis", caliper=0.2)
    with pytest.raises(ValueError, match="needs a propensity score"):
        m.fit(FORMULA)
    # Covariate calipers do not need one
    m = MatchIt(sim_data, method="nearest", distance="mahalanobis", caliper={"age": 5})
    assert len(m.fit(FORMULA).matched_data) > 0


# ==========================================
# Optimal full matching
# ==========================================
def _random_problem(rng, n_t, n_c, p_forbidden=0.0):
    """Distances and allowed pairs; every unit keeps at least one allowed pair."""
    D = rng.uniform(0.1, 3.0, (n_t, n_c))
    F = rng.uniform(size=(n_t, n_c)) >= p_forbidden
    F[np.arange(n_t), rng.randint(n_c, size=n_t)] = True
    F[rng.randint(n_t, size=n_c), np.arange(n_c)] = True
    return D, F


def _exhaustive(D, F, min_controls=1, max_controls=None):
    """Best full matching by trying every set of pairs: every treated unit has
    between min_controls and max_controls controls (one treated unit per
    control if min_controls > 1), as many controls as possible are matched,
    and the total distance is minimal. Returns (controls matched, cost), or
    None when no set of pairs qualifies."""
    n_t, n_c = D.shape
    edges = [(t, c) for t in range(n_t) for c in range(n_c) if F[t, c]]
    upper = n_c if max_controls is None else max_controls
    best = None
    for mask in range(1, 2 ** len(edges)):
        deg_t, deg_c, cost = [0] * n_t, [0] * n_c, 0.0
        for k, (t, c) in enumerate(edges):
            if mask >> k & 1:
                deg_t[t] += 1
                deg_c[c] += 1
                cost += D[t, c]
        if not all(min_controls <= d <= upper for d in deg_t):
            continue
        if min_controls > 1 and max(deg_c) > 1:
            continue
        key = (-sum(d > 0 for d in deg_c), cost)
        if best is None or key < best:
            best = key
    return None if best is None else (-best[0], best[1])


def _lp(D, F, min_controls=1, max_controls=None):
    """The same problem as a linear program (a network flow, so its optimum
    is integral). Returns (controls matched, cost), or None if infeasible."""
    n_t, n_c = D.shape
    t_e, c_e = np.nonzero(F)
    n_e = len(t_e)
    skip_cost = (D[F].max() + 1) * (n_t + n_c + 1)
    objective = np.concatenate([D[t_e, c_e], np.full(n_c, skip_cost)])
    per_t = np.zeros((n_t, n_e + n_c))
    per_t[t_e, np.arange(n_e)] = 1
    per_c = np.zeros((n_c, n_e + n_c))
    per_c[c_e, np.arange(n_e)] = 1
    skipped = np.hstack([np.zeros((n_c, n_e)), np.eye(n_c)])

    A = [-per_t, -(per_c + skipped)]
    b = [-np.full(n_t, min_controls), -np.ones(n_c)]
    if max_controls is not None:
        A.append(per_t)
        b.append(np.full(n_t, max_controls))
    if min_controls > 1:
        A.append(per_c)
        b.append(np.ones(n_c))
    res = linprog(
        objective,
        A_ub=np.vstack(A),
        b_ub=np.concatenate(b),
        bounds=(0, 1),
        method="highs",
    )
    if res.status != 0:
        return None
    return n_c - int(round(res.x[n_e:].sum())), float(D[t_e, c_e] @ res.x[:n_e])


def _solve(D, F, min_controls=1, max_controls=None):
    """What the package finds: (controls matched, cost, subclasses) or None."""
    matcher = FullMatcher(
        min_controls_per_subclass=min_controls, max_controls_per_subclass=max_controls
    )
    edges = matcher._restricted_edges(D, F, min_controls)
    if edges is None:
        return None
    clusters = matcher._clusters_from_edges(*edges, D)
    cost = sum(D[np.ix_(cl["treated"], cl["control"])].sum() for cl in clusters)
    return sum(len(cl["control"]) for cl in clusters), cost, clusters


LIMITS = [(1, None), (1, 2), (2, None), (2, 3)]


@pytest.mark.parametrize("min_controls,max_controls", LIMITS)
def test_full_matching_is_optimal_exhaustive(min_controls, max_controls):
    rng = np.random.RandomState(1)
    n_infeasible = 0
    for trial in range(24):
        n_t, n_c = rng.randint(2, 4), rng.randint(2, 6)
        D, F = _random_problem(rng, n_t, n_c, p_forbidden=0.25 * (trial % 2))
        expected = _exhaustive(D, F, min_controls, max_controls)
        got = _solve(D, F, min_controls, max_controls)
        if expected is None:
            n_infeasible += 1
            assert got is None
            continue
        assert got is not None
        assert got[0] == expected[0]
        assert got[1] == pytest.approx(expected[1])
    assert n_infeasible < 24


@pytest.mark.parametrize("min_controls,max_controls", LIMITS)
@pytest.mark.parametrize("shape", [(40, 110), (70, 60), (25, 200)])
@pytest.mark.parametrize("p_forbidden", [0.0, 0.6])
def test_full_matching_is_optimal_lp(min_controls, max_controls, shape, p_forbidden):
    rng = np.random.RandomState(2)
    D, F = _random_problem(rng, *shape, p_forbidden=p_forbidden)
    expected = _lp(D, F, min_controls, max_controls)
    got = _solve(D, F, min_controls, max_controls)
    if expected is None:
        assert got is None
        return
    assert got is not None
    n_matched, cost, clusters = got
    assert n_matched == expected[0]
    assert cost == pytest.approx(expected[1])
    upper = np.inf if max_controls is None else max_controls
    for cl in clusters:
        # One unit with its partners, all pairs allowed, within the limits
        assert min(len(cl["treated"]), len(cl["control"])) == 1
        assert F[np.ix_(cl["treated"], cl["control"])].all()
        assert len(cl["control"]) <= upper
        assert len(cl["control"]) >= min_controls
        if min_controls > 1:
            assert len(cl["treated"]) == 1


def test_edge_cover_with_tied_distances():
    # Identical scores within groups: every distance is one of two values
    rng = np.random.RandomState(3)
    a = rng.randint(0, 2, 30).astype(float)
    b = rng.randint(0, 2, 50).astype(float)
    D = np.abs(a[:, None] - b[None, :])
    F = np.ones_like(D, dtype=bool)
    clusters = FullMatcher._clusters_from_edges(*_min_cost_edge_cover(D, F), D)
    assert sorted(t for cl in clusters for t in cl["treated"]) == list(range(30))
    assert sorted(c for cl in clusters for c in cl["control"]) == list(range(50))
    for cl in clusters:
        assert min(len(cl["treated"]), len(cl["control"])) == 1
        assert D[np.ix_(cl["treated"], cl["control"])].sum() == 0


def _subclass_sizes(m, treat="treat"):
    md = m.matched_data
    return md.groupby("subclass")[treat].agg(
        treated="sum", control=lambda t: (t == 0).sum()
    )


def test_full_matching_total_distance_is_minimal(sim_data):
    m = MatchIt(sim_data, method="full").fit(FORMULA)
    assert (m.weights > 0).all()
    sizes = _subclass_sizes(m)
    assert (sizes.min(axis=1) == 1).all()

    ps = m.distance_measure
    is_t = (sim_data.treat == 1).values
    D = np.abs(ps.values[is_t][:, None] - ps.values[~is_t][None, :])
    total = 0.0
    for _, grp in m.matched_data.groupby("subclass"):
        d = ps.loc[grp.index]
        total += np.abs(
            d[grp.treat == 1].values[:, None] - d[grp.treat == 0].values[None, :]
        ).sum()
    assert total == pytest.approx(_lp(D, np.ones_like(D, dtype=bool))[1])


def test_full_matching_balances_lalonde():
    # 185 treated and 429 control units with little overlap: pairing every
    # treated unit with a control of its own cannot balance them, optimal
    # full matching (several treated units sharing a control) can
    df = load_lalonde()
    m = MatchIt(df, method="full").fit(LALONDE)
    assert (m.weights > 0).all()
    smd = m.summary(print_output=False)["matched"]["Std. Mean Diff."].abs()
    assert smd["black"] < 0.05
    assert smd.max() < 0.2
    ps, w, t = m.propensity_scores, m.weights, df.treat == 1
    assert np.average(ps[~t], weights=w[~t]) == pytest.approx(ps[t].mean(), abs=0.005)
    sizes = _subclass_sizes(m)
    assert sizes.treated.max() > 1 and sizes.control.max() > 1


@pytest.mark.parametrize("estimand", ["ATT", "ATC", "ATE"])
def test_full_matching_weights_per_subclass(sim_data, estimand):
    m = MatchIt(sim_data, method="full", estimand=estimand).fit(FORMULA)
    for _, grp in m.matched_data.groupby("subclass"):
        n_t, n_c = (grp.treat == 1).sum(), (grp.treat == 0).sum()
        w_t = grp.weights[grp.treat == 1]
        w_c = grp.weights[grp.treat == 0]
        if estimand == "ATT":
            expected = (1.0, n_t / n_c)
        elif estimand == "ATC":
            expected = (n_c / n_t, 1.0)
        else:
            expected = ((n_t + n_c) / n_t, (n_t + n_c) / n_c)
        assert np.allclose(w_t, expected[0]) and np.allclose(w_c, expected[1])


def test_full_matching_max_controls_drops_fewest_controls():
    # 20 treated, 100 controls, at most 3 controls each: 60 can be matched
    rng = np.random.RandomState(4)
    df = pd.DataFrame({"x": rng.normal(size=120), "treat": [1] * 20 + [0] * 100})
    m = MatchIt(df, method="full", max_controls_per_subclass=3).fit("treat ~ x")
    md = m.matched_data
    assert (md.treat == 1).sum() == 20
    assert (md.treat == 0).sum() == 60
    assert (_subclass_sizes(m).control == 3).all()


def test_full_matching_min_controls_merges_when_infeasible():
    # 60 treated and 40 controls: no treated unit can have 2 controls of its own
    rng = np.random.RandomState(5)
    df = pd.DataFrame({"x": rng.normal(size=100), "treat": [1] * 60 + [0] * 40})
    m = MatchIt(df, method="full", min_controls_per_subclass=2)
    with pytest.warns(UserWarning, match="Subclasses were merged"):
        m.fit("treat ~ x")
    assert (m.weights > 0).all()
    sizes = _subclass_sizes(m)
    assert (sizes.control >= 2).all() and (sizes.treated >= 1).all()


@pytest.mark.parametrize(
    "limits",
    [
        dict(min_controls_per_subclass=0),
        dict(min_controls_per_subclass=1.5),
        dict(max_controls_per_subclass=0),
        dict(min_controls_per_subclass=3, max_controls_per_subclass=2),
    ],
)
def test_full_matching_limits_validated(sim_data, limits):
    with pytest.raises(ValueError, match="controls_per_subclass"):
        MatchIt(sim_data, method="full", **limits).fit(FORMULA)
