# tests/test_full_suite.py
import pytest
import pandas as pd
import numpy as np
import matplotlib
from pymatchit.core import MatchIt

# Use non-interactive backend for plot tests
matplotlib.use('Agg')

# --- FIXTURES ---

@pytest.fixture
def lalonde_toy():
    """
    A small, deterministic subset of Lalonde-like data for consistent testing.
    """
    data = pd.DataFrame({
        'treat': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'age':   [25, 30, 35, 20, 40, 45, 50, 22, 28, 32],
        'educ':  [12, 16, 12, 10, 11, 15, 12, 12, 14, 13],
        'income':[50, 60, 55, 40, 58, 45, 48, 49, 51, 52],
        'married':[1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    })
    return data

@pytest.fixture
def separation_data():
    """
    Data with perfect separation to test GLM robustness or errors.
    """
    df = pd.DataFrame({
        'treat': [1, 1, 1, 0, 0, 0],
        'score': [10, 11, 12, 1, 2, 3] # Perfectly predicts treatment
    })
    return df

# --- 1. DISTANCE ESTIMATION TESTS ---

@pytest.mark.parametrize("method", [
    "glm", "randomforest", "decisiontree", "gbm", 
    "adaboost", "lasso", "ridge", "elasticnet", "neuralnet"
])
def test_propensity_estimation_methods(lalonde_toy, method):
    """
    Ensure all supported distance methods run and produce valid probabilities (0-1).
    """
    # Use a simpler formula to ensure convergence for small data
    model = MatchIt(lalonde_toy, method='nearest', distance=method, random_state=42)
    model.fit("treat ~ age + income")
    
    ps = model.data['propensity_score']
    assert ps is not None
    assert len(ps) == len(lalonde_toy)
    assert ps.min() >= 0.0 and ps.max() <= 1.0
    # Check that we actually stored the distance measure
    assert 'distance_measure' in model.data.columns

# --- 2. MATCHING LOGIC TESTS ---

def test_nn_ratio_and_replace(lalonde_toy):
    """
    Test 1:k matching and replacement logic.
    """
    # Ratio = 2, Replace = True
    model = MatchIt(lalonde_toy, method='nearest', ratio=2, replace=True)
    model.fit("treat ~ age")
    
    # We have 4 treated. Ratio 2 -> should seek 8 matches.
    # Since Replace=True, we might match the same control multiple times.
    matched = model.matched_data
    
    # Check if treated units have weight 1.0 (ATT)
    treated_w = matched.loc[matched['treat'] == 1, 'weights']
    assert np.allclose(treated_w, 1.0)
    
    # Check if we successfully generated weights for controls
    control_w = matched.loc[matched['treat'] == 0, 'weights']
    assert control_w.sum() > 0

def test_nn_caliper_filtering(lalonde_toy):
    """
    Test that a strict caliper drops distant matches.
    """
    # Create data where one treated unit is VERY far from any control
    df = lalonde_toy.copy()
    df.loc[0, 'age'] = 100 # Treated outlier
    
    # Strict caliper
    model = MatchIt(df, method='nearest', caliper=0.05, distance='glm')
    model.fit("treat ~ age")
    
    # The outlier (index 0) should NOT find a match
    # So its weight should be 0, or it shouldn't appear in matched_data if only positives are kept
    assert 0 not in model.matched_data.index or model.weights[0] == 0.0

def test_exact_matching_strata(lalonde_toy):
    """
    Test that exact matching only matches units with identical covariate values.
    """
    # We match exactly on 'married'.
    model = MatchIt(lalonde_toy, method='exact')
    model.fit("treat ~ married") # Only exact match on marriage status
    
    matched = model.matched_data
    
    # Iterate through matched pairs (if we could access pairs) or check group consistency
    # For ExactMatcher, all treated=1 and treated=0 in a stratum get weights.
    
    # Let's check a manual strata: Married = 1
    strata_1 = matched[matched['married'] == 1]
    # Ensure we have both treated and control in this strata if it exists
    if not strata_1.empty:
        assert (strata_1['treat'] == 1).any()
        assert (strata_1['treat'] == 0).any()

def test_genetic_matching_skeleton():
    """
    Placeholder if you implement Genetic Matching later.
    Currently just ensures it raises NotImplemented or handles gracefully.
    """
    with pytest.raises(Exception): # Or NotImplementedError
        model = MatchIt(pd.DataFrame(), method='genetic')
        model.fit("treat ~ age")

# --- 3. DIAGNOSTICS & PLOTTING ---

def test_summary_stats(lalonde_toy):
    model = MatchIt(lalonde_toy)
    model.fit("treat ~ age + income")
    
    # Capture stdout to verify print doesn't crash
    summary = model.summary(print_output=True)
    
    # Verify Structure
    assert 'unmatched' in summary
    assert 'matched' in summary
    assert 'sample_sizes' in summary
    
    # Verify SMD calculation
    smd = summary['matched'].loc['age', 'Std. Mean Diff.']
    assert isinstance(smd, float)
    assert not np.isnan(smd) # Should be valid number

def test_plots_smoke_test(lalonde_toy):
    """
    Ensure plotting functions run without error (using Agg backend).
    """
    model = MatchIt(lalonde_toy, method='nearest')
    model.fit("treat ~ age + income")
    
    # Love Plot
    try:
        model.plot(type='balance')
    except Exception as e:
        pytest.fail(f"Love plot failed: {e}")

    # Propensity Plot
    try:
        model.plot(type='propensity')
    except Exception as e:
        pytest.fail(f"Propensity plot failed: {e}")

    # ECDF Plot
    try:
        model.plot(type='ecdf', variable='age')
    except Exception as e:
        pytest.fail(f"ECDF plot failed: {e}")

# --- 4. INTEGRATION / EDGE CASES ---

def test_input_validation_empty_formula(lalonde_toy):
    model = MatchIt(lalonde_toy)
    with pytest.raises(ValueError):
        model.fit("treat ~ ") # Invalid formula

def test_discard_options(lalonde_toy):
    """
    Test 'discard' parameter (Common Support).
    """
    # discard='both' drops units outside the intersection of propensity scores
    model = MatchIt(lalonde_toy, discard='both')
    model.fit("treat ~ age")
    
    # Check that mask_kept was created
    assert model._mask_kept is not None
    # Ensure weights are generated only for kept units
    assert len(model.matched_data) <= len(lalonde_toy)

def test_matches_format_output(lalonde_toy):
    """
    Test the extraction of matched pairs in wide/long format.
    """
    model = MatchIt(lalonde_toy, method='nearest', ratio=1)
    model.fit("treat ~ age")
    
    # Long format
    matches_long = model.matches(format='long')
    assert 'treated_index' in matches_long.columns
    assert 'control_index' in matches_long.columns
    
    # Wide format
    matches_wide = model.matches(format='wide')
    assert 'treated_index' in matches_wide.columns
    assert 'control_1' in matches_wide.columns