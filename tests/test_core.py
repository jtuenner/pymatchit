# tests/test_core.py

import pytest
import pandas as pd
import numpy as np
from pymatchit.core import MatchIt

# 1. Basic Synthetic Data
@pytest.fixture
def synthetic_data():
    # 10 rows: 5 Treated (1), 5 Control (0)
    # Designed to have some overlap
    df = pd.DataFrame({
        'treat': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'age':   [25, 30, 45, 22, 28, 50, 24, 29, 35, 40],
        'educ':  [12, 16, 12, 10, 14, 11, 15, 12, 12, 12],
        'income':[50, 60, 55, 40, 52, 58, 45, 48, 49, 51]
    })
    return df

# 2. Data specifically for Exact Matching (with duplicates)
@pytest.fixture
def exact_data():
    df = pd.DataFrame({
        'treat': [1, 1, 0, 0, 1, 0],
        # Perfect matches on age and educ
        'age':   [20, 20, 20, 20, 30, 40], 
        'educ':  [12, 12, 12, 12, 10, 10]
    })
    return df

def test_nearest_neighbor_default(synthetic_data):
    """
    Test standard Propensity Score Matching (ATT).
    """
    model = MatchIt(synthetic_data, method='nearest', distance='glm')
    model.fit("treat ~ age + educ")
    
    assert model.data['propensity_score'] is not None
    assert not model.matched_data.empty
    # For ATT, treated weights should be 1
    assert np.all(model.matched_data[model.matched_data['treat']==1]['weights'] == 1.0)

def test_mahalanobis_matching(synthetic_data):
    """
    Test Mahalanobis distance (should skip GLM/Propensity Score).
    """
    model = MatchIt(synthetic_data, method='nearest', distance='mahalanobis')
    model.fit("treat ~ age + educ")
    
    # Check that PS was skipped
    assert model.propensity_scores is None
    
    # Check that matching still happened
    assert not model.matched_data.empty
    assert 'weights' in model.data.columns
    assert model.data['weights'].sum() > 0

def test_exact_matching(exact_data):
    """
    Test Exact Matching on a dataset guaranteed to have matches.
    """
    model = MatchIt(exact_data, method='exact')
    model.fit("treat ~ age + educ")
    
    matched = model.matched_data
    
    # In exact_data:
    # Row 0, 1 (Treated, 20, 12) should match with Row 2, 3 (Control, 20, 12)
    # Row 4 (Treated, 30, 10) has NO match.
    
    # Valid matches should be in the output
    assert len(matched) >= 2 
    
    # Verify we didn't match the unmatched unit (Row 4)
    # (Depending on logic, it might have weight 0)
    assert model.data.loc[4, 'weights'] == 0.0

def test_subclassification(synthetic_data):
    """
    Test Subclassification (should produce weights but no 'pair' indices).
    """
    # Use fewer subclasses because dataset is tiny (N=10)
    model = MatchIt(synthetic_data, method='subclass', subclass=3)
    model.fit("treat ~ age + educ")
    
    # Subclassification weights all units (usually), none should be dropped 
    # unless bins are empty.
    assert 'weights' in model.data.columns
    
    # Check that we have weights for both groups
    w_treat = model.data.loc[model.data['treat']==1, 'weights']
    w_control = model.data.loc[model.data['treat']==0, 'weights']
    
    assert w_treat.sum() > 0
    assert w_control.sum() > 0

def test_ate_vs_att_logic(synthetic_data):
    """
    Test that ATE and ATT produce different weights in Subclassification.
    """
    # 1. Run ATT
    model_att = MatchIt(synthetic_data, method='subclass', estimand='ATT', subclass=2)
    model_att.fit("treat ~ age + educ")
    weights_att = model_att.weights.copy()
    
    # 2. Run ATE
    model_ate = MatchIt(synthetic_data, method='subclass', estimand='ATE', subclass=2)
    model_ate.fit("treat ~ age + educ")
    weights_ate = model_ate.weights.copy()
    
    # 3. Compare
    # In ATT, treated weights are always 1.0
    assert np.allclose(weights_att[synthetic_data['treat']==1], 1.0)
    
    # In ATE, treated weights vary (N_total / N_treated in bin)
    # So they should NOT all be 1.0 (unless bins are perfectly balanced 1:1, unlikely here)
    is_different = not np.allclose(weights_ate, weights_att)
    assert is_different, "ATE and ATT weights should differ"

def test_summary_output_structure(synthetic_data):
    """
    Test that summary returns the expected dictionary and DataFrame structure.
    """
    model = MatchIt(synthetic_data)
    model.fit("treat ~ age")
    summary = model.summary(print_output=False)
    
    assert isinstance(summary, dict)
    assert 'matched' in summary
    assert 'unmatched' in summary
    assert 'Std. Mean Diff.' in summary['matched'].columns