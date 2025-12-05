# tests/test_core.py

import pytest
import pandas as pd
import numpy as np
from pymatchit.core import MatchIt

# --- FIXTURES ---

@pytest.fixture
def synthetic_data():
    # 10 rows: 5 Treated (1), 5 Control (0)
    df = pd.DataFrame({
        'treat': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'age':   [25, 30, 45, 22, 28, 50, 24, 29, 35, 40],
        'educ':  [12, 16, 12, 10, 14, 11, 15, 12, 12, 12],
        'income':[50, 60, 55, 40, 52, 58, 45, 48, 49, 51]
    })
    return df

@pytest.fixture
def exact_data():
    # Dataset specifically for Exact Matching (duplicates)
    df = pd.DataFrame({
        'treat': [1, 1, 0, 0, 1, 0],
        'age':   [20, 20, 20, 20, 30, 40], 
        'educ':  [12, 12, 12, 12, 10, 10]
    })
    return df

@pytest.fixture
def outlier_data():
    # Dataset with one Treated unit that has NO overlap with controls
    # Control Age range: 20-30
    # Treated Age range: 25-30... AND one guy who is 80.
    df = pd.DataFrame({
        'treat': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'age':   [20, 22, 24, 26, 28, 25, 27, 29, 26, 80], # <--- 80 is the outlier
        'educ':  [12] * 10
    })
    return df

# --- EXISTING FUNCTIONALITY TESTS ---

def test_nearest_neighbor_default(synthetic_data):
    model = MatchIt(synthetic_data, method='nearest', distance='glm')
    model.fit("treat ~ age + educ")
    assert model.data['propensity_score'] is not None
    assert not model.matched_data.empty
    # ATT: treated weights = 1
    assert np.all(model.matched_data[model.matched_data['treat']==1]['weights'] == 1.0)

def test_mahalanobis_matching(synthetic_data):
    model = MatchIt(synthetic_data, method='nearest', distance='mahalanobis')
    model.fit("treat ~ age + educ")
    assert model.propensity_scores is None
    assert not model.matched_data.empty
    assert model.data['weights'].sum() > 0

def test_exact_matching(exact_data):
    model = MatchIt(exact_data, method='exact')
    model.fit("treat ~ age + educ")
    matched = model.matched_data
    # Row 4 (Treated, 30, 10) has NO match in exact_data.
    assert len(matched) >= 2 
    # Verify we didn't match the unmatched unit (Row 4)
    assert model.data.loc[4, 'weights'] == 0.0

def test_subclassification(synthetic_data):
    model = MatchIt(synthetic_data, method='subclass', subclass=3)
    model.fit("treat ~ age + educ")
    assert 'weights' in model.data.columns
    assert model.data['weights'].sum() > 0

def test_cem_matching(synthetic_data):
    """Smoke test for Coarsened Exact Matching."""
    # FIX: Use coarser bins (3 instead of default 5) so matches are found in this tiny dataset
    custom_cuts = {'age': 3, 'income': 3}
    
    model = MatchIt(synthetic_data, method='cem', cutpoints=custom_cuts)
    model.fit("treat ~ age + income")
    
    assert not model.matched_data.empty
    assert 'weights' in model.data.columns
    assert model.data['weights'].sum() > 0

def test_ate_vs_att_logic(synthetic_data):
    # Use subclass=2 to ensure common support in this tiny dataset
    model_att = MatchIt(synthetic_data, method='subclass', estimand='ATT', subclass=2)
    model_att.fit("treat ~ age + educ")
    weights_att = model_att.weights.copy()
    
    model_ate = MatchIt(synthetic_data, method='subclass', estimand='ATE', subclass=2)
    model_ate.fit("treat ~ age + educ")
    weights_ate = model_ate.weights.copy()
    
    # In ATT, treated weights are 1.0. In ATE, they vary.
    assert np.allclose(weights_att[synthetic_data['treat']==1], 1.0)
    assert not np.allclose(weights_ate, weights_att)

def test_summary_output_structure(synthetic_data):
    model = MatchIt(synthetic_data)
    model.fit("treat ~ age")
    summary = model.summary(print_output=False)
    assert isinstance(summary, dict)
    assert 'matched' in summary
    assert 'Std. Mean Diff.' in summary['matched'].columns

# --- NEW: SAFETY & VALIDATION TESTS ---

def test_input_safety_index_uniqueness(synthetic_data):
    """Should fail if index contains duplicates."""
    bad_df = synthetic_data.copy()
    # Force duplicate index
    bad_df.index = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8] 
    
    model = MatchIt(bad_df)
    with pytest.raises(ValueError, match="index must be unique"):
        model.fit("treat ~ age")

def test_input_safety_missing_treatment(synthetic_data):
    """Should fail if treatment column has NaNs."""
    bad_df = synthetic_data.copy()
    bad_df.loc[0, 'treat'] = np.nan
    
    model = MatchIt(bad_df)
    with pytest.raises(ValueError, match="missing values"):
        model.fit("treat ~ age")

def test_input_safety_missing_covariate(synthetic_data):
    """Should fail if covariate column has NaNs."""
    bad_df = synthetic_data.copy()
    bad_df.loc[0, 'age'] = np.nan
    
    model = MatchIt(bad_df)
    with pytest.raises(ValueError, match="Covariates contain missing values"):
        model.fit("treat ~ age")

def test_input_safety_non_binary_treatment(synthetic_data):
    """Should fail if treatment is not 0/1."""
    bad_df = synthetic_data.copy()
    bad_df.loc[0, 'treat'] = 2 # Invalid class
    
    model = MatchIt(bad_df)
    with pytest.raises(ValueError, match="must be binary"):
        model.fit("treat ~ age")

# --- NEW: DISCARD / COMMON SUPPORT TESTS ---

def test_discard_logic(outlier_data):
    """
    Test that 'discard' correctly drops units outside common support.
    """
    # 1. Run WITHOUT discard -> The outlier (age 80) should be kept
    model_raw = MatchIt(outlier_data, method='nearest', discard='none')
    model_raw.fit("treat ~ age")
    
    assert model_raw.data.shape[0] == 10
    
    # 2. Run WITH discard='treated' -> The outlier should be dropped BEFORE matching
    model_discard = MatchIt(outlier_data, method='nearest', discard='treated')
    model_discard.fit("treat ~ age")
    
    # Check if weights are 0 for the last unit (index 9)
    assert model_discard.weights[9] == 0.0
    assert model_discard._mask_kept[9] == False
    assert model_discard._mask_kept[0] == True # Control should be kept