# tests/test_core.py

import pytest
import pandas as pd
import numpy as np
from pymatchit.core import MatchIt

# 1. Kleiner Datensatz zum Testen
@pytest.fixture
def synthetic_data():
    # 10 Zeilen: 5 Treated (1), 5 Control (0)
    df = pd.DataFrame({
        'treat': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'age':   [25, 30, 45, 22, 28, 50, 24, 29, 35, 40],
        'educ':  [12, 16, 12, 10, 14, 11, 15, 12, 12, 12],
        'income':[50, 60, 55, 40, 52, 58, 45, 48, 49, 51]
    })
    return df

def test_pipeline_execution(synthetic_data):
    """
    Smoke Test: Läuft der Code von Anfang bis Ende durch?
    """
    model = MatchIt(
        data=synthetic_data,
        method='nearest',
        distance='glm',
        replace=False
    )
    
    # Fit ausführen
    model.fit(formula="treat ~ age + educ")
    
    # CHECK 1: Wurden Propensity Scores berechnet?
    assert 'propensity_score' in model.data.columns
    assert model.data['propensity_score'].min() >= 0
    assert model.data['propensity_score'].max() <= 1
    
    # CHECK 2: Wurden Matches gefunden?
    assert model.matched_data is not None
    assert not model.matched_data.empty
    
    # Bei 1:1 Matching mit 5 Treated und 5 Controls sollten Matches entstehen
    assert len(model.matched_data) > 0

def test_weights_logic(synthetic_data):
    """
    Prüft, ob Gewichte korrekt vergeben werden (1 für Treated, >0 für Control).
    """
    model = MatchIt(synthetic_data, replace=False)
    model.fit("treat ~ age + income")
    
    # Check Treated Gewichte
    treated = model.matched_data[model.matched_data['treat'] == 1]
    assert np.all(treated['weights'] == 1.0)
    
    # Check Control Gewichte
    control = model.matched_data[model.matched_data['treat'] == 0]
    assert np.all(control['weights'] > 0)

def test_summary_does_not_crash(synthetic_data):
    """
    Prüft, ob die Summary-Funktion ohne Absturz läuft.
    """
    model = MatchIt(synthetic_data)
    model.fit("treat ~ age + educ")
    
    # Das stürzt ab, wenn diagnostics.py mathematische Fehler hat
    summary_df = model.summary()
    
    assert summary_df is not None
    assert 'Std. Mean Diff.' in summary_df.columns