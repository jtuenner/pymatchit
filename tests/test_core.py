import pytest
import pandas as pd
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt

from pymatchit.core import MatchIt
from pymatchit.datasets import load_lalonde

# ==========================================
# Fixtures (Test-Datenbank)
# ==========================================
@pytest.fixture
def lalonde_data():
    """Lädt die Standard-Lalonde-Daten und fügt den Pandas 2.0 StringDtype hinzu."""
    df = load_lalonde()
    
    # Erstelle eine kategorische String-Spalte für den Patsy-Test
    def interpolate_race(row):
        if row['black']: return 'Black'
        elif row['hispan']: return 'Hispanic'
        else: return 'Neither'
        
    df['race'] = df.apply(interpolate_race, axis=1)
    
    # Explizit in den modernen pandas StringDtype umwandeln, um den Bugfix zu testen
    df['race'] = df['race'].astype('string')
    
    # Wir droppen die alten Dummies
    df = df.drop(columns=['black', 'hispan'])
    return df

# ==========================================
# 1. Tests für Matching-Algorithmen
# ==========================================
def test_nearest_neighbor(lalonde_data):
    m = MatchIt(lalonde_data, method='nearest', distance='glm', ratio=2, replace=False)
    m.fit('treat ~ age + educ + married + race')
    
    assert m.matched_data is not None
    assert len(m.matched_data) > 0
    assert 'weights' in m.matched_data.columns
    assert 'subclass' in m.matched_data.columns
    # Bei Ratio=2 und ohne Replacement sollte jeder behandelte Eintrag bis zu 2 Kontrollen haben
    assert m.matched_data['weights'].sum() > 0

def test_optimal_matching(lalonde_data):
    m = MatchIt(lalonde_data, method='optimal', distance='glm', ratio=1)
    m.fit('treat ~ age + educ + race')
    
    assert len(m.matched_data) > 0
    assert 'subclass' in m.matched_data.columns
    # Bei 1:1 Matching sollten die Gewichte von Behandelten und gematchten Kontrollen gleich sein
    treated_weights = m.matched_data[m.matched_data['treat'] == 1]['weights'].sum()
    control_weights = m.matched_data[m.matched_data['treat'] == 0]['weights'].sum()
    assert treated_weights == control_weights

def test_exact_matching(lalonde_data):
    m = MatchIt(lalonde_data, method='exact')
    m.fit('treat ~ married + nodegree') # Nur wenige Variablen für exaktes Matching
    assert len(m.matched_data) > 0

def test_cem_matching(lalonde_data):
    m = MatchIt(lalonde_data, method='cem')
    m.fit('treat ~ age + educ + re74')
    assert len(m.matched_data) > 0

def test_subclass_matching(lalonde_data):
    m = MatchIt(lalonde_data, method='subclass', subclass=5)
    m.fit('treat ~ age + educ + race')
    assert m.matched_data is not None
    # Subclass generiert keine Paar-Indizes, aber Gewichte und Subklassen-IDs
    assert m.matched_data['subclass'].nunique() <= 5

# ==========================================
# 2. Tests für Caliper und Distanzen
# ==========================================
def test_global_caliper(lalonde_data):
    m_no_caliper = MatchIt(lalonde_data, method='nearest', distance='glm')
    m_no_caliper.fit('treat ~ age + educ + re74')
    
    m_caliper = MatchIt(lalonde_data, method='nearest', distance='glm', caliper=0.01) # Sehr streng
    m_caliper.fit('treat ~ age + educ + re74')
    
    # Mit sehr strengem Caliper sollten weniger Einheiten gematcht werden
    assert len(m_caliper.matched_data) < len(m_no_caliper.matched_data)

def test_covariate_specific_caliper(lalonde_data):
    m = MatchIt(lalonde_data, method='nearest', distance='glm', caliper={'distance': 0.2, 'age': 0.5})
    m.fit('treat ~ age + educ')
    
    # Sicherstellen, dass die Zuweisung nicht abstürzt und Daten generiert
    assert len(m.matched_data) > 0

def test_mahalanobis_distance(lalonde_data):
    m = MatchIt(lalonde_data, method='nearest', distance='mahalanobis')
    m.fit('treat ~ age + educ + married')
    
    # Mahalanobis berechnet KEINE Propensity Scores
    assert m.propensity_scores is None
    assert len(m.matched_data) > 0

# ==========================================
# 3. Tests für Fehlerabfang und Corner Cases
# ==========================================
def test_missing_data_raises_error(lalonde_data):
    df_nan = lalonde_data.copy()
    df_nan.loc[0, 'age'] = np.nan # Füge NaN ein
    
    m = MatchIt(df_nan, method='nearest')
    with pytest.raises(ValueError, match="Missing values"):
        m.fit('treat ~ age + educ')

def test_invalid_formula_raises_error(lalonde_data):
    m = MatchIt(lalonde_data, method='nearest')
    with pytest.raises(ValueError, match="Treatment variable"):
        m.fit('non_existent_var ~ age + educ')

def test_string_dtype_parsing(lalonde_data):
    # Dies testet unseren Bugfix in der __init__ für Pandas StringDtype
    m = MatchIt(lalonde_data, method='nearest')
    try:
        m.fit('treat ~ race + age') # Darf keinen TypeError werfen
    except TypeError:
        pytest.fail("TypeError raised! Der Pandas 2.0 StringDtype Fix funktioniert nicht.")

# ==========================================
# 4. Tests für Diagnostics und Plotting
# ==========================================
def test_summary_generation(lalonde_data):
    m = MatchIt(lalonde_data, method='nearest')
    m.fit('treat ~ age + educ + race')
    
    summary = m.summary(print_output=False)
    assert 'unmatched' in summary
    assert 'matched' in summary
    assert 'sample_sizes' in summary
    assert not summary['matched'].empty

def test_plots_return_axes(lalonde_data):
    # Um plt.show() im Hintergrund zu blockieren
    plt.ioff()
    
    m = MatchIt(lalonde_data, method='nearest', distance='glm')
    m.fit('treat ~ age + educ')
    
    # 1. Love Plot
    ax_bal = m.plot(type='balance', title="Custom Title")
    assert isinstance(ax_bal, matplotlib.axes.Axes)
    assert ax_bal.get_title() == "Custom Title"
    
    # 2. Propensity Plot (Gibt ein Array von Axes zurück, da es 2 Subplots sind)
    ax_prop = m.plot(type='propensity')
    assert isinstance(ax_prop, np.ndarray)
    assert len(ax_prop) == 2
    
    # 3. eCDF Plot
    ax_ecdf = m.plot(type='ecdf', variable='age')
    assert isinstance(ax_ecdf, matplotlib.axes.Axes)
    
    plt.close('all')

def test_ecdf_plot_categorical_error(lalonde_data):
    m = MatchIt(lalonde_data, method='nearest')
    m.fit('treat ~ age + race')
    
    # eCDF Plot einer kategorischen Variable muss einen TypeError werfen
    with pytest.raises(TypeError, match="mathematically defined for continuous/numeric variables only"):
        m.plot(type='ecdf', variable='race')