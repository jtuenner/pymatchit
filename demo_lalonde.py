# File: demo_lalonde.py

from pymatchit import MatchIt, load_lalonde
import numpy as np

# 1. Load the built-in dataset
print("Loading Lalonde dataset...")
df = load_lalonde()
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Run MatchIt
# We use 'treat' as treatment and 're78' (Real Earnings 1978) is usually the outcome (not used in matching)
# Covariates: age, educ, race (black, hispan), married, nodegree, re74, re75
m = MatchIt(df, method='nearest', distance='glm', ratio=1, caliper=0.2)

m.fit("treat ~ age + educ + black + hispan + married + nodegree + re74 + re75")

# 3. Diagnostics
print("\n--- Summary ---")
m.summary()

# 4. Plots
# Rename columns for nicer plots
pretty_names = {
    'age': 'Age',
    'educ': 'Education (Years)',
    're74': 'Income 1974',
    're75': 'Income 1975'
}

print("\nLaunching Plots...")
m.plot(type='balance', var_names=pretty_names)
m.plot(type='propensity')
m.plot(type='ecdf', variable='age')