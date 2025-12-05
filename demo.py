import pandas as pd
import numpy as np
from src.pymatchit.core import MatchIt

# 1. Setup Random Data Generation
np.random.seed(42) # For reproducible "random" numbers

# We create two distinct populations so the matching has work to do
n_treat = 50
n_control = 200 

# TREATED GROUP: Older, higher education, more likely to be married
treat_df = pd.DataFrame({
    'treat': 1,
    'age': np.random.normal(loc=35, scale=5, size=n_treat),     # Mean Age 35
    'educ': np.random.randint(10, 18, size=n_treat),            # 10-18 years school
    'income': np.random.normal(30000, 5000, size=n_treat),      # Higher income
    'married': np.random.binomial(1, 0.7, size=n_treat),        # 70% Married
    'nodegree': np.random.binomial(1, 0.2, size=n_treat)        # 20% No Degree
})

# CONTROL GROUP: Younger, lower education, less likely to be married
control_df = pd.DataFrame({
    'treat': 0,
    'age': np.random.normal(loc=25, scale=6, size=n_control),   # Mean Age 25 (Big gap!)
    'educ': np.random.randint(8, 14, size=n_control),           # 8-14 years school
    'income': np.random.normal(22000, 4000, size=n_control),    # Lower income
    'married': np.random.binomial(1, 0.3, size=n_control),      # 30% Married
    'nodegree': np.random.binomial(1, 0.6, size=n_control)      # 60% No Degree
})

# Combine into one dataset
df = pd.concat([treat_df, control_df]).reset_index(drop=True)

print(f"Generated Dataset: {len(df)} observations")
print("---------------------------------------------------")

# 2. Run MatchIt
model = MatchIt(df, method='exact', replace=False)
model.fit("treat ~ age + educ + income + married + nodegree")


pretty_names = {
    'age': 'Age (Years)',
    'educ': 'Education (Years)',
    'income': 'Annual Income ($)',
    'married': 'Married (Binary)',
    'nodegree': 'No Degree (Binary)'
}

# Run the new plot
print("Launching Improved Plot...")
model.plot(
    var_names=pretty_names,
    colors=("#E69F00", "#56B4E9") # Example: Colorblind-friendly Palette (Orange/Blue)
)
