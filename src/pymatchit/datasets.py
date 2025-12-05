# File: src/pymatchit/datasets.py

import pandas as pd
import importlib.resources
from . import data  # This imports the data "subpackage" folder

def load_lalonde() -> pd.DataFrame:
    """
    Loads the Lalonde (1986) dataset.
    
    This dataset is widely used for benchmarking causal inference methods.
    It contains data from the National Supported Work Demonstration (NSW).
    
    Returns:
        pd.DataFrame: The Lalonde dataset.
    """
    # Robust way to find the file inside the installed package
    # We assume 'lalonde.csv' is inside the 'pymatchit.data' subpackage
    data_path = importlib.resources.files(data).joinpath("lalonde.csv")
    
    if not data_path.is_file():
        raise FileNotFoundError(f"Could not find 'lalonde.csv' in {data_path}. "
                                "Make sure the package data is installed correctly.")
        
    return pd.read_csv(data_path)