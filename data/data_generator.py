import pandas as pd
import numpy as np

def generate_prices(n=1000):
    prices = np.cumsum(np.random.normal(0, 1, n)) + 100
    return pd.DataFrame({"close": prices})
