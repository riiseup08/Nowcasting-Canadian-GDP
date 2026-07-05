# features.py (modified)
import pandas as pd

# Make these functions accept a 'column' parameter
def create_lag_features(df: pd.DataFrame, column: str, lags=[1, 2, 3]):
    df = df.sort_values("date").copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)  # Note: uses column name in output
    return df

def create_rolling_features(df: pd.DataFrame, column: str, windows=[3, 6]):
    df = df.sort_values("date").copy()
    for w in windows:
        df[f"{column}_roll_mean_{w}"] = df[column].rolling(window=w).mean()
    return df

def create_growth_rate(df: pd.DataFrame, column: str):
    df = df.sort_values("date").copy()
    df[f"{column}_growth"] = df[column].pct_change()
    return df