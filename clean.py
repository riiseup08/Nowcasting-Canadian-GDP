# clean.py (modified)
import warnings
import pandas as pd

def load_raw_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df

def clean_statcan(df: pd.DataFrame, geo_filter: str = None, filters: dict = None):
    """
    Reduce a raw StatCan table to a single ``date``/``value`` time series.

    StatCan tables carry many dimensions (industry, seasonal adjustment, age,
    product group, ...). We must select ONE member per dimension so that exactly
    one value remains per reference date -- otherwise the series is a meaningless
    average across every dimension.

    Parameters
    ----------
    geo_filter : str, optional
        Convenience filter matched with case-insensitive ``str.contains`` on GEO
        (e.g. "Canada"). Equivalent to ``filters={"GEO": ...}`` but fuzzy.
    filters : dict, optional
        ``{column_name: member_value}`` pairs matched exactly (``==``) against the
        raw dimension columns, applied BEFORE non-essential columns are dropped.
    """
    df = df.copy()

    # --- Select a single series while all dimension columns still exist ---
    if geo_filter and "GEO" in df.columns:
        df = df[df["GEO"].str.contains(geo_filter, case=False, na=False)]

    if filters:
        for col, member in filters.items():
            if col not in df.columns:
                raise KeyError(
                    f"clean_statcan: filter column {col!r} not in table columns "
                    f"{list(df.columns)}"
                )
            df = df[df[col] == member]

    # --- Reduce to the essentials ---
    cols_to_keep = [c for c in ["REF_DATE", "VALUE"] if c in df.columns]
    df = df[cols_to_keep].copy()

    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["REF_DATE", "VALUE"])
    df = df.sort_values("REF_DATE")

    # Handle dates: if REF_DATE is numeric (e.g. years like 2023),
    # convert to string and parse as year-01-01 to avoid nanosecond-epoch bug
    if pd.api.types.is_numeric_dtype(df["REF_DATE"]):
        df["date"] = pd.to_datetime(df["REF_DATE"].astype(str) + "-01-01", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["REF_DATE"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.rename(columns={"VALUE": "value"})[["date", "value"]]

    # After a correct filter there should be exactly one row per date. If not,
    # the filter is under-specified -- warn loudly instead of silently averaging.
    dupes = df["date"].duplicated().sum()
    if dupes:
        warnings.warn(
            f"clean_statcan: {dupes} duplicate dates remain after filtering "
            f"(filter under-specified) -- averaging as a fallback.",
            stacklevel=2,
        )
        df = df.groupby("date", as_index=False)["value"].mean()

    return df.reset_index(drop=True)

def aggregate_monthly(df: pd.DataFrame):
    return df.groupby("date", as_index=False)["value"].mean()
