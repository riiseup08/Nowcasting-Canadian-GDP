"""
warehouse.py — PostgreSQL data warehouse interface
==================================================
Handles all database interaction for the GDP nowcast pipeline.

Schema
------
The warehouse contains a single table ``gdp_nowcast_training`` with the
full feature-engineered dataset (one row per month).

Environment variables
---------------------
- DB_HOST     (default: localhost)
- DB_PORT     (default: 5432)
- DB_NAME     (default: postgres)
- DB_USER     (default: postgres)
- DB_PASSWORD (default: postgres)

Usage
-----
    from warehouse import get_connection, save_training_data, load_training_data

    conn = get_connection()
    save_training_data(conn, df)
    df = load_training_data(conn)
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

TABLE_NAME = "gdp_nowcast_training"


def get_db_url() -> str:
    """
    Build the PostgreSQL connection URL.

    Prefers a full ``DATABASE_URL`` (e.g. the Supabase connection string) if set,
    otherwise assembles one from the individual DB_* components.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def get_engine():
    """Return a SQLAlchemy engine connected to the warehouse database."""
    return create_engine(get_db_url())


def get_connection():
    """Return a raw database connection for pandas ``io`` methods."""
    return get_engine().connect()


def create_table_if_not_exists(conn):
    """
    Create the warehouse table schema if it doesn't exist.

    The table stores the full feature-engineered dataset with a date column
    and all feature columns plus the target (gdp_value).
    """
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        date DATE NOT NULL UNIQUE,
        lfs_value DOUBLE PRECISION,
        cpi_value DOUBLE PRECISION,
        mfg_value DOUBLE PRECISION,
        gdp_value DOUBLE PRECISION,
        lfs_value_lag_1 DOUBLE PRECISION,
        lfs_value_lag_2 DOUBLE PRECISION,
        lfs_value_lag_3 DOUBLE PRECISION,
        lfs_value_roll_mean_3 DOUBLE PRECISION,
        lfs_value_roll_mean_6 DOUBLE PRECISION,
        lfs_value_growth DOUBLE PRECISION,
        cpi_value_lag_1 DOUBLE PRECISION,
        cpi_value_lag_2 DOUBLE PRECISION,
        cpi_value_lag_3 DOUBLE PRECISION,
        cpi_value_roll_mean_3 DOUBLE PRECISION,
        cpi_value_roll_mean_6 DOUBLE PRECISION,
        cpi_value_growth DOUBLE PRECISION,
        mfg_value_lag_1 DOUBLE PRECISION,
        mfg_value_lag_2 DOUBLE PRECISION,
        mfg_value_lag_3 DOUBLE PRECISION,
        mfg_value_roll_mean_3 DOUBLE PRECISION,
        mfg_value_roll_mean_6 DOUBLE PRECISION,
        mfg_value_growth DOUBLE PRECISION
    );
    """
    conn.execute(text(create_sql))
    conn.commit()


def save_training_data(conn, df: pd.DataFrame):
    """
    Upsert the feature-engineered DataFrame into the warehouse table.

    Uses PostgreSQL's ``INSERT ... ON CONFLICT (date) DO UPDATE`` so that
    re-running the transform updates existing rows instead of duplicating them.
    """
    # Ensure table exists
    create_table_if_not_exists(conn)

    # Get the list of columns that exist in the table
    inspector = inspect(conn)
    table_cols = {col["name"] for col in inspector.get_columns(TABLE_NAME)}

    # Only keep DataFrame columns that match the table schema (skip 'id')
    cols_to_write = [c for c in df.columns if c in table_cols]

    if "date" not in cols_to_write:
        cols_to_write = ["date"] + cols_to_write

    # Build the upsert SQL dynamically
    insert_cols = ", ".join(cols_to_write)
    placeholders = ", ".join([f":{c}" for c in cols_to_write])

    update_parts = [f"{c} = EXCLUDED.{c}" for c in cols_to_write if c != "date"]
    update_clause = ", ".join(update_parts)

    upsert_sql = f"""
    INSERT INTO {TABLE_NAME} ({insert_cols})
    VALUES ({placeholders})
    ON CONFLICT (date) DO UPDATE SET {update_clause};
    """

    # Write row by row using the connection
    records = df[cols_to_write].to_dict(orient="records")
    for record in records:
        # Convert NaN to None for PostgreSQL compatibility
        clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        conn.execute(text(upsert_sql), clean)

    conn.commit()
    print(f"  [ok] Upserted {len(df)} rows to warehouse table '{TABLE_NAME}'")


def load_training_data(conn) -> pd.DataFrame:
    """Load the full warehouse table into a pandas DataFrame."""
    return pd.read_sql_table(TABLE_NAME, conn, parse_dates=["date"])