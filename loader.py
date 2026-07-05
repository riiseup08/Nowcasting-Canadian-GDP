import pandas as pd
from pathlib import Path


def load_statcan_csv(folder_path: str):
    folder_path = Path(folder_path)

    # Each extracted table folder contains the data CSV *and* a "<id>_MetaData.csv".
    # Pick the data file, never the metadata sidecar.
    csv_files = [f for f in folder_path.glob("*.csv") if "MetaData" not in f.name]
    csv_file = csv_files[0]

    df = pd.read_csv(csv_file)

    return df