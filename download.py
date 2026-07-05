import requests
import zipfile
from pathlib import Path

def download_file(url: str, output_path: str):
    """Download any file (zip or csv) from a URL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path

def extract_zip(zip_path: str, extract_to: str):
    """Extract a ZIP file to a folder."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    return extract_to