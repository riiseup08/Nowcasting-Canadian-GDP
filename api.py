import requests

BASE_URL = "https://www150.statcan.gc.ca/t1/wds/rest"

def get_download_link(product_id: str) -> str:
    """
    Get CSV download link for a Statistics Canada table.

    Parameters
    ----------
    product_id : str
        e.g. "33-10-0678-01" or "33-10-0678"  (with or without version suffix)

    Returns
    -------
    str
        URL to download the CSV/ZIP file.
    """
    # Strip the version suffix (e.g., "-01") and hyphens to get the 8-digit numeric ID
    base_id = product_id.split("-")[:3]  # e.g. ["33", "10", "0678"]
    numeric_id = "".join(base_id)        # e.g. "33100678"

    url = f"{BASE_URL}/getFullTableDownloadCSV/{numeric_id}/en"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        # API returns a list: [{"id": 1, "url": "https://..."}]
        return data[0]["url"]
    elif isinstance(data, dict) and "object" in data:
        # Fallback in case format changes
        return data["object"]
    else:
        raise ValueError(f"Unexpected API response format: {data}")
