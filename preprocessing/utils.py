"""
utils.py
--------
General-purpose helper functions for data cleaning and transformation.
"""

import pandas as pd
import re


def nepali_to_english(num_str):
    """Convert Nepali digits to English digits."""
    if pd.isna(num_str):
        return num_str
    mapping = str.maketrans("०१२३४५६७८९", "0123456789")
    return str(num_str).translate(mapping)


def clean_number(value):
    """Clean numeric or currency-like strings and convert to float."""
    if pd.isna(value):
        return None
    value = str(value).replace("रू", "").replace(",", "").strip()
    value = nepali_to_english(value)
    try:
        return float(value)
    except Exception:
        return None


def clean_commodity(name):
    """Normalize Nepali commodity names into standard English categories."""
    name = re.sub(r"\(.*?\)", "", str(name)).strip()
    mapping = {
        "गोलभेडा ठूलो": "Tomato_Big",
        "गोलभेडा सानो": "Tomato_Small",
        "गोलभेडा": "Tomato"
    }
    for np_name, en_name in mapping.items():
        if np_name in name:
            return en_name
    return name

