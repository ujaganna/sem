import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def detect_clusters(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Auto-detect clusters from column names like C1.1, C1.2, C2.1 or C1_1."""
    clusters: Dict[str, List[str]] = {}
    pattern = re.compile(r'^([A-Za-z]+\d+)[._](\d+)$')
    for col in df.columns:
        m = pattern.match(str(col).strip())
        if m:
            name = m.group(1).upper()
            clusters.setdefault(name, []).append(col)
    for k in clusters:
        clusters[k] = sorted(clusters[k])
    return clusters


def validate_likert_range(
    df: pd.DataFrame, columns: List[str], min_val: int = 1, max_val: int = 7
) -> Dict[str, int]:
    """Return columns that have values outside the expected Likert range."""
    issues = {}
    for col in columns:
        bad = df[col].dropna()
        bad = bad[(bad < min_val) | (bad > max_val)]
        if len(bad):
            issues[col] = int(len(bad))
    return issues


def get_missing_summary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    missing = df[columns].isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"Missing N": missing, "Missing %": pct})


def all_items(clusters: Dict[str, List[str]]) -> List[str]:
    return [item for items in clusters.values() for item in items]


def generate_sample_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Likert-scale data for 5 clusters, 4 items each."""
    rng = np.random.default_rng(seed)
    data = {}
    for c in range(1, 6):
        factor = rng.standard_normal(n)
        for q in range(1, 5):
            raw = factor + rng.standard_normal(n) * 0.6
            likert = np.clip(np.round(raw * 1.2 + 4), 1, 7).astype(int)
            data[f"C{c}.{q}"] = likert
    return pd.DataFrame(data)
