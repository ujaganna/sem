import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except ImportError:
    _HAS_PINGOUIN = False


# ── Normality ────────────────────────────────────────────────────────────────

def test_normality_items(df: pd.DataFrame, items: List[str]) -> pd.DataFrame:
    """Univariate normality per Likert item (Shapiro-Wilk ≤ 5000 obs)."""
    rows = []
    for item in items:
        col = df[item].dropna()
        skew = float(stats.skew(col))
        kurt = float(stats.kurtosis(col))
        if len(col) <= 5000:
            sw_stat, sw_p = stats.shapiro(col)
        else:
            sw_stat, sw_p = np.nan, np.nan
        normal = abs(skew) < 2.0 and abs(kurt) < 7.0
        rows.append({
            "Item": item,
            "N": len(col),
            "Mean": round(float(col.mean()), 3),
            "SD": round(float(col.std()), 3),
            "Skewness": round(skew, 3),
            "Kurtosis (excess)": round(kurt, 3),
            "SW Statistic": round(sw_stat, 4) if not np.isnan(sw_stat) else "—",
            "SW p-value": round(sw_p, 4) if not np.isnan(sw_p) else "—",
            "Normal?": "Yes" if normal else "No",
        })
    return pd.DataFrame(rows)


def mardia_test(df: pd.DataFrame, items: List[str]) -> Dict:
    """Mardia's multivariate normality (skewness + kurtosis)."""
    data = df[items].dropna().values
    n, p = data.shape
    mu = data.mean(axis=0)
    S = np.cov(data.T)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return {"Error": "Singular covariance matrix — check for redundant items."}

    diff = data - mu
    D = diff @ S_inv @ diff.T  # n x n
    diag_D = np.diag(D)

    # Mardia skewness b1p
    b1p = float(np.sum(D ** 3) / (n ** 2))
    chi2_b1 = n * b1p / 6.0
    df_b1 = p * (p + 1) * (p + 2) / 6
    p_b1 = float(1 - stats.chi2.cdf(chi2_b1, df_b1))

    # Mardia kurtosis b2p
    b2p = float(np.sum(diag_D ** 2) / n)
    expected_kurt = p * (p + 2)
    z_b2 = (b2p - expected_kurt) / np.sqrt(8 * p * (p + 2) / n)
    p_b2 = float(2 * (1 - stats.norm.cdf(abs(z_b2))))

    mv_normal = p_b1 > 0.05 and p_b2 > 0.05
    return {
        "Mardia Skewness (b1p)": round(b1p, 4),
        "Skewness χ²": round(chi2_b1, 4),
        "Skewness p-value": round(p_b1, 4),
        "Mardia Kurtosis (b2p)": round(b2p, 4),
        "Kurtosis Z": round(z_b2, 4),
        "Kurtosis p-value": round(p_b2, 4),
        "Multivariate Normal?": "Yes" if mv_normal else "No",
    }


# ── Reliability ───────────────────────────────────────────────────────────────

def cronbach_alpha(df: pd.DataFrame, items: List[str]) -> Tuple[float, pd.DataFrame]:
    """Cronbach's alpha + item-level diagnostics."""
    sub = df[items].dropna()
    k = len(items)
    if k < 2:
        return np.nan, pd.DataFrame()

    if _HAS_PINGOUIN:
        alpha_val = float(pg.cronbach_alpha(data=sub)[0])
    else:
        item_vars = sub.var(axis=0, ddof=1)
        total_var = sub.sum(axis=1).var(ddof=1)
        alpha_val = float((k / (k - 1)) * (1 - item_vars.sum() / total_var))

    rows = []
    for item in items:
        others = [i for i in items if i != item]
        rest = sub[others]
        total = rest.sum(axis=1)
        corr = float(sub[item].corr(total))

        rest_vars = rest.var(axis=0, ddof=1)
        rest_total_var = rest.sum(axis=1).var(ddof=1)
        k2 = len(others)
        a_del = float((k2 / (k2 - 1)) * (1 - rest_vars.sum() / rest_total_var)) if k2 >= 2 else np.nan

        rows.append({
            "Item": item,
            "Mean": round(float(sub[item].mean()), 3),
            "SD": round(float(sub[item].std()), 3),
            "Item-Total r": round(corr, 3),
            "α if Deleted": round(a_del, 3) if not np.isnan(a_del) else "—",
            "Keep?": "Yes" if corr >= 0.30 else "Review",
        })
    return alpha_val, pd.DataFrame(rows)


def compute_ave_cr(loadings: np.ndarray) -> Tuple[float, float]:
    """AVE and Composite Reliability from standardised factor loadings."""
    l2 = loadings ** 2
    ave = float(l2.mean())
    cr_num = float(loadings.sum() ** 2)
    cr_den = cr_num + float((1 - l2).sum())
    cr = cr_num / cr_den if cr_den > 0 else np.nan
    return ave, cr


# ── Discriminant Validity ────────────────────────────────────────────────────

def htmt_ratio(df: pd.DataFrame, clusters: Dict[str, List[str]]) -> pd.DataFrame:
    """HTMT matrix (upper-triangle only; < 0.85 passes)."""
    names = list(clusters.keys())
    mat = pd.DataFrame(np.nan, index=names, columns=names)

    for i, c1 in enumerate(names):
        for j, c2 in enumerate(names):
            if i >= j:
                continue
            items1, items2 = clusters[c1], clusters[c2]

            het = [abs(df[a].corr(df[b])) for a in items1 for b in items2]
            mon1 = [abs(df[items1[a]].corr(df[items1[b]]))
                    for a in range(len(items1)) for b in range(a + 1, len(items1))]
            mon2 = [abs(df[items2[a]].corr(df[items2[b]]))
                    for a in range(len(items2)) for b in range(a + 1, len(items2))]

            if het and mon1 and mon2:
                val = np.mean(het) / np.sqrt(np.mean(mon1) * np.mean(mon2))
                mat.loc[c1, c2] = round(float(val), 3)
                mat.loc[c2, c1] = round(float(val), 3)
    return mat


def fornell_larcker(
    ave_dict: Dict[str, float], construct_corr: pd.DataFrame
) -> pd.DataFrame:
    """Fornell-Larcker table: √AVE on diagonal, inter-construct correlations off-diagonal."""
    names = list(ave_dict.keys())
    rows = []
    for c1 in names:
        row: Dict = {"Construct": c1}
        for c2 in names:
            if c1 == c2:
                row[c2] = round(np.sqrt(ave_dict[c1]), 3)
            elif c1 in construct_corr.index and c2 in construct_corr.columns:
                row[c2] = round(float(construct_corr.loc[c1, c2]), 3)
            else:
                row[c2] = "—"
        rows.append(row)
    return pd.DataFrame(rows)
