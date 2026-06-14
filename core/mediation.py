import numpy as np
import pandas as pd
from typing import Dict, List


def bootstrap_indirect(
    df: pd.DataFrame,
    iv_col: str,
    med_col: str,
    dv_col: str,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> Dict:
    """
    Bootstrapped indirect effect (a × b) with percentile CI.
    Uses mean-score columns (observed), not latent scores.
    """
    n = len(df)
    effects = []
    cols = [iv_col, med_col, dv_col]
    data = df[cols].dropna().values

    for _ in range(n_boot):
        idx = np.random.randint(0, len(data), len(data))
        s = data[idx]
        iv, med, dv = s[:, 0], s[:, 1], s[:, 2]

        # a-path: IV → Mediator
        denom_a = np.sum((iv - iv.mean()) ** 2)
        a = np.sum((iv - iv.mean()) * (med - med.mean())) / denom_a if denom_a else 0.0

        # b-path: Mediator → DV controlling for IV
        X = np.column_stack([np.ones(len(s)), iv, med])
        try:
            coef = np.linalg.lstsq(X, dv, rcond=None)[0]
            b = float(coef[2])
        except Exception:
            continue

        effects.append(a * b)

    effects = np.array(effects)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(effects, alpha * 100))
    hi = float(np.percentile(effects, (1 - alpha) * 100))
    point = float(np.mean(effects))
    sig = (lo > 0) or (hi < 0)

    return {
        "IV": iv_col,
        "Mediator": med_col,
        "DV": dv_col,
        "Indirect Effect": round(point, 4),
        f"CI Lower ({int(ci*100)}%)": round(lo, 4),
        f"CI Upper ({int(ci*100)}%)": round(hi, 4),
        "Significant": "Yes" if sig else "No",
    }


def run_all_indirect(
    df: pd.DataFrame,
    clusters: Dict[str, List[str]],
    exogenous: List[str],
    mediators: List[str],
    endogenous: List[str],
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Run bootstrap for every IV × mediator × DV combination."""
    # Compute mean scores for constructs
    scores = pd.DataFrame(index=df.index)
    for c, items in clusters.items():
        scores[c] = df[items].mean(axis=1)

    rows = []
    for iv in exogenous:
        for med in mediators:
            for dv in endogenous:
                result = bootstrap_indirect(scores, iv, med, dv, n_boot=n_boot)
                rows.append(result)
    return pd.DataFrame(rows)
