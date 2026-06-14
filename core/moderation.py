import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from typing import Dict, List, Tuple


def build_mean_scores(
    df: pd.DataFrame, clusters: Dict[str, List[str]], constructs: List[str]
) -> pd.DataFrame:
    """Return DataFrame with mean score per selected construct."""
    out = {}
    for c in constructs:
        if c in clusters:
            out[c] = df[clusters[c]].mean(axis=1)
        elif c in df.columns:
            out[c] = df[c]
    return pd.DataFrame(out, index=df.index)


def create_interaction(
    scores: pd.DataFrame, iv: str, moderator: str
) -> Tuple[pd.DataFrame, str]:
    """Mean-centre IV and moderator, then create interaction product term."""
    scores = scores.copy()
    iv_c = scores[iv] - scores[iv].mean()
    mod_c = scores[moderator] - scores[moderator].mean()
    col_name = f"{iv}_x_{moderator}"
    scores[col_name] = iv_c * mod_c
    return scores, col_name


def simple_slopes(
    scores: pd.DataFrame, iv: str, moderator: str, dv: str
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Simple slopes at moderator = –1SD, Mean, +1SD.
    Returns a table and an interaction plot.
    """
    mod_mean = scores[moderator].mean()
    mod_sd = scores[moderator].std()
    levels = {
        "Low (–1 SD)": mod_mean - mod_sd,
        "Mean": mod_mean,
        "High (+1 SD)": mod_mean + mod_sd,
    }

    rows = []
    iv_vals = np.linspace(scores[iv].min(), scores[iv].max(), 100)
    traces = []

    for label, mod_val in levels.items():
        mod_c = scores[moderator] - mod_val
        iv_c = scores[iv] - scores[iv].mean()
        X = np.column_stack([np.ones(len(scores)), iv_c, mod_c, iv_c * mod_c])
        try:
            coef = np.linalg.lstsq(X, scores[dv], rcond=None)[0]
        except Exception:
            continue

        intercept = coef[0]
        slope = coef[1]
        _slope, _intercept, _r, p_val, _se = stats.linregress(scores[iv], scores[dv])

        rows.append({
            "Moderator Level": label,
            "Slope (b)": round(float(slope), 4),
            "p-value": round(float(p_val), 4),
            "Significant?": "Yes" if p_val < 0.05 else "No",
        })

        y_pred = intercept + slope * (iv_vals - scores[iv].mean())
        traces.append(go.Scatter(x=iv_vals, y=y_pred, mode="lines", name=label))

    slope_df = pd.DataFrame(rows)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"Interaction: {iv} × {moderator} on {dv}",
            xaxis_title=iv,
            yaxis_title=dv,
            height=380,
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ),
    )
    return slope_df, fig
