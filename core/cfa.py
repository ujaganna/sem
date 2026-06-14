import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Tuple, Optional

try:
    import semopy
    _HAS_SEMOPY = True
except ImportError:
    _HAS_SEMOPY = False


# ── Syntax builders ──────────────────────────────────────────────────────────

def build_measurement_syntax(clusters: Dict[str, List[str]]) -> str:
    return "\n".join(
        f"{name} =~ {' + '.join(items)}" for name, items in clusters.items()
    )


# ── Fit stat extraction ───────────────────────────────────────────────────────

def _extract_stats(model) -> Dict:
    """
    Pull fit stats from semopy 2.3.x.
    calc_stats() returns a DataFrame with stats as COLUMNS and a single row
    (index label 'Value') — not stats-as-index as older versions did.
    """
    try:
        raw = semopy.calc_stats(model)
        # Single-row DataFrame; stats are columns
        d = raw.iloc[0].to_dict()
    except Exception:
        return {}

    # Map semopy column names -> display names (kept ASCII-safe for PDF)
    mapping = {
        "chi2":         "chi2",
        "DoF":          "df",
        "chi2 p-value": "p(chi2)",
        "CFI":          "CFI",
        "TLI":          "TLI",
        "GFI":          "GFI",
        "RMSEA":        "RMSEA",
        "AIC":          "AIC",
        "BIC":          "BIC",
    }
    out: Dict = {}
    for raw_key, nice_key in mapping.items():
        if raw_key in d:
            try:
                out[nice_key] = round(float(d[raw_key]), 4)
            except (TypeError, ValueError):
                out[nice_key] = str(d[raw_key])
    return out


def fit_status(fit: Dict) -> Dict[str, str]:
    """Traffic-light status for each fit index."""
    rules = {
        "CFI":   (lambda v: "Good" if v >= 0.95 else ("Acceptable" if v >= 0.90 else "Poor")),
        "TLI":   (lambda v: "Good" if v >= 0.95 else ("Acceptable" if v >= 0.90 else "Poor")),
        "RMSEA": (lambda v: "Good" if v <= 0.06 else ("Acceptable" if v <= 0.08 else "Poor")),
        "GFI":   (lambda v: "Good" if v >= 0.95 else ("Acceptable" if v >= 0.90 else "Poor")),
    }
    return {
        k: rules[k](fit[k]) for k in rules if k in fit and isinstance(fit[k], (int, float))
    }


# ── CFA runner ───────────────────────────────────────────────────────────────

def run_cfa(
    df: pd.DataFrame,
    clusters: Dict[str, List[str]],
    estimator: str = "WLS",
) -> Tuple[Optional[object], pd.DataFrame, Dict, str]:
    """
    Returns (model, loadings_df, fit_dict, warning_msg).
    Tries WLS first; falls back to MLW then ML on failure.

    semopy 2.3.x inspect() encodes =~ as item ~ construct (direction reversed),
    so loadings are op=='~' rows where rval is a construct name.
    """
    if not _HAS_SEMOPY:
        return None, pd.DataFrame(), {}, "semopy not installed."

    syntax = build_measurement_syntax(clusters)
    model = semopy.Model(syntax)
    warning = ""

    for obj in [estimator, "MLW", "ML"]:
        try:
            model.fit(df, obj=obj)
            if obj != estimator:
                warning = f"WLS failed — fell back to {obj}."
            break
        except Exception as e:
            warning = str(e)

    try:
        insp = model.inspect(std_est=True)
    except Exception:
        insp = model.inspect()

    # Standardised loading column: 'Est. Std' in semopy 2.3.x
    std_col = next(
        (c for c in insp.columns if "std" in c.lower() and c != "Std. Err"), None
    )

    # Loadings: semopy stores =~ as  item ~ construct  (op='~', rval=construct)
    construct_names = list(clusters.keys())
    mask = (insp["op"] == "~") & (insp["rval"].isin(construct_names))

    load_cols = ["rval", "lval", "Estimate", "Std. Err", "z-value", "p-value"]
    if std_col:
        load_cols.append(std_col)

    loadings = insp[mask][load_cols].copy()
    rename = {
        "rval": "Construct",
        "lval": "Item",
        "Estimate": "Unstd. Loading",
        "Std. Err": "SE",
        "z-value": "z",
        "p-value": "p",
    }
    if std_col:
        rename[std_col] = "Std. Loading"
    loadings.rename(columns=rename, inplace=True)
    loadings = loadings.reset_index(drop=True)

    fit = _extract_stats(model)
    return model, loadings, fit, warning


def get_construct_correlations(
    model, constructs: List[str]
) -> pd.DataFrame:
    """Extract inter-construct correlations from fitted CFA/SEM model."""
    try:
        insp = model.inspect(std_est=True)
    except Exception:
        insp = model.inspect()

    std_col = next(
        (c for c in insp.columns if "std" in c.lower() and c != "Std. Err"), None
    )
    mask = (insp["op"] == "~~") & (insp["lval"] != insp["rval"])
    cov_df = insp[mask].copy()

    corr = pd.DataFrame(np.eye(len(constructs)), index=constructs, columns=constructs)
    for _, row in cov_df.iterrows():
        a, b = str(row["lval"]), str(row["rval"])
        if a in constructs and b in constructs:
            val = float(row[std_col]) if std_col else float(row["Estimate"])
            corr.loc[a, b] = val
            corr.loc[b, a] = val
    return corr


# ── Path diagram ─────────────────────────────────────────────────────────────

def draw_path_diagram(
    clusters: Dict[str, List[str]],
    loadings: pd.DataFrame,
    roles: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Plotly figure: latent constructs as large nodes, items as small squares,
    with loading arrows. Color-coded by role if roles dict provided.
    """
    role_color = {
        "Exogenous (IV)": "#2980b9",
        "Endogenous (DV)": "#27ae60",
        "Mediator": "#e67e22",
        "Moderator": "#8e44ad",
    }

    pos = {}
    for idx, (construct, items) in enumerate(clusters.items()):
        cx = idx * 3
        cy = len(items) / 2
        pos[construct] = (cx, cy)
        for q, item in enumerate(items):
            pos[item] = (cx + 1.5, q)

    edge_x, edge_y = [], []
    for construct, items in clusters.items():
        for item in items:
            x0, y0 = pos[construct]
            x1, y1 = pos[item]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edges_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#aaa"), hoverinfo="none",
    )

    latent_x, latent_y, latent_text, latent_color = [], [], [], []
    for construct in clusters:
        x, y = pos[construct]
        latent_x.append(x)
        latent_y.append(y)
        latent_text.append(construct)
        col = role_color.get(roles.get(construct, ""), "#2c3e50") if roles else "#2c3e50"
        latent_color.append(col)

    latent_trace = go.Scatter(
        x=latent_x, y=latent_y, mode="markers+text",
        marker=dict(size=40, color=latent_color, line=dict(width=2, color="white")),
        text=latent_text, textposition="middle center",
        textfont=dict(color="white", size=12),
        hoverinfo="text",
    )

    item_x, item_y, item_text = [], [], []
    for construct, items in clusters.items():
        for item in items:
            x, y = pos[item]
            item_x.append(x)
            item_y.append(y)
            item_text.append(item)

    items_trace = go.Scatter(
        x=item_x, y=item_y, mode="markers+text",
        marker=dict(size=20, color="#ecf0f1",
                    line=dict(width=1, color="#aaa"), symbol="square"),
        text=item_text, textposition="middle center",
        textfont=dict(color="#333", size=9),
        hoverinfo="text",
    )

    fig = go.Figure(
        data=[edges_trace, latent_trace, items_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=420,
            plot_bgcolor="white",
            title="Measurement Model",
        ),
    )
    return fig
