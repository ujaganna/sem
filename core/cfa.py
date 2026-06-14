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
    """Pull fit stats from semopy regardless of version differences."""
    try:
        raw = semopy.calc_stats(model)
        if isinstance(raw, pd.DataFrame):
            col = "Value" if "Value" in raw.columns else raw.columns[0]
            d = raw[col].to_dict()
        else:
            d = raw.to_dict()
    except Exception:
        return {}

    mapping = {
        # semopy uses these names (may vary slightly between versions)
        "chi2": "χ²",
        "dof": "df",
        "chi2 p-value": "p(χ²)",
        "CFI": "CFI",
        "TLI": "TLI",
        "RMSEA": "RMSEA",
        "SRMR": "SRMR",
        "AIC": "AIC",
        "BIC": "BIC",
        "GFI": "GFI",
    }
    out: Dict = {}
    for raw_key, nice_key in mapping.items():
        for k, v in d.items():
            if k.strip().lower() == raw_key.lower():
                try:
                    out[nice_key] = round(float(v), 4)
                except (TypeError, ValueError):
                    out[nice_key] = v
                break
    return out


def fit_status(fit: Dict) -> Dict[str, str]:
    """Traffic-light status for each fit index."""
    rules = {
        "CFI":  (lambda v: "Good" if v >= 0.95 else ("Acceptable" if v >= 0.90 else "Poor")),
        "TLI":  (lambda v: "Good" if v >= 0.95 else ("Acceptable" if v >= 0.90 else "Poor")),
        "RMSEA":(lambda v: "Good" if v <= 0.06 else ("Acceptable" if v <= 0.08 else "Poor")),
        "SRMR": (lambda v: "Good" if v <= 0.08 else ("Acceptable" if v <= 0.10 else "Poor")),
        "GFI":  (lambda v: "Good" if v >= 0.95 else ("Acceptable" if v >= 0.90 else "Poor")),
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
    Tries WLS first; falls back to MLW on failure.
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

    # Loadings (=~)
    mask = insp["op"] == "=~"
    load_cols = ["lval", "rval", "Estimate", "Std. Err", "z-value", "p-value"]
    std_col = next((c for c in insp.columns if "std" in c.lower() and c != "Std. Err"), None)
    if std_col:
        load_cols.append(std_col)

    loadings = insp[mask][load_cols].copy()
    rename = {
        "lval": "Construct", "rval": "Item",
        "Estimate": "Unstd. Loading", "Std. Err": "SE",
        "z-value": "z", "p-value": "p",
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
    """Extract inter-construct correlations from fitted CFA model."""
    try:
        insp = model.inspect(std_est=True)
    except Exception:
        insp = model.inspect()

    # Latent covariances ~~
    std_col = next((c for c in insp.columns if "std" in c.lower() and c != "Std. Err"), None)
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
    Plotly figure: latent constructs as large nodes, items as small nodes,
    loading arrows between them. Color-coded by role if roles dict provided.
    """
    G = nx.DiGraph()
    role_color = {
        "Exogenous (IV)": "#2980b9",
        "Endogenous (DV)": "#27ae60",
        "Mediator": "#e67e22",
        "Moderator": "#8e44ad",
    }

    for construct, items in clusters.items():
        G.add_node(construct, node_type="latent")
        for item in items:
            G.add_node(item, node_type="item")
            G.add_edge(construct, item)

    pos = {}
    n_constructs = len(clusters)
    for idx, (construct, items) in enumerate(clusters.items()):
        cx = idx * 3
        cy = len(items) / 2
        pos[construct] = (cx, cy)
        for q, item in enumerate(items):
            pos[item] = (cx + 1.5, q)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edges_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#aaa"), hoverinfo="none"
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
        marker=dict(size=20, color="#ecf0f1", line=dict(width=1, color="#aaa"), symbol="square"),
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
