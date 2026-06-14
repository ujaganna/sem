import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Optional, Tuple

try:
    import semopy
    _HAS_SEMOPY = True
except ImportError:
    _HAS_SEMOPY = False

from core.cfa import build_measurement_syntax, _extract_stats


# ── Syntax builder ────────────────────────────────────────────────────────────

def build_sem_syntax(
    clusters: Dict[str, List[str]],
    exogenous: List[str],
    endogenous: List[str],
    mediators: List[str],
    interaction_cols: Optional[List[str]] = None,   # observed interaction terms
    moderator_targets: Optional[Dict[str, str]] = None,  # {interaction_col: dv}
) -> str:
    """
    Build semopy lavaan-like model string.
    Moderators enter as observed interaction columns (manifest moderation).
    """
    lines = [build_measurement_syntax(clusters), ""]

    all_ivs = exogenous
    all_dvs = endogenous

    # DV ~ all IVs + mediators + interaction terms
    for dv in all_dvs:
        predictors = all_ivs + mediators
        if interaction_cols:
            predictors += interaction_cols
        if predictors:
            lines.append(f"{dv} ~ {' + '.join(predictors)}")

    # Mediator ~ all IVs
    for med in mediators:
        if all_ivs:
            lines.append(f"{med} ~ {' + '.join(all_ivs)}")

    return "\n".join(lines)


# ── SEM runner ────────────────────────────────────────────────────────────────

def run_sem(
    df: pd.DataFrame,
    syntax: str,
    estimator: str = "WLS",
) -> Tuple[Optional[object], pd.DataFrame, Dict, str]:
    """
    Fit SEM. Returns (model, paths_df, fit_dict, warning).
    """
    if not _HAS_SEMOPY:
        return None, pd.DataFrame(), {}, "semopy not installed."

    model = semopy.Model(syntax)
    warning = ""

    for obj in [estimator, "MLW", "ML"]:
        try:
            model.fit(df, obj=obj)
            if obj != estimator:
                warning = f"Fell back to {obj} estimator."
            break
        except Exception as e:
            warning = str(e)

    try:
        insp = model.inspect(std_est=True)
    except Exception:
        insp = model.inspect()

    std_col = next((c for c in insp.columns if "std" in c.lower() and c != "Std. Err"), None)

    # Structural paths ~
    mask = insp["op"] == "~"
    path_cols = ["lval", "rval", "Estimate", "Std. Err", "z-value", "p-value"]
    if std_col:
        path_cols.append(std_col)

    paths = insp[mask][path_cols].copy()
    rename = {
        "lval": "Outcome", "rval": "Predictor",
        "Estimate": "β (unstd.)", "Std. Err": "SE",
        "z-value": "z", "p-value": "p",
    }
    if std_col:
        rename[std_col] = "β (std.)"
    paths.rename(columns=rename, inplace=True)
    paths = paths.reset_index(drop=True)

    fit = _extract_stats(model)
    return model, paths, fit, warning


# ── Structural path diagram ───────────────────────────────────────────────────

def draw_structural_diagram(
    exogenous: List[str],
    endogenous: List[str],
    mediators: List[str],
    moderators: List[str],
    paths: pd.DataFrame,
) -> go.Figure:
    """
    Plotly graph of the structural model only (latent nodes, arrows with β labels).
    Layout: IVs left, mediators centre, DVs right, moderators bottom.
    """
    role_color = {
        "iv": "#2980b9",
        "dv": "#27ae60",
        "med": "#e67e22",
        "mod": "#8e44ad",
    }

    # Position nodes
    pos: Dict[str, Tuple[float, float]] = {}
    n_iv = len(exogenous)
    n_dv = len(endogenous)
    n_med = len(mediators)

    for i, node in enumerate(exogenous):
        pos[node] = (0.0, (i - (n_iv - 1) / 2) * 1.5)
    for i, node in enumerate(endogenous):
        pos[node] = (4.0, (i - (n_dv - 1) / 2) * 1.5)
    for i, node in enumerate(mediators):
        pos[node] = (2.0, (i - (n_med - 1) / 2) * 1.5)
    for i, node in enumerate(moderators):
        pos[node] = (1.0 + i * 2.0, -3.0)

    role_map = (
        {n: "iv" for n in exogenous}
        | {n: "dv" for n in endogenous}
        | {n: "med" for n in mediators}
        | {n: "mod" for n in moderators}
    )

    # Build annotation-based arrows (plotly doesn't do true directed edges cleanly)
    annotations = []
    for _, row in paths.iterrows():
        src, dst = str(row["Predictor"]), str(row["Outcome"])
        if src in pos and dst in pos:
            beta_col = "β (std.)" if "β (std.)" in paths.columns else "β (unstd.)"
            try:
                beta_val = f"{float(row[beta_col]):.3f}"
            except (ValueError, KeyError):
                beta_val = ""
            p_val = float(row["p"])
            sig = "***" if p_val < .001 else ("**" if p_val < .01 else ("*" if p_val < .05 else ""))
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            annotations.append(dict(
                ax=x0, ay=y0, x=x1, y=y1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowwidth=2, arrowcolor="#555",
                text=f"<b>{beta_val}{sig}</b>",
                font=dict(size=10),
            ))

    # Nodes
    node_x, node_y, node_text, node_color = [], [], [], []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_color.append(role_color.get(role_map.get(node, "iv"), "#888"))

    trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=55, color=node_color, line=dict(width=2, color="white")),
        text=node_text, textposition="middle center",
        textfont=dict(color="white", size=13, family="Arial Bold"),
        hoverinfo="text",
    )

    # Legend via invisible scatter
    legend_items = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color=c),
                   name=label, showlegend=True)
        for label, c in [
            ("Exogenous (IV)", "#2980b9"),
            ("Endogenous (DV)", "#27ae60"),
            ("Mediator", "#e67e22"),
            ("Moderator", "#8e44ad"),
        ]
    ]

    fig = go.Figure(
        data=[trace] + legend_items,
        layout=go.Layout(
            annotations=annotations,
            showlegend=True,
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 5.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=460,
            plot_bgcolor="white",
            title="Structural Path Diagram  (* p<.05, ** p<.01, *** p<.001)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ),
    )
    return fig
