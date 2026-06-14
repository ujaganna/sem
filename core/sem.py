import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    interaction_cols: Optional[List[str]] = None,
) -> str:
    """
    Build semopy lavaan-like model string.
    Moderators enter as pre-computed mean-centred interaction columns
    (manifest moderation approach).
    """
    lines = [build_measurement_syntax(clusters), ""]

    # DV ~ all IVs + mediators + interaction terms
    for dv in endogenous:
        predictors = exogenous + mediators
        if interaction_cols:
            predictors += interaction_cols
        if predictors:
            lines.append(f"{dv} ~ {' + '.join(predictors)}")

    # Mediator ~ all IVs
    for med in mediators:
        if exogenous:
            lines.append(f"{med} ~ {' + '.join(exogenous)}")

    return "\n".join(lines)


# ── SEM runner ────────────────────────────────────────────────────────────────

def run_sem(
    df: pd.DataFrame,
    syntax: str,
    estimator: str = "WLS",
) -> Tuple[Optional[object], pd.DataFrame, Dict, str]:
    """
    Fit SEM. Returns (model, paths_df, fit_dict, warning).

    semopy 2.3.x inspect() uses op='~' for both loadings and structural paths.
    Structural paths: both lval and rval are latent construct names.
    Loadings: lval is an observed item, rval is a latent construct.
    We extract structural paths by keeping only rows where lval is a construct.
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

    std_col = next(
        (c for c in insp.columns if "std" in c.lower() and c != "Std. Err"), None
    )

    # Structural paths: op=='~' and lval is a latent construct (not an item)
    # Infer construct names from all =~ definitions in the syntax
    construct_names = [
        line.split("=~")[0].strip()
        for line in syntax.splitlines()
        if "=~" in line
    ]

    mask = (insp["op"] == "~") & (insp["lval"].isin(construct_names))
    path_cols = ["lval", "rval", "Estimate", "Std. Err", "z-value", "p-value"]
    if std_col:
        path_cols.append(std_col)

    paths = insp[mask][path_cols].copy()
    # Use ASCII-safe column names — avoids encoding errors on Windows console and PDF
    rename = {
        "lval": "Outcome",
        "rval": "Predictor",
        "Estimate": "b (unstd.)",
        "Std. Err": "SE",
        "z-value": "z",
        "p-value": "p",
    }
    if std_col:
        rename[std_col] = "b (std.)"
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
    Plotly figure: latent construct nodes with directed arrows and b (std.) labels.
    Layout: IVs left, mediators centre, DVs right, moderators bottom.
    """
    role_color = {
        "iv": "#2980b9", "dv": "#27ae60",
        "med": "#e67e22", "mod": "#8e44ad",
    }

    pos: Dict[str, Tuple[float, float]] = {}
    n_iv, n_dv, n_med = len(exogenous), len(endogenous), len(mediators)

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

    annotations = []
    beta_col = "b (std.)" if "b (std.)" in paths.columns else "b (unstd.)"
    for _, row in paths.iterrows():
        src, dst = str(row["Predictor"]), str(row["Outcome"])
        if src in pos and dst in pos:
            try:
                bval = f"{float(row[beta_col]):.3f}"
            except (ValueError, KeyError):
                bval = ""
            try:
                pv = float(row["p"])
                sig = "***" if pv < .001 else ("**" if pv < .01 else ("*" if pv < .05 else ""))
            except (ValueError, TypeError):
                sig = ""
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            annotations.append(dict(
                ax=x0, ay=y0, x=x1, y=y1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowwidth=2, arrowcolor="#555",
                text=f"<b>{bval}{sig}</b>",
                font=dict(size=10),
            ))

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

    legend_items = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=12, color=c), name=label, showlegend=True)
        for label, c in [
            ("Exogenous (IV)", "#2980b9"), ("Endogenous (DV)", "#27ae60"),
            ("Mediator", "#e67e22"),       ("Moderator", "#8e44ad"),
        ]
    ]

    fig = go.Figure(
        data=[trace] + legend_items,
        layout=go.Layout(
            annotations=annotations,
            showlegend=True,
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-1, 5.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=460,
            plot_bgcolor="white",
            title="Structural Path Diagram  (* p<.05, ** p<.01, *** p<.001)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        ),
    )
    return fig
