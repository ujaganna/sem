"""
PDF report generator for the SEM Analysis Tool.
All numeric values are formatted to 4 decimal places.
Table cells use Paragraph objects for automatic word-wrap (prevents overflow).
Structural path diagram is drawn with matplotlib and embedded as a PNG image.
"""

import io
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be set before pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

# ── Palette ───────────────────────────────────────────────────────────────────
_HEADER = colors.HexColor("#2c3e50")
_EVEN   = colors.HexColor("#f4f6f7")
_ODD    = colors.white
_C_GOOD   = "#1e8449"
_C_ACCEPT = "#d4ac0d"
_C_POOR   = "#c0392b"
_C_NEUTRAL = "#000000"

_STATUS_COLORS = {
    "Good":       _C_GOOD,   "Pass":    _C_GOOD,   "Yes":   _C_GOOD,
    "***":        _C_GOOD,
    "Acceptable": _C_ACCEPT, "**":      _C_ACCEPT, "*":     _C_ACCEPT,
    "Poor":       _C_POOR,   "Fail":    _C_POOR,   "No":    _C_POOR,
    "n.s.":       _C_POOR,   "Review":  _C_POOR,
}

_ROLE_COLORS = {
    "Exogenous (IV)": "#2980b9",
    "Endogenous (DV)": "#27ae60",
    "Mediator":        "#e67e22",
    "Moderator":       "#8e44ad",
}


# ── String sanitisers ─────────────────────────────────────────────────────────

def _sp(s) -> str:
    """Safe plain text for table cells — replaces Unicode not in Latin-1."""
    return (
        str(s)
        .replace("χ²", "chi2").replace("χ", "chi").replace("²", "2")
        .replace("—", "-").replace("–", "-")
        .replace("α", "alpha").replace("β", "b")
        .replace("√", "sqrt").replace("≥", ">=").replace("≤", "<=")
        .replace("→", "->").replace("×", "x").replace("±", "+/-")
    )


def _sx(s) -> str:
    """Safe XML text for Paragraph — also escapes &, <, >."""
    return (
        _sp(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fmt(v) -> str:
    """Format any value: floats to 4 decimal places, others as safe plain text."""
    raw = _sp(str(v))
    if raw in ("-", "", "nan", "NaN", "None", "none"):
        return "-"
    try:
        f = float(raw.replace(",", ""))
        return f"{f:.4f}"
    except (ValueError, TypeError):
        return raw


# ── Paragraph cell helpers ────────────────────────────────────────────────────

def _cell(text, style) -> Paragraph:
    return Paragraph(_sx(_fmt(text)), style)


def _status_cell(text, style) -> Paragraph:
    val = _sp(str(text))
    hex_c = _STATUS_COLORS.get(val, _C_NEUTRAL)
    return Paragraph(f'<font color="{hex_c}"><b>{_sx(val)}</b></font>', style)


# ── Column width heuristic ────────────────────────────────────────────────────

_NARROW = {"n", "z", "p", "se", "df", "sig.", "sig", "rank", "id", "cr", "ave"}
_WIDE   = {"benchmark", "description", "mediators", "mediation type",
           "mediation", "direct path"}


def _auto_widths(headers: List[str], total_cm: float = 17.0) -> List[float]:
    weights = []
    for h in headers:
        hl = h.lower().strip()
        if hl in _NARROW or len(hl) <= 3:
            weights.append(1.0)
        elif hl in _WIDE or len(hl) > 14:
            weights.append(3.2)
        else:
            weights.append(1.9)
    s = sum(weights)
    return [w / s * total_cm * cm for w in weights]


# ── Base table style (no TEXTCOLOR — colours handled inside Paragraphs) ───────

def _tbl_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  _HEADER),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
        ("PADDING",        (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_ODD, _EVEN]),
    ])


# ── DataFrame → Table ─────────────────────────────────────────────────────────

def _df_table(
    df: pd.DataFrame,
    status_col: Optional[str] = None,
    col_widths: Optional[List[float]] = None,
    cell_style=None,
    hdr_style=None,
) -> Table:
    if cell_style is None:
        cell_style = ParagraphStyle("cell", fontSize=8, leading=10, wordWrap="CJK")
    if hdr_style is None:
        hdr_style = ParagraphStyle(
            "hdr", fontSize=8, fontName="Helvetica-Bold",
            textColor=colors.white, leading=10, wordWrap="CJK",
        )

    safe_hdrs = [_sp(h) for h in df.columns]
    status_idx = safe_hdrs.index(_sp(status_col)) if status_col and _sp(status_col) in safe_hdrs else None

    headers_row = [Paragraph(_sx(h), hdr_style) for h in safe_hdrs]
    data = [headers_row]
    for _, row in df.iterrows():
        cells = []
        for j, v in enumerate(row):
            if j == status_idx:
                cells.append(_status_cell(v, cell_style))
            else:
                cells.append(_cell(v, cell_style))
        data.append(cells)

    widths = col_widths if col_widths else _auto_widths(safe_hdrs)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(_tbl_style())
    return t


# ── Matplotlib structural path diagram ───────────────────────────────────────

def _draw_diagram(roles: Dict[str, str], sem_paths: pd.DataFrame) -> bytes:
    """Return PNG bytes of the structural path diagram."""
    exog = [c for c, r in roles.items() if r == "Exogenous (IV)"]
    endo = [c for c, r in roles.items() if r == "Endogenous (DV)"]
    meds = [c for c, r in roles.items() if r == "Mediator"]
    mods = [c for c, r in roles.items() if r == "Moderator"]

    pos: Dict[str, tuple] = {}
    for i, c in enumerate(exog):
        pos[c] = (0.8, (i - (len(exog) - 1) / 2) * 1.6)
    for i, c in enumerate(endo):
        pos[c] = (5.2, (i - (len(endo) - 1) / 2) * 1.6)
    for i, c in enumerate(meds):
        pos[c] = (3.0, (i - (len(meds) - 1) / 2) * 1.6)
    for i, c in enumerate(mods):
        pos[c] = (1.6 + i * 2.0, -2.5)

    all_y = [y for _, y in pos.values()] if pos else [0.0, 1.0]
    y_lo, y_hi = min(all_y) - 1.5, max(all_y) + 1.5
    fig_h = max(5.0, (y_hi - y_lo) * 1.3)

    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(-0.3, 6.3)
    ax.set_ylim(y_lo, y_hi)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    beta_col = "b (std.)" if "b (std.)" in sem_paths.columns else "b (unstd.)"

    for _, row in sem_paths.iterrows():
        src = str(row.get("Predictor", ""))
        dst = str(row.get("Outcome", ""))
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color="#444",
                lw=1.8, mutation_scale=18,
                connectionstyle="arc3,rad=0.06",
            ),
        )
        try:
            bval = f"{float(row.get(beta_col, 'nan')):.4f}"
            pval = float(row.get("p", 1.0))
            sig = "***" if pval < .001 else ("**" if pval < .01 else ("*" if pval < .05 else ""))
            lbl = f"{bval}{sig}"
        except (ValueError, TypeError):
            lbl = ""
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.14
        ax.text(mx, my, lbl, ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#111",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))

    radius = 0.42
    for c, (x, y) in pos.items():
        fc = _ROLE_COLORS.get(roles.get(c, ""), "#888")
        circ = plt.Circle((x, y), radius, color=fc, zorder=5, linewidth=0)
        ax.add_patch(circ)
        ax.text(x, y, c, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=6)

    patches = [mpatches.Patch(color=_ROLE_COLORS[r], label=r)
               for r in ["Exogenous (IV)", "Endogenous (DV)", "Mediator", "Moderator"]
               if any(rv == r for rv in roles.values())]
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.85, edgecolor="#ccc")
    ax.set_title(
        "Structural Path Diagram   (* p < .05, ** p < .01, *** p < .001)",
        fontsize=11, fontweight="bold", pad=10,
    )

    buf = io.BytesIO()
    fig.tight_layout(pad=0.6)
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Fit status helper ─────────────────────────────────────────────────────────

def _fit_status(key: str, val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "-"
    rules = {
        "CFI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
        "TLI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
        "RMSEA": lambda x: "Good" if x <= 0.06 else ("Acceptable" if x <= 0.08 else "Poor"),
        "GFI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
    }
    return rules.get(_sp(key), lambda _: "-")(v)


def _sig_label(p) -> str:
    try:
        v = float(p)
        if v < 0.001: return "***"
        if v < 0.01:  return "**"
        if v < 0.05:  return "*"
        return "n.s."
    except (TypeError, ValueError):
        return "-"


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf(
    filepath: str,
    model_name: str,
    constructs: Dict[str, str],          # {construct: role}
    reliability: Dict[str, Dict],
    normality_df: pd.DataFrame,
    mardia: Dict,
    cfa_loadings: pd.DataFrame,
    cfa_fit: Dict,
    sem_paths: pd.DataFrame,
    sem_fit: Dict,
    roles: Optional[Dict[str, str]] = None,
    indirect_effects: Optional[pd.DataFrame] = None,
    model_comparison: Optional[pd.DataFrame] = None,
) -> None:

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    SS = getSampleStyleSheet()
    H1   = ParagraphStyle("H1",  parent=SS["Heading1"], fontSize=13,
                          spaceAfter=5, textColor=_HEADER)
    H2   = ParagraphStyle("H2",  parent=SS["Heading2"], fontSize=10,
                          spaceAfter=4, textColor=_HEADER)
    BODY = ParagraphStyle("BODY", parent=SS["Normal"],  fontSize=8, spaceAfter=4)
    TTL  = ParagraphStyle("TTL",  parent=SS["Title"],   fontSize=20, spaceAfter=4)
    NOTE = ParagraphStyle("NOTE", parent=SS["Normal"],  fontSize=7,
                          textColor=colors.HexColor("#666"))
    CELL = ParagraphStyle("CELL", fontSize=8, leading=10, wordWrap="CJK")
    HDR  = ParagraphStyle("HDR",  fontSize=8, fontName="Helvetica-Bold",
                          textColor=colors.white, leading=10, wordWrap="CJK")

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#bdc3c7"), spaceAfter=6)

    def sp(h=0.3):
        return Spacer(1, h * cm)

    def fit_table(fit_dict: Dict) -> Table:
        benchmarks = {
            "chi2":    "-", "df": "-", "p(chi2)": "-",
            "CFI":     ">= 0.95 good / >= 0.90 acceptable",
            "TLI":     ">= 0.95 good / >= 0.90 acceptable",
            "RMSEA":   "<= 0.06 good / <= 0.08 acceptable",
            "GFI":     ">= 0.95 good / >= 0.90 acceptable",
            "AIC":     "Lower is better",
            "BIC":     "Lower is better",
        }
        hdrs = ["Index", "Value", "Benchmark", "Status"]
        hdr_row = [Paragraph(_sx(h), HDR) for h in hdrs]
        data = [hdr_row]
        for k, v in fit_dict.items():
            status = _fit_status(k, v)
            data.append([
                _cell(k, CELL),
                _cell(v, CELL),
                Paragraph(_sx(benchmarks.get(_sp(k), "-")), CELL),
                _status_cell(status, CELL),
            ])
        t = Table(data, colWidths=[2.5*cm, 2.8*cm, 8.2*cm, 3.5*cm], repeatRows=1)
        t.setStyle(_tbl_style())
        return t

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story += [
        sp(1),
        Paragraph("SEM Analysis Report", TTL),
        Paragraph(f"Model: {_sx(model_name)}", H1),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", BODY),
        sp(0.4), hr(), sp(0.2),
    ]

    # ── 1. Constructs ──────────────────────────────────────────────────────────
    story.append(Paragraph("1. Latent Constructs and Roles", H1))
    hdr_row = [Paragraph(_sx(h), HDR) for h in ["Construct", "Role"]]
    role_data = [hdr_row] + [
        [_cell(k, CELL), _cell(v, CELL)] for k, v in constructs.items()
    ]
    t = Table(role_data, colWidths=[5*cm, 12*cm])
    t.setStyle(_tbl_style())
    story += [t, sp()]

    # ── 2. Normality ───────────────────────────────────────────────────────────
    story.append(Paragraph("2. Normality Assessment", H1))
    story.append(Paragraph(
        "WLS estimator is used throughout - robust to non-normal and ordinal "
        "Likert-scale data. Normality is assessed for reporting purposes only.", BODY,
    ))
    story.append(sp(0.2))

    if not normality_df.empty:
        story.append(Paragraph(
            "Univariate Normality  (|Skew| &lt; 2 and |Kurtosis| &lt; 7 = Normal)", H2))
        nd = normality_df.copy()
        nd.columns = [_sp(c) for c in nd.columns]
        story += [_df_table(nd, status_col="Normal?"), sp(0.2)]

    if mardia and "Error" not in mardia:
        mv   = _sp(mardia.get("Multivariate Normal?", "-"))
        skch = _fmt(mardia.get("Skewness chi2", mardia.get("Skewness chi2", "-")))
        skp  = _fmt(mardia.get("Skewness p-value", "-"))
        kuz  = _fmt(mardia.get("Kurtosis Z", "-"))
        kup  = _fmt(mardia.get("Kurtosis p-value", "-"))
        col  = _C_GOOD if mv == "Yes" else _C_POOR
        story.append(Paragraph(
            f'Mardia Multivariate Normality: '
            f'<font color="{col}"><b>{_sx(mv)}</b></font>  '
            f'(Skewness chi2={skch}, p={skp};  Kurtosis Z={kuz}, p={kup})',
            BODY,
        ))
    story.append(sp())

    # ── 3. Reliability ─────────────────────────────────────────────────────────
    story.append(Paragraph("3. Reliability and Convergent Validity", H1))
    hdrs3 = ["Construct", "Cronbach alpha", "CR", "AVE",
             "alpha >= 0.70", "CR >= 0.70", "AVE >= 0.50"]
    hdr_row3 = [Paragraph(_sx(h), HDR) for h in hdrs3]
    rel_data = [hdr_row3]

    def _ok(v, thr):
        return "Pass" if (isinstance(v, float) and not np.isnan(v) and v >= thr) else "Fail"

    for name, vals in reliability.items():
        a   = vals.get("alpha", np.nan)
        cr  = vals.get("CR",    np.nan)
        ave = vals.get("AVE",   np.nan)
        rel_data.append([
            _cell(name, CELL),
            _cell(a, CELL), _cell(cr, CELL), _cell(ave, CELL),
            _status_cell(_ok(a, 0.70),  CELL),
            _status_cell(_ok(cr, 0.70), CELL),
            _status_cell(_ok(ave, 0.50), CELL),
        ])

    t = Table(rel_data,
              colWidths=[3.2*cm, 2.6*cm, 2.0*cm, 2.0*cm, 2.4*cm, 2.4*cm, 2.4*cm])
    t.setStyle(_tbl_style())
    story += [t, sp()]

    # ── 4. CFA Loadings ────────────────────────────────────────────────────────
    story.append(Paragraph("4. CFA Factor Loadings", H1))
    if not cfa_loadings.empty:
        ld = cfa_loadings.copy()
        if "p" in ld.columns:
            ld["Sig."] = ld["p"].apply(_sig_label)
        story += [
            _df_table(ld, status_col="Sig."),
            sp(0.2),
            Paragraph("* p&lt;.05  ** p&lt;.01  *** p&lt;.001", NOTE),
            sp(),
        ]

    # ── 5. CFA Fit ────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Measurement Model Fit (CFA)", H1))
    if cfa_fit:
        story += [fit_table(cfa_fit), sp()]

    # ── 6. Structural Path Coefficients ───────────────────────────────────────
    story.append(Paragraph("6. Structural Path Coefficients", H1))
    if not sem_paths.empty:
        pd_disp = sem_paths.copy()
        if "p" in pd_disp.columns:
            pd_disp["Sig."] = pd_disp["p"].apply(_sig_label)
        story += [
            _df_table(pd_disp, status_col="Sig."),
            sp(0.2),
            Paragraph("* p&lt;.05  ** p&lt;.01  *** p&lt;.001", NOTE),
            sp(),
        ]

    # ── 7. Structural Model Fit ────────────────────────────────────────────────
    story.append(Paragraph("7. Structural Model Fit", H1))
    if sem_fit:
        story += [fit_table(sem_fit), sp()]

    # ── 8. Structural Path Diagram ─────────────────────────────────────────────
    if roles and not sem_paths.empty:
        story.append(Paragraph("8. Structural Path Diagram", H1))
        try:
            png = _draw_diagram(roles, sem_paths)
            img = Image(io.BytesIO(png), width=17*cm, height=9*cm)
            story += [img, sp()]
        except Exception as e:
            story.append(Paragraph(f"Diagram unavailable: {_sx(str(e))}", NOTE))

    # ── 9. Indirect Effects ───────────────────────────────────────────────────
    if indirect_effects is not None and not indirect_effects.empty:
        story.append(Paragraph("9. Indirect Effects - Mediation (Bootstrap 2000 samples)", H1))
        story += [_df_table(indirect_effects, status_col="Significant"), sp()]

    # ── 10. Model Comparison ───────────────────────────────────────────────────
    if model_comparison is not None and not model_comparison.empty:
        story += [PageBreak()]
        story.append(Paragraph("10. Model Comparison (Permutations - ranked by AIC)", H1))
        story.append(Paragraph(
            "All permutations fitted. Lower AIC/BIC and higher CFI/TLI = better fit. "
            "Rank 1 is the recommended model.", BODY,
        ))
        story.append(sp(0.2))
        story += [_df_table(model_comparison), sp()]

    # ── Footer ────────────────────────────────────────────────────────────────
    story += [
        hr(),
        Paragraph(
            f"Generated by SEM Analysis Tool  |  {datetime.now().strftime('%Y-%m-%d')}",
            NOTE,
        ),
    ]

    doc.build(story)
