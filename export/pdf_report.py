from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

# ── Colour palette ────────────────────────────────────────────────────────────
_HEADER = colors.HexColor("#2c3e50")
_EVEN   = colors.HexColor("#f4f6f7")
_ODD    = colors.white
_GOOD   = colors.HexColor("#1e8449")
_ACCEPT = colors.HexColor("#d4ac0d")
_POOR   = colors.HexColor("#c0392b")
_PASS   = colors.HexColor("#1e8449")
_REVIEW = colors.HexColor("#e67e22")


def _table_style(data, status_col: Optional[int] = None) -> TableStyle:
    cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0),  _HEADER),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
        ("PADDING",     (0, 0), (-1, -1), 5),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(data)):
        bg = _EVEN if i % 2 == 0 else _ODD
        cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

    if status_col is not None:
        for i, row in enumerate(data[1:], start=1):
            val = str(row[status_col]).strip()
            if val in ("Good", "Pass", "Yes"):
                cmds.append(("TEXTCOLOR", (status_col, i), (status_col, i), _PASS))
                cmds.append(("FONTNAME",  (status_col, i), (status_col, i), "Helvetica-Bold"))
            elif val in ("Acceptable",):
                cmds.append(("TEXTCOLOR", (status_col, i), (status_col, i), _ACCEPT))
                cmds.append(("FONTNAME",  (status_col, i), (status_col, i), "Helvetica-Bold"))
            elif val in ("Poor", "Review", "No"):
                cmds.append(("TEXTCOLOR", (status_col, i), (status_col, i), _POOR))
                cmds.append(("FONTNAME",  (status_col, i), (status_col, i), "Helvetica-Bold"))
    return TableStyle(cmds)


def _df_to_table(df: pd.DataFrame, status_col_name: Optional[str] = None):
    headers = list(df.columns)
    status_col = headers.index(status_col_name) if status_col_name in headers else None
    data = [headers] + [[str(v) for v in row] for _, row in df.iterrows()]
    n_cols = len(headers)
    col_w = 17 * cm / n_cols
    t = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
    t.setStyle(_table_style(data, status_col))
    return t


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _fit_status(key: str, val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    rules = {
        "CFI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
        "TLI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
        "RMSEA": lambda x: "Good" if x <= 0.06 else ("Acceptable" if x <= 0.08 else "Poor"),
        "SRMR":  lambda x: "Good" if x <= 0.08 else ("Acceptable" if x <= 0.10 else "Poor"),
        "GFI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
    }
    return rules[key](v) if key in rules else "—"


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf(
    filepath: str,
    model_name: str,
    constructs: Dict[str, str],           # {name: role}
    reliability: Dict[str, Dict],         # {name: {alpha, CR, AVE}}
    normality_df: pd.DataFrame,           # from test_normality_items
    mardia: Dict,
    cfa_loadings: pd.DataFrame,
    cfa_fit: Dict,
    sem_paths: pd.DataFrame,
    sem_fit: Dict,
    indirect_effects: Optional[pd.DataFrame] = None,
    model_comparison: Optional[pd.DataFrame] = None,
) -> None:
    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=6, textColor=_HEADER)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceAfter=4, textColor=_HEADER)
    BODY = styles["Normal"]
    TITLE_STYLE = ParagraphStyle("TITLE", parent=styles["Title"], fontSize=20, spaceAfter=4)
    NOTE = ParagraphStyle("NOTE", parent=BODY, fontSize=8, textColor=colors.HexColor("#666"))

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bdc3c7"), spaceAfter=8)

    def sp(h=0.3):
        return Spacer(1, h * cm)

    story = []

    # ── Cover ──
    story += [
        sp(1),
        Paragraph("SEM Analysis Report", TITLE_STYLE),
        Paragraph(f"<b>Model:</b> {model_name}", H1),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", BODY),
        sp(0.4), hr(), sp(0.3),
    ]

    # ── 1. Constructs ──
    story += [Paragraph("1. Latent Constructs & Roles", H1)]
    role_data = [["Construct", "Role"]] + [[k, v] for k, v in constructs.items()]
    t = Table(role_data, colWidths=[6 * cm, 11 * cm])
    t.setStyle(_table_style(role_data))
    story += [t, sp()]

    # ── 2. Normality ──
    story += [Paragraph("2. Normality Assessment", H1)]
    story += [Paragraph(
        "WLS estimator is used throughout — robust to non-normal and ordinal Likert-scale data. "
        "Normality is assessed for reporting purposes only.",
        BODY,
    ), sp(0.2)]

    if not normality_df.empty:
        story += [Paragraph("Univariate Normality (|Skew| < 2, |Kurtosis| < 7 = Normal)", H2)]
        story += [_df_to_table(normality_df, status_col_name="Normal?"), sp(0.2)]

    if mardia:
        mv_normal = mardia.get("Multivariate Normal?", "—")
        story += [Paragraph(
            f"<b>Mardia's Multivariate Normality:</b> {mv_normal}  "
            f"(Skewness χ²={mardia.get('Skewness χ²','—')}, "
            f"p={mardia.get('Skewness p-value','—')}; "
            f"Kurtosis Z={mardia.get('Kurtosis Z','—')}, "
            f"p={mardia.get('Kurtosis p-value','—')})",
            BODY,
        ), sp()]

    # ── 3. Reliability ──
    story += [Paragraph("3. Reliability & Convergent Validity", H1)]
    rel_rows = [["Construct", "Cronbach α", "CR", "AVE", "α ≥ 0.70", "CR ≥ 0.70", "AVE ≥ 0.50"]]
    for name, vals in reliability.items():
        a = vals.get("alpha", np.nan)
        cr = vals.get("CR", np.nan)
        ave = vals.get("AVE", np.nan)

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "—"

        def ok(v, thr):
            return "Pass" if isinstance(v, float) and not np.isnan(v) and v >= thr else "Fail"

        rel_rows.append([name, fmt(a), fmt(cr), fmt(ave), ok(a, 0.70), ok(cr, 0.70), ok(ave, 0.50)])

    t = Table(rel_rows, colWidths=[3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
    t.setStyle(_table_style(rel_rows, status_col=4))
    story += [t, sp()]

    # ── 4. CFA Loadings ──
    story += [Paragraph("4. CFA Factor Loadings", H1)]
    if not cfa_loadings.empty:
        disp = cfa_loadings.copy()
        if "p" in disp.columns:
            disp["Sig."] = disp["p"].apply(lambda x: _sig_label(float(x)) if x not in ("—", None) else "—")
        story += [_df_to_table(disp), sp(0.2)]
        story += [Paragraph("<i>* p&lt;.05, ** p&lt;.01, *** p&lt;.001</i>", NOTE), sp()]

    # ── 5. CFA Fit ──
    story += [Paragraph("5. Measurement Model Fit (CFA)", H1)]
    thresholds = {
        "CFI": "≥ 0.95 good / ≥ 0.90 acceptable",
        "TLI": "≥ 0.95 good / ≥ 0.90 acceptable",
        "RMSEA": "≤ 0.06 good / ≤ 0.08 acceptable",
        "SRMR": "≤ 0.08 good / ≤ 0.10 acceptable",
        "GFI": "≥ 0.95 good / ≥ 0.90 acceptable",
        "χ²": "—",
        "df": "—",
        "p(χ²)": "—",
        "AIC": "Lower is better",
        "BIC": "Lower is better",
    }
    fit_rows = [["Index", "Value", "Benchmark", "Status"]]
    for k, v in cfa_fit.items():
        status = _fit_status(k, v)
        fit_rows.append([k, str(v), thresholds.get(k, "—"), status])
    t = Table(fit_rows, colWidths=[3 * cm, 3.5 * cm, 7 * cm, 3.5 * cm])
    t.setStyle(_table_style(fit_rows, status_col=3))
    story += [t, sp()]

    # ── 6. SEM Paths ──
    story += [Paragraph("6. Structural Path Coefficients", H1)]
    if not sem_paths.empty:
        disp = sem_paths.copy()
        if "p" in disp.columns:
            disp["Sig."] = disp["p"].apply(lambda x: _sig_label(float(x)))
        story += [_df_to_table(disp, status_col_name="Sig."), sp(0.2)]
        story += [Paragraph("<i>* p&lt;.05, ** p&lt;.01, *** p&lt;.001</i>", NOTE), sp()]

    # ── 7. SEM Fit ──
    story += [Paragraph("7. Structural Model Fit", H1)]
    sem_fit_rows = [["Index", "Value", "Benchmark", "Status"]]
    for k, v in sem_fit.items():
        status = _fit_status(k, v)
        sem_fit_rows.append([k, str(v), thresholds.get(k, "—"), status])
    t = Table(sem_fit_rows, colWidths=[3 * cm, 3.5 * cm, 7 * cm, 3.5 * cm])
    t.setStyle(_table_style(sem_fit_rows, status_col=3))
    story += [t, sp()]

    # ── 8. Indirect Effects ──
    if indirect_effects is not None and not indirect_effects.empty:
        story += [Paragraph("8. Indirect Effects — Mediation (Bootstrap 2000 samples)", H1)]
        story += [_df_to_table(indirect_effects, status_col_name="Significant"), sp()]

    # ── 9. Model Comparison ──
    if model_comparison is not None and not model_comparison.empty:
        story += [PageBreak(), Paragraph("9. Model Comparison", H1)]
        story += [Paragraph(
            "All permutations fitted; lower AIC/BIC and higher CFI/TLI indicate better fit. "
            "Rank 1 is the recommended model.",
            BODY,
        ), sp(0.2)]
        story += [_df_to_table(model_comparison), sp()]

    # ── Footer ──
    story += [
        hr(),
        Paragraph(
            f"<i>Generated by SEM Analysis Tool &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d')}</i>",
            NOTE,
        ),
    ]

    doc.build(story)
