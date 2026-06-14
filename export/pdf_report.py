from datetime import datetime
from typing import Dict, Optional

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


def _safe(s) -> str:
    """Replace non-Latin-1 / XML-special chars for ReportLab Paragraph (Helvetica)."""
    return (
        str(s)
        .replace("χ²", "chi2").replace("χ", "chi").replace("²", "2")
        .replace("—", "-").replace("–", "-")   # em/en dash
        .replace("α", "alpha").replace("β", "b")
        .replace("√", "sqrt").replace("≥", ">=").replace("≤", "<=")
        .replace("→", "->").replace("×", "x").replace("±", "+/-")
        .replace("‘", "'").replace("’", "'")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _safe_plain(s) -> str:
    """Safe string for Table cells (plain drawString — no XML escaping needed)."""
    return (
        str(s)
        .replace("χ²", "chi2").replace("χ", "chi").replace("²", "2")
        .replace("—", "-").replace("–", "-")
        .replace("α", "alpha").replace("β", "b")
        .replace("√", "sqrt").replace("≥", ">=").replace("≤", "<=")
        .replace("→", "->").replace("×", "x").replace("±", "+/-")
    )


# ── Table helpers ─────────────────────────────────────────────────────────────

def _table_style(data, status_col: Optional[int] = None) -> TableStyle:
    cmds = [
        ("BACKGROUND",     (0, 0), (-1, 0),  _HEADER),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  9),
        ("FONTSIZE",       (0, 1), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
        ("PADDING",        (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_ODD, _EVEN]),
    ]
    if status_col is not None:
        for i, row in enumerate(data[1:], start=1):
            val = str(row[status_col]).strip()
            if val in ("Good", "Pass", "Yes", "***"):
                cmds += [("TEXTCOLOR", (status_col, i), (status_col, i), _GOOD),
                         ("FONTNAME",  (status_col, i), (status_col, i), "Helvetica-Bold")]
            elif val in ("Acceptable", "**", "*"):
                cmds += [("TEXTCOLOR", (status_col, i), (status_col, i), _ACCEPT),
                         ("FONTNAME",  (status_col, i), (status_col, i), "Helvetica-Bold")]
            elif val in ("Poor", "Fail", "Review", "No", "n.s."):
                cmds += [("TEXTCOLOR", (status_col, i), (status_col, i), _POOR),
                         ("FONTNAME",  (status_col, i), (status_col, i), "Helvetica-Bold")]
    return TableStyle(cmds)


def _df_to_table(df: pd.DataFrame, status_col_name: Optional[str] = None,
                 col_widths=None):
    headers = [_safe_plain(h) for h in df.columns]
    rows    = [[_safe_plain(v) for v in row] for _, row in df.iterrows()]
    data    = [headers] + rows

    safe_status = _safe_plain(status_col_name) if status_col_name else None
    status_col  = headers.index(safe_status) if safe_status in headers else None

    if col_widths is None:
        col_widths = [17 * cm / len(headers)] * len(headers)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(_table_style(data, status_col))
    return t


def _sig_label(p) -> str:
    try:
        v = float(p)
        if v < 0.001: return "***"
        if v < 0.01:  return "**"
        if v < 0.05:  return "*"
        return "n.s."
    except (TypeError, ValueError):
        return "-"


def _fit_status(key: str, val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "-"
    safe_k = _safe_plain(key)
    rules = {
        "CFI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
        "TLI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
        "RMSEA": lambda x: "Good" if x <= 0.06 else ("Acceptable" if x <= 0.08 else "Poor"),
        "SRMR":  lambda x: "Good" if x <= 0.08 else ("Acceptable" if x <= 0.10 else "Poor"),
        "GFI":   lambda x: "Good" if x >= 0.95 else ("Acceptable" if x >= 0.90 else "Poor"),
    }
    return rules[safe_k](v) if safe_k in rules else "-"


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf(
    filepath: str,
    model_name: str,
    constructs: Dict[str, str],
    reliability: Dict[str, Dict],
    normality_df: pd.DataFrame,
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
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    H1   = ParagraphStyle("H1",  parent=styles["Heading1"], fontSize=13,
                          spaceAfter=5, textColor=_HEADER)
    H2   = ParagraphStyle("H2",  parent=styles["Heading2"], fontSize=10,
                          spaceAfter=4, textColor=_HEADER)
    BODY = ParagraphStyle("BODY",parent=styles["Normal"],   fontSize=8,
                          spaceAfter=4)
    TTL  = ParagraphStyle("TTL", parent=styles["Title"],    fontSize=20,
                          spaceAfter=4)
    NOTE = ParagraphStyle("NOTE",parent=styles["Normal"],   fontSize=7,
                          textColor=colors.HexColor("#666"))

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#bdc3c7"), spaceAfter=6)

    def sp(h=0.3):
        return Spacer(1, h * cm)

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story += [
        sp(1),
        Paragraph("SEM Analysis Report", TTL),
        Paragraph(f"Model: {_safe(model_name)}", H1),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", BODY),
        sp(0.4), hr(), sp(0.2),
    ]

    # ── 1. Constructs ──────────────────────────────────────────────────────────
    story.append(Paragraph("1. Latent Constructs and Roles", H1))
    role_data = [["Construct", "Role"]] + [
        [_safe_plain(k), _safe_plain(v)] for k, v in constructs.items()
    ]
    t = Table(role_data, colWidths=[5*cm, 12*cm])
    t.setStyle(_table_style(role_data))
    story += [t, sp()]

    # ── 2. Normality ───────────────────────────────────────────────────────────
    story.append(Paragraph("2. Normality Assessment", H1))
    story.append(Paragraph(
        "WLS estimator is used throughout - robust to non-normal and ordinal "
        "Likert-scale data. Normality is assessed for reporting purposes only.",
        BODY,
    ))
    story.append(sp(0.2))

    if not normality_df.empty:
        story.append(Paragraph(
            "Univariate Normality  (|Skew| &lt; 2 and |Kurtosis| &lt; 7 = Normal)", H2))
        nd = normality_df.rename(columns=lambda c: _safe_plain(c))
        story += [_df_to_table(nd, status_col_name="Normal?"), sp(0.2)]

    if mardia and "Error" not in mardia:
        mv    = _safe_plain(mardia.get("Multivariate Normal?", "-"))
        sk_ch = _safe_plain(mardia.get("Skewness chi2",
                            mardia.get("Skewness χ²", "-")))
        sk_p  = _safe_plain(mardia.get("Skewness p-value", "-"))
        ku_z  = _safe_plain(mardia.get("Kurtosis Z", "-"))
        ku_p  = _safe_plain(mardia.get("Kurtosis p-value", "-"))
        story.append(Paragraph(
            f"Mardia Multivariate Normality: <b>{_safe(mv)}</b>  "
            f"(Skewness chi2={sk_ch}, p={sk_p};  Kurtosis Z={ku_z}, p={ku_p})",
            BODY,
        ))
    story.append(sp())

    # ── 3. Reliability ─────────────────────────────────────────────────────────
    story.append(Paragraph("3. Reliability and Convergent Validity", H1))
    rel_rows = [["Construct", "Cronbach alpha", "CR", "AVE",
                 "alpha >= 0.70", "CR >= 0.70", "AVE >= 0.50"]]

    def _fmt(v):
        return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "-"

    def _ok(v, thr):
        return "Pass" if (isinstance(v, float) and not np.isnan(v) and v >= thr) else "Fail"

    for name, vals in reliability.items():
        a   = vals.get("alpha", np.nan)
        cr  = vals.get("CR",    np.nan)
        ave = vals.get("AVE",   np.nan)
        rel_rows.append([
            _safe_plain(name), _fmt(a), _fmt(cr), _fmt(ave),
            _ok(a, 0.70), _ok(cr, 0.70), _ok(ave, 0.50),
        ])

    cw_rel = [3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.8*cm, 2.8*cm, 2.8*cm]
    t = Table(rel_rows, colWidths=cw_rel)
    t.setStyle(_table_style(rel_rows, status_col=4))
    story += [t, sp()]

    # ── 4. CFA Loadings ────────────────────────────────────────────────────────
    story.append(Paragraph("4. CFA Factor Loadings", H1))
    if not cfa_loadings.empty:
        ld = cfa_loadings.copy()
        if "p" in ld.columns:
            ld["Sig."] = ld["p"].apply(_sig_label)
        ld.columns = [_safe_plain(c) for c in ld.columns]
        story += [
            _df_to_table(ld),
            sp(0.2),
            Paragraph("* p&lt;.05  ** p&lt;.01  *** p&lt;.001", NOTE),
            sp(),
        ]

    # ── 5. CFA Fit ────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Measurement Model Fit (CFA)", H1))
    benchmarks = {
        "chi2":    "-",
        "df":      "-",
        "p(chi2)": "-",
        "CFI":     ">= 0.95 good / >= 0.90 acceptable",
        "TLI":     ">= 0.95 good / >= 0.90 acceptable",
        "RMSEA":   "<= 0.06 good / <= 0.08 acceptable",
        "SRMR":    "<= 0.08 good / <= 0.10 acceptable",
        "GFI":     ">= 0.95 good / >= 0.90 acceptable",
        "AIC":     "Lower is better",
        "BIC":     "Lower is better",
    }
    cfa_rows = [["Index", "Value", "Benchmark", "Status"]]
    for k, v in cfa_fit.items():
        sk = _safe_plain(k)
        cfa_rows.append([sk, _safe_plain(v), benchmarks.get(sk, "-"), _fit_status(k, v)])
    t = Table(cfa_rows, colWidths=[2.8*cm, 3*cm, 7.7*cm, 3.5*cm])
    t.setStyle(_table_style(cfa_rows, status_col=3))
    story += [t, sp()]

    # ── 6. SEM Paths ──────────────────────────────────────────────────────────
    story.append(Paragraph("6. Structural Path Coefficients", H1))
    if not sem_paths.empty:
        pd_disp = sem_paths.copy()
        if "p" in pd_disp.columns:
            pd_disp["Sig."] = pd_disp["p"].apply(_sig_label)
        pd_disp.columns = [_safe_plain(c) for c in pd_disp.columns]
        story += [
            _df_to_table(pd_disp, status_col_name="Sig."),
            sp(0.2),
            Paragraph("* p&lt;.05  ** p&lt;.01  *** p&lt;.001", NOTE),
            sp(),
        ]

    # ── 7. SEM Fit ────────────────────────────────────────────────────────────
    story.append(Paragraph("7. Structural Model Fit", H1))
    sem_rows = [["Index", "Value", "Benchmark", "Status"]]
    for k, v in sem_fit.items():
        sk = _safe_plain(k)
        sem_rows.append([sk, _safe_plain(v), benchmarks.get(sk, "-"), _fit_status(k, v)])
    t = Table(sem_rows, colWidths=[2.8*cm, 3*cm, 7.7*cm, 3.5*cm])
    t.setStyle(_table_style(sem_rows, status_col=3))
    story += [t, sp()]

    # ── 8. Indirect Effects ───────────────────────────────────────────────────
    if indirect_effects is not None and not indirect_effects.empty:
        story.append(Paragraph(
            "8. Indirect Effects - Mediation (Bootstrap 2000 samples)", H1))
        ie = indirect_effects.copy()
        ie.columns = [_safe_plain(c) for c in ie.columns]
        story += [_df_to_table(ie, status_col_name="Significant"), sp()]

    # ── 9. Model Comparison ───────────────────────────────────────────────────
    if model_comparison is not None and not model_comparison.empty:
        story.append(PageBreak())
        story.append(Paragraph("9. Model Comparison (Permutations - ranked by AIC)", H1))
        story.append(Paragraph(
            "All permutations fitted. Lower AIC/BIC and higher CFI/TLI = better fit. "
            "Rank 1 is the recommended model.",
            BODY,
        ))
        story.append(sp(0.2))
        mc = model_comparison.copy()
        mc.columns = [_safe_plain(c) for c in mc.columns]
        story += [_df_to_table(mc), sp()]

    # ── Footer ────────────────────────────────────────────────────────────────
    story += [
        hr(),
        Paragraph(
            f"Generated by SEM Analysis Tool  |  "
            f"{datetime.now().strftime('%Y-%m-%d')}",
            NOTE,
        ),
    ]

    doc.build(story)
