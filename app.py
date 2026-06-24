"""
SEM Analysis Tool — Streamlit App
Run: streamlit run app.py
"""

import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from core.cfa import draw_path_diagram, fit_status, run_cfa, get_construct_correlations
from core.mediation import run_all_indirect
from core.moderation import build_mean_scores, create_interaction, simple_slopes
from core.parser import (
    all_items, detect_clusters, generate_sample_data, get_missing_summary,
    validate_likert_range,
)
from core.permutations import generate_permutations
from core.sem import build_sem_syntax, draw_structural_diagram, run_sem
from core.validity import (
    compute_ave_cr, cronbach_alpha, fornell_larcker, htmt_ratio,
    mardia_test, test_normality_items,
)
from export.pdf_report import generate_pdf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEM Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "page": 0,
    "df": None,
    "clusters": {},
    "likert_max": 7,
    "dropped_items": set(),
    "normality_df": None,
    "mardia": {},
    "reliability": {},
    "ave_dict": {},
    "cfa_model": None,
    "cfa_loadings": pd.DataFrame(),
    "cfa_fit": {},
    "construct_corr": pd.DataFrame(),
    "model_name": "Model 1",
    "selected_constructs": [],
    "roles": {},
    "moderators": [],
    "mediators": [],
    "exogenous": [],
    "endogenous": [],
    "sem_model": None,
    "sem_paths": pd.DataFrame(),
    "sem_fit": {},
    "indirect_effects": pd.DataFrame(),
    "model_comparison": pd.DataFrame(),
    "slope_df": pd.DataFrame(),
    "interaction_fig": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar navigation ────────────────────────────────────────────────────────
STEPS = [
    "1 · Upload & Validate",
    "2 · Normality & Reliability",
    "3 · Confirmatory Factor Analysis",
    "4 · Model Configuration",
    "5 · Results & Export",
]
ICONS = ["📂", "📋", "🔬", "⚙️", "📊"]

with st.sidebar:
    st.title("📊 SEM Tool")
    st.markdown("---")
    for i, (icon, label) in enumerate(zip(ICONS, STEPS)):
        active = st.session_state.page == i
        style = "background:#2c3e50;color:white;border-radius:6px;padding:6px 10px;" if active \
                else "color:#555;padding:6px 10px;"
        btn_label = f"{icon} **{label}**" if active else f"{icon} {label}"
        if st.button(btn_label, key=f"nav_{i}", use_container_width=True):
            st.session_state.page = i
            st.rerun()
    st.markdown("---")
    st.caption("Estimator: **WLS** (robust for Likert/ordinal data)")


def go_to(page: int):
    st.session_state.page = page
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Upload & Validate
# ═══════════════════════════════════════════════════════════════════════════════
def page_upload():
    st.header("📂 Upload & Validate Data")

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload CSV or Excel (columns: C1.1, C1.2 … C2.1, C2.2 …)",
            type=["csv", "xlsx", "xls"],
        )
    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Load sample data (5 clusters, 4 items)"):
            st.session_state.df = generate_sample_data()
            st.session_state.clusters = detect_clusters(st.session_state.df)

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            st.session_state.clusters = detect_clusters(df)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

    df = st.session_state.df
    if df is None:
        st.info("Upload a file or load the sample data to begin.")
        return

    clusters = st.session_state.clusters
    if not clusters:
        st.warning(
            "No clusters detected. Ensure columns are named C1.1, C1.2, C2.1 etc."
        )
        st.dataframe(df.head())
        return

    # ── Likert scale range ──
    st.subheader("Likert Scale Settings")
    c1, c2 = st.columns(2)
    with c1:
        likert_min = st.number_input("Min value", value=1, min_value=1, max_value=3)
    with c2:
        likert_max = st.number_input("Max value", value=7, min_value=3, max_value=10)
    st.session_state.likert_max = int(likert_max)

    items = all_items(clusters)

    # ── Detected clusters ──
    st.subheader("Detected Clusters")
    summary_rows = [
        {"Cluster": c, "Items": ", ".join(v), "N Items": len(v)}
        for c, v in clusters.items()
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Missing values ──
    st.subheader("Missing Values")
    miss = get_missing_summary(df, items)
    total_missing = int(miss["Missing N"].sum())
    if total_missing == 0:
        st.success("No missing values.")
    else:
        st.warning(f"{total_missing} missing cell(s) detected.")
        st.dataframe(miss[miss["Missing N"] > 0], use_container_width=True)

    # ── Likert range check ──
    st.subheader("Likert Range Check")
    out_of_range = validate_likert_range(df, items, int(likert_min), int(likert_max))
    if not out_of_range:
        st.success(f"All values within [{int(likert_min)}, {int(likert_max)}].")
    else:
        for col, n in out_of_range.items():
            st.warning(f"{col}: {n} value(s) outside range.")

    # ── Data preview ──
    with st.expander("Data preview (first 20 rows)"):
        st.dataframe(df[items].head(20), use_container_width=True)

    st.divider()
    if st.button("▶ Proceed to Normality & Reliability", type="primary"):
        go_to(1)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Normality & Reliability
# ═══════════════════════════════════════════════════════════════════════════════
def page_reliability():
    st.header("📋 Normality & Reliability")
    df = st.session_state.df
    clusters = st.session_state.clusters
    if df is None or not clusters:
        st.warning("Please upload data first.")
        return

    # ── Item exclusion ──
    st.subheader("Item Management")
    all_it = all_items(clusters)
    dropped = st.multiselect(
        "Drop items before analysis (e.g., low inter-item correlation)",
        options=all_it,
        default=list(st.session_state.dropped_items),
    )
    st.session_state.dropped_items = set(dropped)
    active_clusters = {
        c: [i for i in items if i not in dropped]
        for c, items in clusters.items()
        if [i for i in items if i not in dropped]
    }

    # ── Normality ──
    st.subheader("Univariate Normality")
    active_items = all_items(active_clusters)
    norm_df = test_normality_items(df, active_items)
    st.session_state.normality_df = norm_df

    def colour_normal(val):
        if val == "Yes":
            return "color: #1e8449; font-weight:bold"
        if val == "No":
            return "color: #c0392b; font-weight:bold"
        return ""

    st.dataframe(
        norm_df.style.map(colour_normal, subset=["Normal?"]),
        use_container_width=True, hide_index=True,
    )
    n_non_normal = (norm_df["Normal?"] == "No").sum()
    if n_non_normal > 0:
        st.info(
            f"{n_non_normal} item(s) show non-normal distributions (|skew|≥2 or |kurtosis|≥7). "
            "WLS estimator will be used — robust to non-normality."
        )

    # Mardia
    st.subheader("Mardia's Multivariate Normality")
    with st.spinner("Computing Mardia's test…"):
        mardia = mardia_test(df, active_items)
    st.session_state.mardia = mardia
    if "Error" in mardia:
        st.error(mardia["Error"])
    else:
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("MV Skewness χ²", mardia["Mardia Skewness (b1p)"])
            st.metric("Skewness p", mardia["Skewness p-value"])
        with m_col2:
            st.metric("MV Kurtosis Z", mardia["Mardia Kurtosis (b2p)"])
            st.metric("Kurtosis p", mardia["Kurtosis p-value"])
        mv = mardia.get("Multivariate Normal?", "—")
        if mv == "Yes":
            st.success("Data are multivariate normal.")
        else:
            st.warning("Data are NOT multivariate normal → WLS is the appropriate estimator.")

    # ── Reliability per cluster ──
    st.subheader("Reliability per Cluster")
    reliability = {}
    for construct, items in active_clusters.items():
        if len(items) < 2:
            st.warning(f"{construct}: fewer than 2 items — skipped.")
            continue
        alpha, item_df = cronbach_alpha(df, items)
        reliability[construct] = {"alpha": alpha, "item_df": item_df}

        with st.expander(
            f"{construct}  —  α = {alpha:.3f}  "
            f"{'✅' if alpha >= 0.70 else '⚠️ Below threshold (0.70)'}"
        ):
            def colour_keep(val):
                return "color:#1e8449;font-weight:bold" if val == "Yes" else "color:#c0392b"
            st.dataframe(
                item_df.style.applymap(colour_keep, subset=["Keep?"]),
                use_container_width=True, hide_index=True,
            )

    st.session_state.reliability = reliability

    # ── HTMT ──
    st.subheader("HTMT Discriminant Validity (threshold < 0.85)")
    if len(active_clusters) >= 2:
        htmt_df = htmt_ratio(df, active_clusters)
        def colour_htmt(val):
            try:
                v = float(val)
                return "color:#c0392b;font-weight:bold" if v >= 0.85 else ""
            except (TypeError, ValueError):
                return ""
        st.dataframe(
            htmt_df.style.applymap(colour_htmt),
            use_container_width=True,
        )
    else:
        st.info("Need ≥ 2 constructs for HTMT.")

    st.divider()
    if st.button("▶ Proceed to CFA", type="primary"):
        go_to(2)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CFA
# ═══════════════════════════════════════════════════════════════════════════════
def page_cfa():
    st.header("🔬 Confirmatory Factor Analysis")
    df = st.session_state.df
    clusters = st.session_state.clusters
    dropped = st.session_state.dropped_items
    if df is None or not clusters:
        st.warning("Please upload data first.")
        return

    active_clusters = {
        c: [i for i in items if i not in dropped]
        for c, items in clusters.items()
        if [i for i in items if i not in dropped]
    }

    st.info("CFA uses WLS — appropriate for 5–7 point Likert scale data.")

    run_btn = st.button("▶ Run CFA", type="primary")

    if run_btn or (not st.session_state.cfa_loadings.empty):
        if run_btn:
            with st.spinner("Fitting CFA model…"):
                model, loadings, fit, warning = run_cfa(df, active_clusters, estimator="WLS")
            st.session_state.cfa_model = model
            st.session_state.cfa_loadings = loadings
            st.session_state.cfa_fit = fit
            if warning:
                st.warning(f"Estimator fallback: {warning}")

            # Compute AVE / CR from std loadings
            std_col = "Std. Loading" if "Std. Loading" in loadings.columns else None
            ave_dict, cr_dict = {}, {}
            if std_col:
                for construct in active_clusters:
                    mask = loadings["Construct"] == construct
                    l = loadings.loc[mask, std_col].values.astype(float)
                    if len(l):
                        ave, cr = compute_ave_cr(l)
                        ave_dict[construct] = ave
                        cr_dict[construct] = cr
                        if construct in st.session_state.reliability:
                            st.session_state.reliability[construct]["AVE"] = ave
                            st.session_state.reliability[construct]["CR"] = cr

            st.session_state.ave_dict = ave_dict

            # Construct correlations
            if model is not None:
                constructs = list(active_clusters.keys())
                corr = get_construct_correlations(model, constructs)
                st.session_state.construct_corr = corr

        loadings = st.session_state.cfa_loadings
        fit = st.session_state.cfa_fit

        # ── Loadings table ──
        st.subheader("Factor Loadings")
        if not loadings.empty:
            std_col = "Std. Loading" if "Std. Loading" in loadings.columns else None

            def colour_loading(val):
                try:
                    v = abs(float(val))
                    if v >= 0.70:
                        return "color:#1e8449;font-weight:bold"
                    if v >= 0.40:
                        return "color:#d4ac0d"
                    return "color:#c0392b"
                except (TypeError, ValueError):
                    return ""

            style = loadings.style
            if std_col:
                style = style.applymap(colour_loading, subset=[std_col])
            st.dataframe(style, use_container_width=True, hide_index=True)
            st.caption("Green ≥ 0.70 · Amber 0.40–0.70 · Red < 0.40 (consider dropping)")
        else:
            st.warning("No loading results — check that semopy is installed.")

        # ── AVE / CR ──
        ave_dict = st.session_state.ave_dict
        rel = st.session_state.reliability
        if ave_dict:
            st.subheader("AVE & Composite Reliability")
            ave_rows = []
            for c, ave in ave_dict.items():
                cr = rel.get(c, {}).get("CR", np.nan)
                ave_rows.append({
                    "Construct": c,
                    "AVE": round(ave, 3),
                    "AVE ≥ 0.50": "Pass" if ave >= 0.50 else "Fail",
                    "CR": round(cr, 3) if not np.isnan(cr) else "—",
                    "CR ≥ 0.70": "Pass" if (not np.isnan(cr) and cr >= 0.70) else "Fail",
                })
            ave_df = pd.DataFrame(ave_rows)
            def colour_pass(val):
                if val == "Pass":
                    return "color:#1e8449;font-weight:bold"
                if val == "Fail":
                    return "color:#c0392b;font-weight:bold"
                return ""
            st.dataframe(
                ave_df.style.applymap(colour_pass, subset=["AVE ≥ 0.50", "CR ≥ 0.70"]),
                use_container_width=True, hide_index=True,
            )

        # ── Fornell-Larcker ──
        cc = st.session_state.construct_corr
        if ave_dict and not cc.empty:
            st.subheader("Fornell-Larcker Criterion (√AVE on diagonal > off-diagonal correlations)")
            fl_df = fornell_larcker(ave_dict, cc)
            st.dataframe(fl_df.set_index("Construct"), use_container_width=True)

        # ── Fit indices ──
        st.subheader("Model Fit Indices")
        if fit:
            status = fit_status(fit)
            fit_rows = []
            benchmarks = {
                "CFI": "≥ 0.95 good / ≥ 0.90 acceptable",
                "TLI": "≥ 0.95 good / ≥ 0.90 acceptable",
                "RMSEA": "≤ 0.06 good / ≤ 0.08 acceptable",
                "SRMR": "≤ 0.08 good / ≤ 0.10 acceptable",
                "GFI": "≥ 0.95 good / ≥ 0.90 acceptable",
                "χ²": "—", "df": "—", "p(χ²)": "—",
                "AIC": "Lower is better", "BIC": "Lower is better",
            }
            for k, v in fit.items():
                fit_rows.append({
                    "Index": k, "Value": v,
                    "Benchmark": benchmarks.get(k, "—"),
                    "Status": status.get(k, "—"),
                })
            fit_df = pd.DataFrame(fit_rows)

            def colour_status(val):
                return {
                    "Good": "color:#1e8449;font-weight:bold",
                    "Acceptable": "color:#d4ac0d;font-weight:bold",
                    "Poor": "color:#c0392b;font-weight:bold",
                }.get(str(val), "")

            st.dataframe(
                fit_df.style.applymap(colour_status, subset=["Status"]),
                use_container_width=True, hide_index=True,
            )

        # ── Path diagram ──
        st.subheader("Measurement Model Diagram")
        fig = draw_path_diagram(active_clusters, loadings)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    if st.button("▶ Proceed to Model Configuration", type="primary"):
        go_to(3)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Configuration
# ═══════════════════════════════════════════════════════════════════════════════
def page_model_config():
    st.header("⚙️ SEM Model Configuration")
    df = st.session_state.df
    clusters = st.session_state.clusters
    dropped = st.session_state.dropped_items
    if df is None or not clusters:
        st.warning("Please upload data first.")
        return

    active_constructs = [
        c for c, items in clusters.items()
        if [i for i in items if i not in dropped]
    ]

    # ── Model name ──
    st.session_state.model_name = st.text_input(
        "Model Name", value=st.session_state.model_name
    )

    # ── Select constructs (max 10: 5 IV + 2 med + 2 mod + 1 DV) ──
    st.subheader("Select Latent Constructs (max 10)")
    selected = st.multiselect(
        "Which constructs to include in this model?",
        options=active_constructs,
        default=st.session_state.selected_constructs or active_constructs[:6],
        max_selections=10,
    )
    st.session_state.selected_constructs = selected

    if len(selected) < 2:
        st.info("Select at least 2 constructs to configure roles.")
        return

    # ── Role assignment ──
    st.subheader("Assign Roles")
    st.caption(
        "Exogenous (IV): max 5  ·  Endogenous (DV): max 1  ·  "
        "Mediators: max 2  ·  Moderators: max 2"
    )

    ROLES = ["Exogenous (IV)", "Endogenous (DV)", "Mediator", "Moderator"]
    roles = {}
    exo_count = med_count = mod_count = end_count = 0

    cols = st.columns(min(len(selected), 5))
    for i, construct in enumerate(selected):
        with cols[i % min(len(selected), 5)]:
            prev_role = st.session_state.roles.get(construct, ROLES[0])
            role = st.selectbox(
                construct,
                options=ROLES,
                index=ROLES.index(prev_role) if prev_role in ROLES else 0,
                key=f"role_{construct}",
            )
            roles[construct] = role
            if role == "Exogenous (IV)":
                exo_count += 1
            elif role == "Endogenous (DV)":
                end_count += 1
            elif role == "Mediator":
                med_count += 1
            elif role == "Moderator":
                mod_count += 1

    # Enforce limits with clear messages
    errors = []
    if exo_count > 5:
        errors.append(f"Maximum 5 Exogenous (IV) allowed — currently {exo_count}.")
    if end_count > 1:
        errors.append(f"Maximum 1 Endogenous (DV) allowed — currently {end_count}.")
    if med_count > 2:
        errors.append(f"Maximum 2 Mediators allowed — currently {med_count}.")
    if mod_count > 2:
        errors.append(f"Maximum 2 Moderators allowed — currently {mod_count}.")
    if errors:
        for msg in errors:
            st.error(msg)
        return

    st.session_state.roles = roles
    exogenous  = [c for c, r in roles.items() if r == "Exogenous (IV)"]
    endogenous = [c for c, r in roles.items() if r == "Endogenous (DV)"]
    mediators  = [c for c, r in roles.items() if r == "Mediator"]
    moderators = [c for c, r in roles.items() if r == "Moderator"]

    st.session_state.exogenous  = exogenous
    st.session_state.endogenous = endogenous
    st.session_state.mediators  = mediators
    st.session_state.moderators = moderators

    if not exogenous:
        st.warning("Assign at least one Exogenous (IV) construct.")
        return
    if not endogenous:
        st.warning("Assign exactly one Endogenous (DV) construct.")
        return

    # ── Model preview ──
    st.subheader("Model Preview")
    preview_lines = []
    for iv in exogenous:
        for dv in endogenous:
            preview_lines.append(f"{iv} → {dv}")
        for med in mediators:
            preview_lines.append(f"{iv} → {med} → {' + '.join(endogenous)}")
    for mod in moderators:
        for iv in exogenous:
            preview_lines.append(f"{iv} × {mod} → {' + '.join(endogenous)} (moderation)")

    st.code("\n".join(preview_lines), language=None)

    # ── Permutation option ──
    st.subheader("Permutation Mode")
    run_permutations = st.checkbox(
        "Also run all mediator permutations and compare models",
        value=False,
        help="Generates all subsets of mediators (partial vs. full mediation) and ranks by AIC.",
    )

    st.divider()
    if st.button("▶ Run SEM & View Results", type="primary"):
        # ── Build mean scores for moderators ──
        active_clusters = {
            c: [i for i in items if i not in dropped]
            for c, items in clusters.items()
        }
        scores = build_mean_scores(df, active_clusters, selected)
        interaction_cols = []
        mod_df = df.copy()

        for mod in moderators:
            for iv in exogenous:
                scores, int_col = create_interaction(scores, iv, mod)
                mod_df[int_col] = scores[int_col]
                interaction_cols.append(int_col)

        # Merge interaction columns into df for semopy
        analysis_df = df.copy()
        for col in interaction_cols:
            analysis_df[col] = mod_df[col]

        syntax = build_sem_syntax(
            active_clusters,
            exogenous, endogenous, mediators,
            interaction_cols=interaction_cols if interaction_cols else None,
        )

        with st.spinner("Fitting SEM…"):
            model, paths, fit, warning = run_sem(analysis_df, syntax, estimator="WLS")

        if warning:
            st.warning(warning)

        st.session_state.sem_model = model
        st.session_state.sem_paths = paths
        st.session_state.sem_fit = fit

        # ── Mediation bootstrap ──
        if mediators:
            with st.spinner("Bootstrap indirect effects (2000 samples)…"):
                ie = run_all_indirect(
                    df, active_clusters, exogenous, mediators, endogenous, n_boot=2000
                )
            st.session_state.indirect_effects = ie

        # ── Simple slopes for moderation ──
        if moderators:
            iv = exogenous[0]
            dv = endogenous[0]
            mod = moderators[0]
            slope_df, slope_fig = simple_slopes(scores, iv, mod, dv)
            st.session_state.slope_df = slope_df
            st.session_state.interaction_fig = slope_fig

        # ── Permutations ──
        if run_permutations and mediators:
            perm_results = []
            perms = generate_permutations(exogenous, endogenous, mediators, max_models=30)
            progress = st.progress(0)
            for k, perm in enumerate(perms):
                perm_syntax = build_sem_syntax(
                    active_clusters,
                    exogenous, endogenous,
                    perm["mediators_used"],
                    interaction_cols=interaction_cols if interaction_cols else None,
                )
                _, _, perm_fit, _ = run_sem(analysis_df, perm_syntax, estimator="WLS")
                perm_results.append({
                    "Rank": "—",
                    "ID": perm["id"],
                    "Mediation": perm["mediation_type"],
                    "Mediators": ", ".join(perm["mediators_used"]) or "None",
                    "Direct Path": "Yes" if perm["include_direct"] else "No",
                    "CFI": perm_fit.get("CFI", np.nan),
                    "RMSEA": perm_fit.get("RMSEA", np.nan),
                    "SRMR": perm_fit.get("SRMR", np.nan),
                    "AIC": perm_fit.get("AIC", np.nan),
                    "BIC": perm_fit.get("BIC", np.nan),
                })
                progress.progress((k + 1) / len(perms))

            comp_df = pd.DataFrame(perm_results)
            try:
                comp_df = comp_df.sort_values("AIC").reset_index(drop=True)
                comp_df["Rank"] = comp_df.index + 1
            except Exception:
                pass
            st.session_state.model_comparison = comp_df

        go_to(4)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Results & Export
# ═══════════════════════════════════════════════════════════════════════════════
def page_results():
    st.header("📊 Results & Export")

    paths = st.session_state.sem_paths
    fit = st.session_state.sem_fit
    ie = st.session_state.indirect_effects
    comp = st.session_state.model_comparison
    model_name = st.session_state.model_name

    if paths.empty and not fit:
        st.info("Run the SEM model first (Step 4).")
        return

    # ── Path coefficients ──
    st.subheader("Structural Path Coefficients")
    if not paths.empty:
        def sig_marker(p_val):
            try:
                p = float(p_val)
                return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "n.s."))
            except (TypeError, ValueError):
                return "—"

        disp = paths.copy()
        disp["Sig."] = disp["p"].apply(sig_marker)

        def colour_sig(val):
            return {
                "***": "color:#1e8449;font-weight:bold",
                "**":  "color:#2980b9;font-weight:bold",
                "*":   "color:#d4ac0d;font-weight:bold",
                "n.s.": "color:#aaa",
            }.get(str(val), "")

        st.dataframe(
            disp.style.applymap(colour_sig, subset=["Sig."]),
            use_container_width=True, hide_index=True,
        )
        st.caption("* p<.05  ** p<.01  *** p<.001")

    # ── Fit indices ──
    st.subheader("Model Fit")
    if fit:
        benchmarks = {
            "CFI": "≥ 0.95 good / ≥ 0.90 acceptable",
            "TLI": "≥ 0.95 good / ≥ 0.90 acceptable",
            "RMSEA": "≤ 0.06 good / ≤ 0.08 acceptable",
            "SRMR": "≤ 0.08",
            "GFI": "≥ 0.95",
            "χ²": "—", "df": "—", "p(χ²)": "—",
            "AIC": "Lower is better", "BIC": "Lower is better",
        }
        from core.cfa import fit_status
        status = fit_status(fit)
        fit_rows = [
            {"Index": k, "Value": v, "Benchmark": benchmarks.get(k, "—"), "Status": status.get(k, "—")}
            for k, v in fit.items()
        ]
        fit_df = pd.DataFrame(fit_rows)

        def colour_status(val):
            return {
                "Good": "color:#1e8449;font-weight:bold",
                "Acceptable": "color:#d4ac0d;font-weight:bold",
                "Poor": "color:#c0392b;font-weight:bold",
            }.get(str(val), "")

        st.dataframe(
            fit_df.style.applymap(colour_status, subset=["Status"]),
            use_container_width=True, hide_index=True,
        )

    # ── Structural path diagram ──
    st.subheader("Structural Path Diagram")
    if not paths.empty:
        fig = draw_structural_diagram(
            st.session_state.exogenous,
            st.session_state.endogenous,
            st.session_state.mediators,
            st.session_state.moderators,
            paths,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Mediation ──
    if ie is not None and not ie.empty:
        st.subheader("Indirect Effects — Mediation")
        def colour_sig_ie(val):
            return "color:#1e8449;font-weight:bold" if val == "Yes" else "color:#c0392b"
        st.dataframe(
            ie.style.applymap(colour_sig_ie, subset=["Significant"]),
            use_container_width=True, hide_index=True,
        )

    # ── Moderation ──
    if st.session_state.interaction_fig is not None:
        st.subheader("Moderation — Simple Slopes")
        st.dataframe(st.session_state.slope_df, use_container_width=True, hide_index=True)
        st.plotly_chart(st.session_state.interaction_fig, use_container_width=True)

    # ── Permutation comparison ──
    if comp is not None and not comp.empty:
        st.subheader("Model Comparison (Permutations — ranked by AIC)")
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("Rank 1 = best-fitting model by AIC.")

    # ── PDF Export ──
    st.divider()
    st.subheader("Export PDF Report")
    if st.button("📄 Generate PDF Report", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        constructs_role = {
            c: st.session_state.roles.get(c, "—")
            for c in st.session_state.selected_constructs
        }
        reliability_clean = {
            k: {kk: vv for kk, vv in v.items() if kk != "item_df"}
            for k, v in st.session_state.reliability.items()
        }
        norm_df = st.session_state.normality_df if st.session_state.normality_df is not None \
                  else pd.DataFrame()

        try:
            generate_pdf(
                filepath=tmp_path,
                model_name=model_name,
                constructs=constructs_role,
                reliability=reliability_clean,
                normality_df=norm_df,
                mardia=st.session_state.mardia,
                cfa_loadings=st.session_state.cfa_loadings,
                cfa_fit=st.session_state.cfa_fit,
                sem_paths=paths,
                sem_fit=fit,
                roles=st.session_state.roles,
                indirect_effects=ie if (ie is not None and not ie.empty) else None,
                model_comparison=comp if (comp is not None and not comp.empty) else None,
            )
            with open(tmp_path, "rb") as f:
                pdf_bytes = f.read()

            safe_name = model_name.replace(" ", "_").replace("/", "-")
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"SEM_{safe_name}.pdf",
                mime="application/pdf",
            )
            st.success("PDF ready — click Download above.")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════════
PAGE_FNS = [page_upload, page_reliability, page_cfa, page_model_config, page_results]
PAGE_FNS[st.session_state.page]()
