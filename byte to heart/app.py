"""
app.py — Cardio-Lens: Democratizing Heart Health
Multi-page Streamlit Application | Two-Tier AI System
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from backend import (
    train_tier1_model, predict_tier1, simulate_bp_reduction,
    train_tier2_model, predict_tier2,
    TIER2_FEATURES
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio-Lens",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0a1628 50%, #0d1f3c 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.25);
}
.metric-card .value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tier cards */
.tier-card {
    border-radius: 20px;
    padding: 32px;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.tier-card-1 {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
    border: 1px solid rgba(56,189,248,0.3);
}
.tier-card-2 {
    background: linear-gradient(135deg, #3b1f5e 0%, #2d1b4e 100%);
    border: 1px solid rgba(167,139,250,0.3);
}
.tier-card h3 {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 12px;
}
.tier-card p { color: #94a3b8; line-height: 1.7; }
.tier-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.badge-1 { background: rgba(56,189,248,0.2); color: #38bdf8; border: 1px solid rgba(56,189,248,0.4); }
.badge-2 { background: rgba(167,139,250,0.2); color: #a78bfa; border: 1px solid rgba(167,139,250,0.4); }

/* Risk gauge */
.risk-container {
    text-align: center;
    padding: 32px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(15,23,42,0.8) 0%, rgba(30,27,75,0.6) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    backdrop-filter: blur(10px);
}
.risk-value {
    font-size: 5rem;
    font-weight: 900;
    line-height: 1;
    margin: 16px 0;
}
.risk-label { font-size: 1rem; color: #94a3b8; font-weight: 500; }
.risk-low    { color: #34d399; }
.risk-medium { color: #fbbf24; }
.risk-high   { color: #f87171; }

/* Section headers */
.section-header {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.section-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 28px;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, rgba(52,211,153,0.1) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
}
.insight-box p { color: #6ee7b7; margin: 0; font-size: 0.9rem; }

/* Diagnosis box */
.diag-box {
    border-radius: 20px;
    padding: 32px;
    text-align: center;
}
.diag-positive {
    background: linear-gradient(135deg, rgba(248,113,113,0.15) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(248,113,113,0.4);
}
.diag-negative {
    background: linear-gradient(135deg, rgba(52,211,153,0.15) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(52,211,153,0.4);
}
.diag-prob {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    margin: 12px 0;
}
.diag-positive .diag-prob { color: #f87171; }
.diag-negative .diag-prob { color: #34d399; }

/* Streamlit overrides */
div[data-testid="stSlider"] label { color: #94a3b8 !important; }
.stSelectbox label, .stNumberInput label, .stRadio label { color: #94a3b8 !important; }
div[data-testid="stForm"] { background: transparent; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 32px;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def get_models():
    m1, acc1 = train_tier1_model()
    m2, acc2 = train_tier2_model()
    return m1, acc1, m2, acc2

model1, acc1, model2, acc2 = get_models()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:3rem;'>🫀</div>
        <div style='font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#818cf8,#c084fc);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;'>Cardio-Lens</div>
        <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>Two-Tier AI Heart Health</div>
    </div>
    <hr style='border-color:rgba(99,102,241,0.2); margin:16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  The Pitch", "📡  Tier 1: Screening", "🔬  Tier 2: Diagnosis", "🧬  Health Twin"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(99,102,241,0.2); margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#475569; padding:0 4px;'>
        <div style='margin-bottom:8px;'>
            <span style='color:#38bdf8; font-weight:600;'>Tier 1 Accuracy</span><br>
            <span style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>{acc1:.1%}</span>
        </div>
        <div>
            <span style='color:#a78bfa; font-weight:600;'>Tier 2 Accuracy</span><br>
            <span style='font-size:1.1rem; font-weight:700; color:#e2e8f0;'>{acc2:.1%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 1 — THE PITCH
# ═══════════════════════════════════════════════════════════
if page == "🏠  The Pitch":

    st.markdown("""
    <div style='text-align:center; padding: 48px 0 32px;'>
        <div style='font-size:1rem; color:#6366f1; font-weight:600; letter-spacing:0.15em;
                    text-transform:uppercase; margin-bottom:16px;'>
            🏆 Hackathon Project 2026
        </div>
        <h1 style='font-size:3.5rem; font-weight:900; line-height:1.1; margin:0;
                   background:linear-gradient(135deg,#818cf8 0%,#c084fc 50%,#f472b6 100%);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   background-clip:text;'>
            Cardio-Lens
        </h1>
        <p style='font-size:1.4rem; color:#94a3b8; margin-top:12px; font-weight:400;'>
            Democratizing Heart Health with a Two-Tier AI System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='value'>70K</div>
            <div class='label'>Training Records</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{acc1:.0%}</div>
            <div class='label'>Tier 1 Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{acc2:.0%}</div>
            <div class='label'>Tier 2 Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='value'>2</div>
            <div class='label'>AI Models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-tier explanation
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("""
        <div class='tier-card tier-card-1'>
            <div class='tier-badge badge-1'>📡 Tier 1 · Mass Screening</div>
            <h3 style='color:#38bdf8;'>The "Watch" Model</h3>
            <p>Designed for <strong style='color:#e2e8f0;'>wearables & home devices</strong>.
            Uses 12 basic biometric signals — no lab tests required. Screens 70,000+ population
            records to flag at-risk individuals <em>before</em> symptoms appear.</p>
            <br>
            <div style='display:flex; gap:12px; flex-wrap:wrap;'>
                <span style='background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3);
                             border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#7dd3fc;'>
                    ⚡ Instant Results
                </span>
                <span style='background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3);
                             border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#7dd3fc;'>
                    🏠 Home-Ready
                </span>
                <span style='background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3);
                             border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#7dd3fc;'>
                    📊 BP Simulator
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class='tier-card tier-card-2'>
            <div class='tier-badge badge-2'>🔬 Tier 2 · Clinical Precision</div>
            <h3 style='color:#a78bfa;'>The "Clinical" Model</h3>
            <p>Built for <strong style='color:#e2e8f0;'>doctors & hospitals</strong>.
            Trained on 918 clinical records with ECG readings, ST-slope analysis, and
            chest pain classification. Delivers high-precision diagnosis with full
            <em>explainability</em>.</p>
            <br>
            <div style='display:flex; gap:12px; flex-wrap:wrap;'>
                <span style='background:rgba(167,139,250,0.1); border:1px solid rgba(167,139,250,0.3);
                             border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#c4b5fd;'>
                    🧠 XAI Ready
                </span>
                <span style='background:rgba(167,139,250,0.1); border:1px solid rgba(167,139,250,0.3);
                             border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#c4b5fd;'>
                    📋 Clinical Grade
                </span>
                <span style='background:rgba(167,139,250,0.1); border:1px solid rgba(167,139,250,0.3);
                             border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#c4b5fd;'>
                    📈 Feature Importance
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline flow
    st.markdown("""
    <div style='text-align:center; margin:32px 0 16px;'>
        <div class='section-header'>The AI Pipeline</div>
        <div class='section-sub'>From wearable to clinical — a seamless escalation path</div>
    </div>
    """, unsafe_allow_html=True)

    flow_cols = st.columns(5)
    steps = [
        ("📱", "Wearable\nDevice", "#38bdf8"),
        ("→", "", "#475569"),
        ("📡", "Tier 1\nScreening", "#818cf8"),
        ("→", "", "#475569"),
        ("🔬", "Tier 2\nDiagnosis", "#a78bfa"),
    ]
    for col, (icon, label, color) in zip(flow_cols, steps):
        with col:
            if icon == "→":
                st.markdown(f"<div style='text-align:center; font-size:2rem; color:{color}; padding-top:24px;'>→</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align:center; padding:20px 12px;
                            background:rgba(255,255,255,0.03); border-radius:16px;
                            border:1px solid rgba(255,255,255,0.07);'>
                    <div style='font-size:2.5rem;'>{icon}</div>
                    <div style='font-size:0.8rem; color:{color}; font-weight:600;
                                margin-top:8px; white-space:pre-line;'>{label}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#475569; font-size:0.85rem;'>
        ⚠️ <em>For educational & research purposes only. Not a substitute for professional medical advice.</em>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 — TIER 1: POPULATION SCREENING
# ═══════════════════════════════════════════════════════════
elif page == "📡  Tier 1: Screening":

    st.markdown("""
    <div style='padding: 24px 0 8px;'>
        <div class='tier-badge badge-1' style='display:inline-block;'>📡 Tier 1 · Mass Screening</div>
        <div class='section-header'>Population Screening</div>
        <div class='section-sub'>Enter your biometric data to get an instant cardiovascular risk score</div>
    </div>
    """, unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1, 1.2], gap="large")

    with col_inputs:
        st.markdown("#### 📋 Your Biometrics")

        age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, step=1)
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        gender_val = 2 if gender == "Male" else 1

        c1, c2 = st.columns(2)
        with c1:
            height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170, step=1)
        with c2:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.5)

        bmi_display = weight / ((height / 100) ** 2)
        bmi_color = "#34d399" if bmi_display < 25 else ("#fbbf24" if bmi_display < 30 else "#f87171")
        st.markdown(f"<div style='font-size:0.85rem; color:{bmi_color}; margin:-8px 0 8px;'>BMI: {bmi_display:.1f}</div>",
                    unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            ap_hi = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=200, value=130, step=1)
        with c4:
            ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=140, value=85, step=1)

        cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
        chol_val = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]

        gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
        gluc_val = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]

        c5, c6, c7 = st.columns(3)
        with c5:
            smoke = st.checkbox("🚬 Smoker")
        with c6:
            alco = st.checkbox("🍺 Alcohol")
        with c7:
            active = st.checkbox("🏃 Active", value=True)

        predict_btn = st.button("🫀 Calculate Risk Score", use_container_width=True)

    with col_results:
        if predict_btn or "tier1_result" in st.session_state:
            if predict_btn:
                risk = predict_tier1(
                    model1, age, gender_val, height, weight,
                    ap_hi, ap_lo, chol_val, gluc_val,
                    int(smoke), int(alco), int(active)
                )
                st.session_state["tier1_result"] = risk
                st.session_state["tier1_inputs"] = dict(
                    age=age, gender_val=gender_val, height=height, weight=weight,
                    ap_hi=ap_hi, ap_lo=ap_lo, chol_val=chol_val, gluc_val=gluc_val,
                    smoke=int(smoke), alco=int(alco), active=int(active)
                )

            risk = st.session_state["tier1_result"]
            inp  = st.session_state["tier1_inputs"]
            risk_pct = risk * 100

            if risk_pct < 30:
                risk_class, risk_label, risk_emoji = "risk-low", "LOW RISK", "✅"
            elif risk_pct < 60:
                risk_class, risk_label, risk_emoji = "risk-medium", "MODERATE RISK", "⚠️"
            else:
                risk_class, risk_label, risk_emoji = "risk-high", "HIGH RISK", "🚨"

            st.markdown(f"""
            <div class='risk-container'>
                <div class='risk-label'>Cardiovascular Risk Score</div>
                <div class='risk-value {risk_class}'>{risk_pct:.1f}%</div>
                <div style='font-size:1.2rem; font-weight:700; margin-bottom:8px;'>{risk_emoji} {risk_label}</div>
                <div style='font-size:0.85rem; color:#64748b;'>Based on your biometric profile</div>
            </div>
            """, unsafe_allow_html=True)

            # ── ACTIONABLE INSIGHTS SIMULATOR ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='margin-bottom:8px;'>
                <span style='font-size:1.1rem; font-weight:700; color:#818cf8;'>
                    💡 Actionable Insights Simulator
                </span><br>
                <span style='font-size:0.85rem; color:#64748b;'>
                    Drag the slider to see how lowering your blood pressure reduces your risk
                </span>
            </div>
            """, unsafe_allow_html=True)

            current_ap_hi = inp["ap_hi"]
            min_bp = max(90, current_ap_hi - 50)

            target_bp = st.slider(
                "🎯 Target Systolic BP (mmHg)",
                min_value=min_bp,
                max_value=current_ap_hi,
                value=min_bp,
                step=1,
                help="Slide left to simulate the effect of lowering your blood pressure"
            )

            # Simulate risk across BP range
            sim_df = simulate_bp_reduction(
                model1,
                inp["age"], inp["gender_val"], inp["height"], inp["weight"],
                current_ap_hi, inp["ap_lo"],
                inp["chol_val"], inp["gluc_val"],
                inp["smoke"], inp["alco"], inp["active"],
                target_bp=target_bp
            )

            # Highlight current vs target
            current_risk_row = sim_df[sim_df["Systolic BP"] == current_ap_hi]
            target_risk_row  = sim_df[sim_df["Systolic BP"] == target_bp]

            if not current_risk_row.empty and not target_risk_row.empty:
                current_r = current_risk_row["Risk (%)"].values[0]
                target_r  = target_risk_row["Risk (%)"].values[0]
                reduction  = current_r - target_r

                # Altair chart
                line = alt.Chart(sim_df).mark_line(
                    color="#818cf8", strokeWidth=3, interpolate="monotone"
                ).encode(
                    x=alt.X("Systolic BP:Q",
                            scale=alt.Scale(domain=[target_bp, current_ap_hi]),
                            axis=alt.Axis(title="Systolic Blood Pressure (mmHg)",
                                          labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.05)")),
                    y=alt.Y("Risk (%):Q",
                            scale=alt.Scale(domain=[max(0, sim_df["Risk (%)"].min() - 5),
                                                     min(100, sim_df["Risk (%)"].max() + 5)]),
                            axis=alt.Axis(title="Cardiovascular Risk (%)",
                                          labelColor="#94a3b8", titleColor="#94a3b8",
                                          gridColor="rgba(255,255,255,0.05)")),
                    tooltip=["Systolic BP:Q", alt.Tooltip("Risk (%):Q", format=".1f")]
                )

                area = alt.Chart(sim_df).mark_area(
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="rgba(129,140,248,0.4)", offset=0),
                            alt.GradientStop(color="rgba(129,140,248,0.0)", offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0
                    ),
                    interpolate="monotone"
                ).encode(
                    x="Systolic BP:Q",
                    y="Risk (%):Q"
                )

                # Target point
                target_point_df = pd.DataFrame([{"Systolic BP": target_bp, "Risk (%)": target_r}])
                target_point = alt.Chart(target_point_df).mark_point(
                    color="#34d399", size=120, filled=True
                ).encode(x="Systolic BP:Q", y="Risk (%):Q")

                # Current point
                current_point_df = pd.DataFrame([{"Systolic BP": current_ap_hi, "Risk (%)": current_r}])
                current_point = alt.Chart(current_point_df).mark_point(
                    color="#f87171", size=120, filled=True
                ).encode(x="Systolic BP:Q", y="Risk (%):Q")

                chart = (area + line + target_point + current_point).properties(
                    height=260,
                    background="transparent",
                    title=alt.TitleParams(
                        "Risk Reduction Simulation",
                        color="#e2e8f0", fontSize=14, fontWeight="bold"
                    )
                ).configure_view(
                    strokeWidth=0,
                    fill="transparent"
                ).configure_axis(
                    domainColor="rgba(255,255,255,0.1)",
                    tickColor="rgba(255,255,255,0.1)"
                )

                st.altair_chart(chart, use_container_width=True)

                if reduction > 0:
                    st.markdown(f"""
                    <div class='insight-box'>
                        <p>🎯 By lowering your systolic BP from <strong>{current_ap_hi}</strong>
                        to <strong>{target_bp} mmHg</strong>, your estimated risk drops by
                        <strong>{reduction:.1f} percentage points</strong>
                        ({current_r:.1f}% → {target_r:.1f}%).</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='insight-box'>
                        <p>✅ Your current blood pressure is already at the target level.</p>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:80px 20px; color:#475569;'>
                <div style='font-size:4rem; margin-bottom:16px;'>📡</div>
                <div style='font-size:1.1rem; font-weight:600; color:#64748b;'>
                    Enter your biometrics and click<br>"Calculate Risk Score"
                </div>
                <div style='font-size:0.85rem; margin-top:8px;'>
                    The Actionable Insights Simulator will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3 — TIER 2: CLINICAL DIAGNOSIS
# ═══════════════════════════════════════════════════════════
elif page == "🔬  Tier 2: Diagnosis":

    st.markdown("""
    <div style='padding: 24px 0 8px;'>
        <div class='tier-badge badge-2' style='display:inline-block;'>🔬 Tier 2 · Clinical Precision</div>
        <div class='section-header'>Clinical Diagnosis</div>
        <div class='section-sub'>Advanced clinical parameters for high-precision heart disease detection</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_diag = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown("#### 🏥 Clinical Parameters")

        c1, c2 = st.columns(2)
        with c1:
            t2_age = st.number_input("Age", min_value=20, max_value=100, value=55, step=1)
        with c2:
            t2_sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        sex_m = 1 if t2_sex == "Male" else 0

        c3, c4 = st.columns(2)
        with c3:
            t2_rbp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=140, step=1)
        with c4:
            t2_chol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=250, step=5)

        c5, c6 = st.columns(2)
        with c5:
            t2_maxhr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=130, step=1)
        with c6:
            t2_oldpeak = st.number_input("Oldpeak (ST Depr.)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

        t2_fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"], horizontal=True)
        fbs_val = 1 if t2_fbs == "Yes" else 0

        t2_cpt = st.selectbox(
            "Chest Pain Type",
            ["ASY — Asymptomatic", "ATA — Atypical Angina", "NAP — Non-Anginal Pain", "TA — Typical Angina"]
        )
        cpt_code = t2_cpt.split(" — ")[0]
        cpt_ata = int(cpt_code == "ATA")
        cpt_nap = int(cpt_code == "NAP")
        cpt_ta  = int(cpt_code == "TA")

        t2_ecg = st.selectbox("Resting ECG", ["Normal", "LVH — Left Ventricular Hypertrophy", "ST — ST-T Wave Abnormality"])
        ecg_code = t2_ecg.split(" — ")[0]
        ecg_normal = int(ecg_code == "Normal")
        ecg_st     = int(ecg_code == "ST")

        t2_ea = st.radio("Exercise-Induced Angina?", ["No", "Yes"], horizontal=True)
        ea_val = 1 if t2_ea == "Yes" else 0

        t2_slope = st.selectbox("ST Slope", ["Up — Upsloping", "Flat — Flat", "Down — Downsloping"])
        slope_code = t2_slope.split(" — ")[0]
        slope_flat = int(slope_code == "Flat")
        slope_up   = int(slope_code == "Up")

        diag_btn = st.button("🔬 Run Clinical Diagnosis", use_container_width=True)

    with col_diag:
        if diag_btn or "tier2_result" in st.session_state:
            if diag_btn:
                features = {
                    "Age":               t2_age,
                    "RestingBP":         t2_rbp,
                    "Cholesterol":       t2_chol,
                    "FastingBS":         fbs_val,
                    "MaxHR":             t2_maxhr,
                    "Oldpeak":           t2_oldpeak,
                    "Sex_M":             sex_m,
                    "ChestPainType_ATA": cpt_ata,
                    "ChestPainType_NAP": cpt_nap,
                    "ChestPainType_TA":  cpt_ta,
                    "RestingECG_Normal": ecg_normal,
                    "RestingECG_ST":     ecg_st,
                    "ExerciseAngina_Y":  ea_val,
                    "ST_Slope_Flat":     slope_flat,
                    "ST_Slope_Up":       slope_up,
                }
                prob, importances = predict_tier2(model2, features)
                st.session_state["tier2_result"] = (prob, importances)

            prob, importances = st.session_state["tier2_result"]
            prob_pct = prob * 100

            if prob_pct >= 50:
                diag_class = "diag-positive"
                diag_label = "Heart Disease Likely"
                diag_emoji = "🚨"
                diag_advice = "High probability detected. Immediate clinical consultation recommended."
            else:
                diag_class = "diag-negative"
                diag_label = "Heart Disease Unlikely"
                diag_emoji = "✅"
                diag_advice = "Low probability detected. Continue regular health monitoring."

            st.markdown(f"""
            <div class='diag-box {diag_class}'>
                <div style='font-size:0.9rem; color:#94a3b8; font-weight:500;'>Diagnosis Probability</div>
                <div class='diag-prob'>{prob_pct:.1f}%</div>
                <div style='font-size:1.2rem; font-weight:700; margin-bottom:8px;'>{diag_emoji} {diag_label}</div>
                <div style='font-size:0.82rem; color:#64748b;'>{diag_advice}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── FEATURE IMPORTANCE CHART ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='margin-bottom:8px;'>
                <span style='font-size:1.1rem; font-weight:700; color:#a78bfa;'>
                    🧠 Model Explainability — Feature Importance
                </span><br>
                <span style='font-size:0.85rem; color:#64748b;'>
                    Which clinical factors drive this model's decisions
                </span>
            </div>
            """, unsafe_allow_html=True)

            imp_df = importances.reset_index()
            imp_df.columns = ["Feature", "Importance"]
            imp_df["Importance (%)"] = (imp_df["Importance"] * 100).round(2)

            # Color scale: higher importance → brighter purple
            bars = alt.Chart(imp_df).mark_bar(
                cornerRadiusTopRight=6,
                cornerRadiusBottomRight=6
            ).encode(
                y=alt.Y("Feature:N",
                        sort=alt.EncodingSortField(field="Importance", order="descending"),
                        axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                      labelFontSize=11, gridColor="rgba(255,255,255,0.05)")),
                x=alt.X("Importance (%):Q",
                        axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8",
                                      gridColor="rgba(255,255,255,0.05)",
                                      title="Importance (%)")),
                color=alt.Color("Importance (%):Q",
                                scale=alt.Scale(range=["#4f46e5", "#c084fc"]),
                                legend=None),
                tooltip=["Feature:N", alt.Tooltip("Importance (%):Q", format=".2f")]
            ).properties(
                height=340,
                background="transparent",
                title=alt.TitleParams(
                    "Feature Importance (Random Forest)",
                    color="#e2e8f0", fontSize=13, fontWeight="bold"
                )
            ).configure_view(
                strokeWidth=0,
                fill="transparent"
            ).configure_axis(
                domainColor="rgba(255,255,255,0.1)",
                tickColor="rgba(255,255,255,0.1)"
            )

            st.altair_chart(bars, use_container_width=True)

            # Top 3 insight
            top3 = imp_df.nlargest(3, "Importance")
            top3_names = ", ".join(f"**{r['Feature']}** ({r['Importance (%)']:.1f}%)"
                                   for _, r in top3.iterrows())
            st.markdown(f"""
            <div class='insight-box'>
                <p>🔍 Top 3 drivers of this prediction: {top3_names.replace('**', '<strong>').replace('**', '</strong>')}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:80px 20px; color:#475569;'>
                <div style='font-size:4rem; margin-bottom:16px;'>🔬</div>
                <div style='font-size:1.1rem; font-weight:600; color:#64748b;'>
                    Fill in the clinical parameters<br>and click "Run Clinical Diagnosis"
                </div>
                <div style='font-size:0.85rem; margin-top:8px;'>
                    Feature importance chart will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#374151; font-size:0.8rem; padding:16px;
                background:rgba(255,255,255,0.02); border-radius:12px;
                border:1px solid rgba(255,255,255,0.05);'>
        ⚠️ <strong style='color:#475569;'>Medical Disclaimer:</strong>
        This tool is for research and educational purposes only.
        Always consult a qualified healthcare professional for medical decisions.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4 — 🧬 HEALTH TWIN SIMULATOR (UNIQUE FEATURE)
# ═══════════════════════════════════════════════════════════
elif page == "🧬  Health Twin":

    st.markdown("""
    <style>
    .twin-card {
        border-radius: 20px;
        padding: 28px;
        height: 100%;
    }
    .twin-current {
        background: linear-gradient(135deg, rgba(248,113,113,0.12) 0%, rgba(239,68,68,0.05) 100%);
        border: 1px solid rgba(248,113,113,0.35);
    }
    .twin-future {
        background: linear-gradient(135deg, rgba(52,211,153,0.12) 0%, rgba(16,185,129,0.05) 100%);
        border: 1px solid rgba(52,211,153,0.35);
    }
    .twin-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; margin-bottom: 12px;
    }
    .twin-risk {
        font-size: 3.8rem; font-weight: 900; line-height: 1; margin: 8px 0;
    }
    .twin-current .twin-risk { color: #f87171; }
    .twin-future  .twin-risk { color: #34d399; }
    .rx-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.07) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .rx-item {
        display: flex; align-items: flex-start; gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .rx-item:last-child { border-bottom: none; }
    .rx-icon { font-size: 1.4rem; flex-shrink: 0; }
    .rx-text { font-size: 0.88rem; color: #94a3b8; line-height: 1.5; }
    .rx-text strong { color: #e2e8f0; }
    .years-saved {
        text-align: center;
        background: linear-gradient(135deg, rgba(251,191,36,0.15) 0%, rgba(245,158,11,0.05) 100%);
        border: 1px solid rgba(251,191,36,0.35);
        border-radius: 16px;
        padding: 24px;
        margin-top: 16px;
    }
    .years-saved .big { font-size: 3rem; font-weight: 900; color: #fbbf24; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='padding: 24px 0 8px;'>
        <div class='tier-badge' style='display:inline-block; background:rgba(251,191,36,0.15);
             color:#fbbf24; border:1px solid rgba(251,191,36,0.4);'>🧬 Unique Feature</div>
        <div class='section-header'>Health Twin Simulator</div>
        <div class='section-sub'>
            Meet your Future Healthy Self — AI-powered 10-year risk trajectory &amp; personalised prescription
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUTS ──
    st.markdown("#### 👤 Your Current Profile")
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        ht_age    = st.number_input("Age", min_value=18, max_value=80, value=42, step=1, key="ht_age")
        ht_gender = st.radio("Gender", ["Female", "Male"], horizontal=True, key="ht_gender")
        ht_gval   = 2 if ht_gender == "Male" else 1
    with ci2:
        ht_height = st.number_input("Height (cm)", min_value=140, max_value=220, value=172, step=1, key="ht_h")
        ht_weight = st.number_input("Weight (kg)", min_value=40.0, max_value=180.0, value=88.0, step=0.5, key="ht_w")
    with ci3:
        ht_aphi   = st.number_input("Systolic BP", min_value=90, max_value=200, value=148, step=1, key="ht_bp")
        ht_aplo   = st.number_input("Diastolic BP", min_value=50, max_value=140, value=92, step=1, key="ht_bpd")

    ci4, ci5 = st.columns(2)
    with ci4:
        ht_chol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"], index=1, key="ht_chol")
        ht_cval = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[ht_chol]
        ht_gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"], key="ht_gluc")
        ht_gval2 = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[ht_gluc]
    with ci5:
        ht_smoke  = st.checkbox("🚬 Currently Smoking", value=True, key="ht_smoke")
        ht_alco   = st.checkbox("🍺 Regular Alcohol", value=True, key="ht_alco")
        ht_active = st.checkbox("🏃 Physically Active", value=False, key="ht_active")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Design Your Future Self")
    st.markdown("<div style='font-size:0.85rem; color:#64748b; margin-bottom:16px;'>Adjust the sliders to set your health goals — the AI will instantly project your new risk trajectory</div>",
                unsafe_allow_html=True)

    fi1, fi2 = st.columns(2)
    with fi1:
        goal_bp     = st.slider("🩺 Target Systolic BP", 90, ht_aphi, min(120, ht_aphi), key="goal_bp")
        goal_weight = st.slider("⚖️ Target Weight (kg)", max(40, int(ht_weight) - 30),
                                int(ht_weight), int(ht_weight), key="goal_w")
    with fi2:
        goal_chol   = st.selectbox("🧪 Target Cholesterol", ["Normal", "Above Normal", "Well Above Normal"],
                                   index=max(0, ht_cval - 2), key="goal_chol")
        goal_cval   = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[goal_chol]
        goal_smoke  = st.checkbox("🚭 Quit Smoking",    value=ht_smoke,  key="goal_smoke")
        goal_active = st.checkbox("🏋️ Become Active",  value=not ht_active, key="goal_active")

    simulate_btn = st.button("🧬 Generate My Health Twin", use_container_width=True)

    if simulate_btn or "twin_result" in st.session_state:
        if simulate_btn:
            # ── Compute current risk ──
            current_risk = predict_tier1(
                model1, ht_age, ht_gval, ht_height, ht_weight,
                ht_aphi, ht_aplo, ht_cval, ht_gval2,
                int(ht_smoke), int(ht_alco), int(ht_active)
            )
            # ── Compute future risk ──
            future_bmi_weight = goal_weight
            future_risk = predict_tier1(
                model1, ht_age, ht_gval, ht_height, future_bmi_weight,
                goal_bp, ht_aplo, goal_cval, 1,
                int(not goal_smoke), 0, int(goal_active)
            )
            # ── 10-year trajectory ──
            # Simulate risk aging from current age to current age + 10
            trajectory_rows = []
            for yr in range(11):
                age_now  = ht_age + yr
                r_curr = predict_tier1(
                    model1, age_now, ht_gval, ht_height, ht_weight,
                    ht_aphi, ht_aplo, ht_cval, ht_gval2,
                    int(ht_smoke), int(ht_alco), int(ht_active)
                )
                r_fut = predict_tier1(
                    model1, age_now, ht_gval, ht_height, future_bmi_weight,
                    goal_bp, ht_aplo, goal_cval, 1,
                    int(not goal_smoke), 0, int(goal_active)
                )
                trajectory_rows.append({"Year": f"Age {age_now}", "Age": age_now,
                                         "Current Path": round(r_curr * 100, 2),
                                         "Healthy Twin": round(r_fut * 100, 2)})
            traj_df = pd.DataFrame(trajectory_rows)

            # ── Prescription ──
            prescription = []
            if goal_bp < ht_aphi:
                prescription.append(("🩺", "Blood Pressure",
                                      f"Reduce systolic BP from {ht_aphi} → {goal_bp} mmHg",
                                      f"−{ht_aphi - goal_bp} mmHg"))
            if goal_weight < ht_weight:
                prescription.append(("⚖️", "Weight Loss",
                                      f"Lose {ht_weight - goal_weight:.1f} kg through diet & exercise",
                                      f"−{ht_weight - goal_weight:.1f} kg"))
            if goal_cval < ht_cval:
                prescription.append(("🧪", "Cholesterol",
                                      "Improve cholesterol through diet, statins if needed",
                                      "Improved"))
            if ht_smoke and goal_smoke:
                prescription.append(("🚭", "Quit Smoking",
                                      "Cessation reduces cardiovascular risk within 1 year",
                                      "Eliminated"))
            if not ht_active and goal_active:
                prescription.append(("🏋️", "Exercise",
                                      "30 min moderate activity, 5× per week",
                                      "Active"))
            if not prescription:
                prescription.append(("✅", "Already Optimal",
                                      "Your goals match your current lifestyle — great work!",
                                      "Maintained"))

            st.session_state["twin_result"] = {
                "current": current_risk,
                "future":  future_risk,
                "traj":    traj_df,
                "rx":      prescription,
            }

        res = st.session_state["twin_result"]
        curr_pct  = res["current"] * 100
        fut_pct   = res["future"]  * 100
        reduction = curr_pct - fut_pct
        traj_df   = res["traj"]
        rx        = res["rx"]

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SIDE-BY-SIDE TWIN CARDS ──
        tc1, tc_mid, tc2 = st.columns([1, 0.15, 1])
        with tc1:
            st.markdown(f"""
            <div class='twin-card twin-current'>
                <div class='twin-label' style='color:#f87171;'>😔 Current You</div>
                <div class='twin-risk'>{curr_pct:.1f}%</div>
                <div style='font-size:0.85rem; color:#94a3b8; margin-top:6px;'>Cardiovascular Risk</div>
                <hr style='border-color:rgba(248,113,113,0.2); margin:16px 0;'>
                <div style='font-size:0.82rem; color:#94a3b8; line-height:1.8;'>
                    BP: {ht_aphi}/{ht_aplo} mmHg<br>
                    Weight: {ht_weight} kg<br>
                    Cholesterol: {ht_chol}<br>
                    Smoking: {'Yes' if ht_smoke else 'No'} &nbsp;|&nbsp; Active: {'Yes' if ht_active else 'No'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tc_mid:
            st.markdown("<div style='text-align:center; font-size:2rem; padding-top:60px; color:#6366f1;'>→</div>",
                        unsafe_allow_html=True)

        with tc2:
            st.markdown(f"""
            <div class='twin-card twin-future'>
                <div class='twin-label' style='color:#34d399;'>🌟 Future Healthy You</div>
                <div class='twin-risk'>{fut_pct:.1f}%</div>
                <div style='font-size:0.85rem; color:#94a3b8; margin-top:6px;'>Cardiovascular Risk</div>
                <hr style='border-color:rgba(52,211,153,0.2); margin:16px 0;'>
                <div style='font-size:0.82rem; color:#94a3b8; line-height:1.8;'>
                    BP: {goal_bp}/{ht_aplo} mmHg<br>
                    Weight: {goal_weight} kg<br>
                    Cholesterol: {goal_chol}<br>
                    Smoking: {'Yes' if not goal_smoke else 'No'} &nbsp;|&nbsp; Active: {'Yes' if goal_active else 'No'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── YEARS SAVED BADGE ──
        if reduction > 0:
            years_equiv = round(reduction / 3.5, 1)  # ~3.5% risk per year of aging
            st.markdown(f"""
            <div class='years-saved'>
                <div style='font-size:0.85rem; color:#94a3b8; margin-bottom:4px;'>Estimated Risk Reduction</div>
                <div class='big'>−{reduction:.1f}%</div>
                <div style='font-size:0.9rem; color:#fbbf24; margin-top:4px;'>
                    ≈ {years_equiv} years of cardiovascular aging reversed
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 10-YEAR TRAJECTORY CHART ──
        st.markdown("""
        <div style='margin-bottom:8px;'>
            <span style='font-size:1.1rem; font-weight:700; color:#fbbf24;'>📈 10-Year Risk Trajectory</span><br>
            <span style='font-size:0.85rem; color:#64748b;'>
                How your cardiovascular risk evolves over the next decade — two futures, one choice
            </span>
        </div>
        """, unsafe_allow_html=True)

        traj_long = traj_df.melt(
            id_vars=["Year", "Age"],
            value_vars=["Current Path", "Healthy Twin"],
            var_name="Scenario",
            value_name="Risk (%)"
        )

        color_scale = alt.Scale(
            domain=["Current Path", "Healthy Twin"],
            range=["#f87171", "#34d399"]
        )

        traj_line = alt.Chart(traj_long).mark_line(
            strokeWidth=3, interpolate="monotone"
        ).encode(
            x=alt.X("Age:Q",
                    axis=alt.Axis(title="Age (years)", labelColor="#94a3b8",
                                  titleColor="#94a3b8", gridColor="rgba(255,255,255,0.05)",
                                  tickCount=11)),
            y=alt.Y("Risk (%):Q",
                    scale=alt.Scale(domain=[max(0, traj_long["Risk (%)"].min() - 5),
                                            min(100, traj_long["Risk (%)"].max() + 5)]),
                    axis=alt.Axis(title="Cardiovascular Risk (%)", labelColor="#94a3b8",
                                  titleColor="#94a3b8", gridColor="rgba(255,255,255,0.05)")),
            color=alt.Color("Scenario:N", scale=color_scale,
                            legend=alt.Legend(orient="top-right", labelColor="#e2e8f0",
                                              titleColor="#94a3b8", labelFontSize=12)),
            tooltip=["Year:N", "Scenario:N", alt.Tooltip("Risk (%):Q", format=".1f")]
        )

        traj_area = alt.Chart(traj_long).mark_area(
            opacity=0.15, interpolate="monotone"
        ).encode(
            x="Age:Q",
            y="Risk (%):Q",
            color=alt.Color("Scenario:N", scale=color_scale, legend=None)
        )

        traj_points = alt.Chart(traj_long).mark_point(
            filled=True, size=60
        ).encode(
            x="Age:Q",
            y="Risk (%):Q",
            color=alt.Color("Scenario:N", scale=color_scale, legend=None),
            tooltip=["Year:N", "Scenario:N", alt.Tooltip("Risk (%):Q", format=".1f")]
        )

        traj_chart = (traj_area + traj_line + traj_points).properties(
            height=300,
            background="transparent",
            title=alt.TitleParams(
                "10-Year Cardiovascular Risk Projection",
                color="#e2e8f0", fontSize=14, fontWeight="bold"
            )
        ).configure_view(
            strokeWidth=0, fill="transparent"
        ).configure_axis(
            domainColor="rgba(255,255,255,0.1)",
            tickColor="rgba(255,255,255,0.1)"
        )

        st.altair_chart(traj_chart, use_container_width=True)

        # ── AI PRESCRIPTION CARD ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:1.1rem; font-weight:700; color:#818cf8; margin-bottom:12px;'>
            💊 Your AI-Generated Health Prescription
        </div>
        """, unsafe_allow_html=True)

        rx_html = "<div class='rx-card'>"
        for icon, title, desc, impact in rx:
            rx_html += f"""
            <div class='rx-item'>
                <div class='rx-icon'>{icon}</div>
                <div class='rx-text'>
                    <strong>{title}</strong> &nbsp;
                    <span style='background:rgba(99,102,241,0.2); color:#a5b4fc;
                                 border-radius:6px; padding:2px 8px; font-size:0.75rem;
                                 font-weight:600;'>{impact}</span><br>
                    {desc}
                </div>
            </div>"""
        rx_html += "</div>"
        st.markdown(rx_html, unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center; color:#374151; font-size:0.8rem; padding:16px;
                    background:rgba(255,255,255,0.02); border-radius:12px;
                    border:1px solid rgba(255,255,255,0.05); margin-top:16px;'>
            ⚠️ <strong style='color:#475569;'>Medical Disclaimer:</strong>
            Projections are AI estimates based on population data. Consult a healthcare professional.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align:center; padding:80px 20px; color:#475569;'>
            <div style='font-size:5rem; margin-bottom:16px;'>🧬</div>
            <div style='font-size:1.2rem; font-weight:700; color:#64748b;'>
                Set your health goals above<br>and click "Generate My Health Twin"
            </div>
            <div style='font-size:0.85rem; margin-top:12px; color:#374151;'>
                You'll see your Current Self vs Future Healthy Self,<br>
                a 10-year AI risk trajectory, and a personalised prescription
            </div>
        </div>
        """, unsafe_allow_html=True)
