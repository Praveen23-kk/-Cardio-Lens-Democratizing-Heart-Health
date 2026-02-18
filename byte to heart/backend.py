"""
backend.py — Cardio-Lens Data Pipeline & Model Training
Two-Tier AI System for Heart Disease Detection
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDIO_PATH = os.path.join(BASE_DIR, "dataset", "cardio_base.csv")
HEART_PATH  = os.path.join(BASE_DIR, "dataset", "heart_processed.csv")

# ─────────────────────────────────────────────
# TIER 1 — POPULATION SCREENING MODEL
# ─────────────────────────────────────────────

TIER1_FEATURES = [
    "age_years", "gender", "height", "weight",
    "bmi", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]

def load_and_preprocess_tier1() -> pd.DataFrame:
    df = pd.read_csv(CARDIO_PATH, sep=";")
    # Convert age from days → years
    df["age_years"] = (df["age"] / 365.25).round(1)
    # Compute BMI
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    # Filter unreasonable blood pressure values
    df = df[(df["ap_hi"] >= 90) & (df["ap_hi"] <= 200)]
    df = df[(df["ap_lo"] >= 50) & (df["ap_lo"] <= 140)]
    # Drop rows with NaN
    df = df.dropna(subset=TIER1_FEATURES + ["cardio"])
    return df


@st.cache_resource(show_spinner="🫀 Training Tier 1 Screening Model…")
def train_tier1_model():
    df = load_and_preprocess_tier1()
    X = df[TIER1_FEATURES]
    y = df["cardio"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


def predict_tier1(model, age, gender, height, weight, ap_hi, ap_lo,
                  cholesterol, gluc, smoke, alco, active) -> float:
    """Return cardiovascular risk probability (0–1)."""
    bmi = weight / ((height / 100) ** 2)
    features = pd.DataFrame([{
        "age_years":   age,
        "gender":      gender,
        "height":      height,
        "weight":      weight,
        "bmi":         bmi,
        "ap_hi":       ap_hi,
        "ap_lo":       ap_lo,
        "cholesterol": cholesterol,
        "gluc":        gluc,
        "smoke":       smoke,
        "alco":        alco,
        "active":      active,
    }])
    prob = model.predict_proba(features)[0][1]
    return float(prob)


def simulate_bp_reduction(model, age, gender, height, weight, ap_hi, ap_lo,
                           cholesterol, gluc, smoke, alco, active,
                           target_bp: int) -> pd.DataFrame:
    """
    Simulate risk across a range of systolic BP values from target_bp to ap_hi.
    Returns a DataFrame with columns ['Systolic BP', 'Risk (%)'].
    """
    bmi = weight / ((height / 100) ** 2)
    bp_range = list(range(target_bp, ap_hi + 1, 1))
    risks = []
    for bp in bp_range:
        row = {
            "age_years":   age,
            "gender":      gender,
            "height":      height,
            "weight":      weight,
            "bmi":         bmi,
            "ap_hi":       bp,
            "ap_lo":       ap_lo,
            "cholesterol": cholesterol,
            "gluc":        gluc,
            "smoke":       smoke,
            "alco":        alco,
            "active":      active,
        }
        prob = model.predict_proba(pd.DataFrame([row]))[0][1]
        risks.append({"Systolic BP": bp, "Risk (%)": round(prob * 100, 2)})
    return pd.DataFrame(risks)


# ─────────────────────────────────────────────
# TIER 2 — CLINICAL DIAGNOSIS MODEL
# ─────────────────────────────────────────────

TIER2_FEATURES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_M",
    "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat", "ST_Slope_Up"
]

TIER2_FEATURE_LABELS = {
    "Age":               "Age",
    "RestingBP":         "Resting BP",
    "Cholesterol":       "Cholesterol",
    "FastingBS":         "Fasting Blood Sugar",
    "MaxHR":             "Max Heart Rate",
    "Oldpeak":           "ST Depression (Oldpeak)",
    "Sex_M":             "Sex (Male)",
    "ChestPainType_ATA": "Chest Pain: Atypical Angina",
    "ChestPainType_NAP": "Chest Pain: Non-Anginal",
    "ChestPainType_TA":  "Chest Pain: Typical Angina",
    "RestingECG_Normal": "ECG: Normal",
    "RestingECG_ST":     "ECG: ST Abnormality",
    "ExerciseAngina_Y":  "Exercise-Induced Angina",
    "ST_Slope_Flat":     "ST Slope: Flat",
    "ST_Slope_Up":       "ST Slope: Upsloping",
}


def load_and_preprocess_tier2() -> pd.DataFrame:
    df = pd.read_csv(HEART_PATH)
    df = df.dropna(subset=TIER2_FEATURES + ["HeartDisease"])
    # Ensure boolean columns are int (0/1) for sklearn
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


@st.cache_resource(show_spinner="🔬 Training Tier 2 Clinical Model…")
def train_tier2_model():
    df = load_and_preprocess_tier2()
    X = df[TIER2_FEATURES]
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


def predict_tier2(model, features_dict: dict) -> tuple[float, pd.Series]:
    """
    Returns (probability, feature_importances_series).
    feature_importances_series is indexed by human-readable labels.
    """
    row = pd.DataFrame([features_dict])[TIER2_FEATURES]
    prob = model.predict_proba(row)[0][1]
    importances = pd.Series(
        model.feature_importances_,
        index=[TIER2_FEATURE_LABELS.get(f, f) for f in TIER2_FEATURES]
    ).sort_values(ascending=True)
    return float(prob), importances


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Tier 1 pipeline…")
    df1 = load_and_preprocess_tier1()
    print(f"  Tier 1 records after cleaning: {len(df1):,}")
    print(f"  Cardio prevalence: {df1['cardio'].mean():.1%}")

    print("\nTesting Tier 2 pipeline…")
    df2 = load_and_preprocess_tier2()
    print(f"  Tier 2 records: {len(df2):,}")
    print(f"  HeartDisease prevalence: {df2['HeartDisease'].mean():.1%}")
    print("\nAll pipelines OK ✓")
