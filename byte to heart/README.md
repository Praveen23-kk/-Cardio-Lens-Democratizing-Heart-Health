# 🫀 Cardio-Lens: Democratizing Heart Health

> **Hackathon Project 2026** — A Two-Tier AI System for Heart Disease Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 What is Cardio-Lens?

Cardio-Lens is a **multi-page Streamlit application** that uses a **Two-Tier AI System** to make heart disease detection accessible to everyone — from home wearables to clinical settings.

```
📱 Wearable Device  →  📡 Tier 1: Mass Screening  →  🔬 Tier 2: Clinical Diagnosis
```

---

## 🚀 Features

### 📡 Tier 1 — Population Screening ("The Watch Model")
- Trained on **70,000+ records** from `cardio_base.csv`
- Inputs: Age, Gender, Height, Weight, Blood Pressure, Cholesterol, Glucose, Lifestyle
- **Actionable Insights Simulator**: Drag a slider to see how lowering your BP reduces your risk in real-time (interactive Altair chart)

### 🔬 Tier 2 — Clinical Diagnosis ("The Clinical Model")
- Trained on **918 clinical records** from `heart_processed.csv`
- Inputs: Chest Pain Type, ST Slope, MaxHR, RestingECG, Exercise Angina, and more
- **Feature Importance Chart**: Explainable AI — see exactly which clinical factors drive the prediction

### 🧬 Health Twin Simulator *(Unique Feature)*
The standout differentiator — **no other heart disease app does this**:
- **Current You vs Future Healthy You** — side-by-side risk comparison cards
- **10-Year AI Risk Trajectory** — dual-line chart projecting risk over the next decade
- **"Years of Aging Reversed"** — converts risk reduction into an intuitive metric
- **AI Health Prescription** — auto-generated action plan (BP, weight, smoking, exercise)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Streamlit** | Multi-page web UI |
| **scikit-learn** | Random Forest Classifiers (×2) |
| **Pandas / NumPy** | Data pipeline & preprocessing |
| **Altair** | Interactive charts & visualizations |

---

## 📂 Project Structure

```
byte-to-heart/
├── app.py              # Main Streamlit application (4 pages)
├── backend.py          # Data pipelines + model training + prediction functions
├── requirements.txt    # Python dependencies
├── dataset/
│   ├── cardio_base.csv       # Tier 1: 70k population records (delimiter: ;)
│   └── heart_processed.csv   # Tier 2: 918 clinical records
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Praveen23-kk/new-life-.git
cd new-life-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Model Performance

| Model | Dataset | Records | Accuracy |
|---|---|---|---|
| Tier 1 (Screening) | cardio_base.csv | 68,551 (after cleaning) | ~72% |
| Tier 2 (Clinical) | heart_processed.csv | 918 | ~87% |

---

## 🧬 The Health Twin Simulator — How It Works

1. Enter your **current health profile** (age, BP, weight, smoking status, etc.)
2. Use sliders to **design your Future Healthy Self** (target BP, weight goal, quit smoking)
3. Click **"Generate My Health Twin"**
4. The AI runs **22 predictions** (11 years × 2 scenarios) to build your 10-year trajectory
5. Get your **AI Health Prescription** — a personalised action plan

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 👨‍💻 Author

Built for the **2026 Hackathon** — Cardio-Lens: Democratizing Heart Health with AI.
