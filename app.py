# ============================================================
# CSTE 3207 ML PROJECT
# Breast Cancer Classification Web Application
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. STREAMLIT PAGE SETTINGS & MOBILE CSS STYLING
# ============================================================

st.set_page_config(
    page_title="Breast Cancer ML Classifier",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for Mobile & UI Polish
st.markdown("""
<style>
    /* Main container padding adjustments for mobile */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 1rem;
        }
        h1 {
            font-size: 1.6rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
    }

    /* Modern Card Design */
    .metric-card {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }

    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background-color: #fcfcfc;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. TRAIN ALL MODELS
# ============================================================

@st.cache_resource
def train_models():

    data = load_breast_cancer(
        as_frame=True
    )

    df = data.frame.copy()

    df["diagnosis"] = df["target"].map({
        0: "Malignant",
        1: "Benign"
    })

    df = df.drop(
        columns="target"
    )

    X = df.drop(
        columns="diagnosis"
    ).copy()

    y = df["diagnosis"].map({
        "Malignant": 1,
        "Benign": 0
    })

    # FEATURE ENGINEERING
    X["area_perimeter_ratio"] = (
        X["mean area"] /
        (X["mean perimeter"] + 1e-9)
    )

    X["concavity_compactness_ratio"] = (
        X["mean concavity"] /
        (X["mean compactness"] + 1e-9)
    )

    X["worst_radius_mean_ratio"] = (
        X["worst radius"] /
        (X["mean radius"] + 1e-9)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = {

        "Dummy Baseline": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", DummyClassifier(strategy="most_frequent"))
        ]),

        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=5000))
        ]),

        "Random Forest": GridSearchCV(
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            {
                "m__n_estimators": [200, 400],
                "m__max_depth": [None, 6, 10],
                "m__min_samples_split": [2, 5]
            },
            cv=cv,
            scoring="f1",
            n_jobs=-1
        ),

        "SVM": GridSearchCV(
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", SVC(probability=True))
            ]),
            {
                "m__C": [0.1, 1, 10],
                "m__gamma": ["scale", 0.01, 0.1],
                "m__kernel": ["rbf"]
            },
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, prob)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "ROC-AUC": round(roc_auc, 3)
        })

    performance = pd.DataFrame(results)

    return data, models, performance


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

data, models, performance = train_models()
feature_names = data.feature_names.tolist()


# ============================================================
# 4. CALLBACK FUNCTIONS FOR SIDEBAR BUTTONS
# ============================================================

def load_sample_patient():
    sample = data.data.iloc[0]
    for feature in feature_names:
        st.session_state[feature] = float(sample[feature])


def clear_patient_inputs():
    for feature in feature_names:
        st.session_state[feature] = 0.0


# ============================================================
# 5. SIDEBAR: PATIENT INPUT SECTION
# ============================================================

st.sidebar.title("🩺 Patient Features")
st.sidebar.write("Enter values manually or load a sample patient.")

col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    st.sidebar.button("📋 Load Sample", on_click=load_sample_patient, use_container_width=True)
with col_b2:
    st.sidebar.button("🗑️️ Clear", on_click=clear_patient_inputs, use_container_width=True)

st.sidebar.divider()

user_data = {}
for feature in feature_names:
    user_data[feature] = st.sidebar.number_input(
        feature,
        key=feature,
        value=0.0,
        format="%.6f"
    )

st.sidebar.divider()
predict_clicked = st.sidebar.button("🔍 Predict Diagnosis", type="primary", use_container_width=True)


# ============================================================
# 6. MAIN PAGE HEADER & CONTENT
# ============================================================

st.title("🧬 Breast Cancer Classification System")
st.write("Machine-learning demonstration based on the Breast Cancer Wisconsin Diagnostic dataset.")
st.info("Educational ML demonstration only — this application is not a medical diagnosis tool.")

st.divider()


# ============================================================
# 7. PREDICTION LOGIC & DISPLAY
# ============================================================

if predict_clicked:

    user_df = pd.DataFrame([user_data])

    # Feature Engineering for user input
    user_df["area_perimeter_ratio"] = (
        user_df["mean area"] / (user_df["mean perimeter"] + 1e-9)
    )
    user_df["concavity_compactness_ratio"] = (
        user_df["mean concavity"] / (user_df["mean compactness"] + 1e-9)
    )
    user_df["worst_radius_mean_ratio"] = (
        user_df["worst radius"] / (user_df["mean radius"] + 1e-9)
    )

    prediction_results = []

    for name, model in models.items():
        prediction = int(model.predict(user_df)[0])
        probability = float(model.predict_proba(user_df)[0][1])

        diagnosis = "MALIGNANT" if prediction == 1 else "BENIGN"

        prediction_results.append({
            "Model": name,
            "Prediction": diagnosis,
            "Malignant Probability (%)": round(probability * 100, 2)
        })

    result_df = pd.DataFrame(prediction_results)

    # Layout for Results (Automatically stacks nicely on mobile screens)
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.subheader("Patient Prediction Results")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    with res_col2:
        st.subheader("Prediction Summary")
        
        malignant_count = result_df["Prediction"].eq("MALIGNANT").sum()
        benign_count = result_df["Prediction"].eq("BENIGN").sum()
        
        majority_pred = "MALIGNANT" if malignant_count > benign_count else "BENIGN"
        agreement_text = f"{max(malignant_count, benign_count)} / {len(models)} Models"
        
        # Enhanced responsive card container
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin-bottom: 8px;">🟢 <b>BENIGN:</b> &nbsp;&nbsp; {benign_count} Model{'s' if benign_count > 1 else ''}</p>
            <p style="margin-bottom: 12px;">🔴 <b>MALIGNANT:</b> &nbsp;&nbsp; {malignant_count} Model{'s' if malignant_count > 1 else ''}</p>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #ddd;">
            <p style="margin-bottom: 4px; color: #555; font-size: 14px;"><b>Majority Prediction:</b></p>
            <p style="font-size: 18px; font-weight: bold; margin-bottom: 12px;">
                {'🔴 MALIGNANT' if majority_pred == 'MALIGNANT' else '🟢 BENIGN'}
            </p>
            <p style="margin-bottom: 4px; color: #555; font-size: 14px;"><b>Model Agreement:</b></p>
            <p style="font-size: 16px; font-weight: bold; margin: 0;">{agreement_text}</p>
        </div>
        """, unsafe_allow_html=True)

    st.warning(
        "The displayed probability is the model's estimated probability for the Malignant class. "
        "It must not be interpreted as a clinical diagnosis or medical certainty."
    )

else:
    st.info("👈 Please enter patient details in the sidebar and click **Predict Diagnosis** to see results.")


# ============================================================
# 8. MODEL PERFORMANCE SECTION
# ============================================================

st.divider()
st.subheader("Model Performance on Test Set")
st.dataframe(performance, use_container_width=True, hide_index=True)


# ============================================================
# 9. PROJECT FOOTER
# ============================================================

st.divider()
st.caption("Train/test split: 80/20, stratified. Random Forest and SVM use 5-fold GridSearchCV with F1 scoring.")
st.caption("CSTE 3207 ML Project | Breast Cancer Classification")