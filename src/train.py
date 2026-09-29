import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

df["diagnosis"] = df.target.map({
    0: "Malignant",
    1: "Benign"
})

df = df.drop(columns="target")

X = df.drop(columns="diagnosis").copy()

y = df["diagnosis"].map({
    "Malignant": 1,
    "Benign": 0
})

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

X["area_perimeter_ratio"] = (
    X["mean area"] / (X["mean perimeter"] + 1e-9)
)

X["concavity_compactness_ratio"] = (
    X["mean concavity"] / (X["mean compactness"] + 1e-9)
)

X["worst_radius_mean_ratio"] = (
    X["worst radius"] / (X["mean radius"] + 1e-9)
)

# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

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

# ============================================================
# 4. MODELS
# ============================================================

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
            ("m", RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ))
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

# ============================================================
# 5. TRAIN + EVALUATE MODELS
# ============================================================

results_list = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    p = model.predict(X_test)
    s = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, p)
    prec = precision_score(y_test, p, zero_division=0)
    rec = recall_score(y_test, p, zero_division=0)
    f1 = f1_score(y_test, p, zero_division=0)
    roc = roc_auc_score(y_test, s)

    results_list.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "ROC-AUC": round(roc, 3)
    })

# ============================================================
# 6. PERFORMANCE TABLE
# ============================================================

df_results = pd.DataFrame(results_list)

print("\n" + "=" * 70)
print("              MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(df_results.to_string(index=False))

print("=" * 70)


# ============================================================
# 7. USER INPUT PREDICTION
# ============================================================

print("\n")
print("=" * 70)
print("          BREAST CANCER PREDICTION SYSTEM")
print("=" * 70)

print("\nPlease enter the following patient feature values.")
print("You can enter decimal values such as 14.2, 19.5, etc.\n")


# Take first trained model for prediction
prediction_model = models["Logistic Regression"]


# ------------------------------------------------------------
# Get feature names
# ------------------------------------------------------------

original_features = data.feature_names.tolist()

user_data = {}

for feature in original_features:

    while True:

        try:
            value = float(input(f"Enter {feature}: "))
            user_data[feature] = value
            break

        except ValueError:
            print("Please enter a valid numerical value.")


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

user_df = pd.DataFrame([user_data])


# ------------------------------------------------------------
# Apply same feature engineering
# ------------------------------------------------------------

user_df["area_perimeter_ratio"] = (
    user_df["mean area"] /
    (user_df["mean perimeter"] + 1e-9)
)

user_df["concavity_compactness_ratio"] = (
    user_df["mean concavity"] /
    (user_df["mean compactness"] + 1e-9)
)

user_df["worst_radius_mean_ratio"] = (
    user_df["worst radius"] /
    (user_df["mean radius"] + 1e-9)
)


# ------------------------------------------------------------
# Prediction using all models
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("                 PREDICTION RESULTS")
print("=" * 70)

for name, model in models.items():

    prediction = model.predict(user_df)[0]

    probability = model.predict_proba(user_df)[0][1]

    if prediction == 1:
        diagnosis = "MALIGNANT"
    else:
        diagnosis = "BENIGN"

    print(f"\nModel: {name}")
    print(f"Prediction: {diagnosis}")
    print(f"Malignant Probability: {probability * 100:.2f}%")

print("\n" + "=" * 70)