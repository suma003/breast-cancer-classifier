# CSTE 3207 Machine Learning Project
## Breast Cancer Wisconsin (Diagnostic) Classification

### Project objective
An end-to-end supervised machine learning pipeline that classifies breast tumor samples as **malignant** or **benign**.

### Dataset
The project uses the public **Breast Cancer Wisconsin (Diagnostic)** dataset through `sklearn.datasets.load_breast_cancer`, which contains 569 samples and 30 numeric predictive features. The original source is the UCI Machine Learning Repository.

### Guideline coverage
- Problem formulation: supervised binary classification
- EDA: distributions, missing values, outliers, correlations
- Preprocessing: median imputation, standardization
- Feature engineering: three ratio features
- Baseline: Dummy Classifier
- Advanced models: Logistic Regression, Random Forest, SVM
- Hyperparameter tuning: GridSearchCV with 5-fold stratified cross-validation
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves
- Error analysis: misclassified samples and limitations
- Feature importance: Random Forest
- Deliverables: report, notebook, source code, figures, presentation

### Reproducibility
Run:
```bash
pip install -r requirements.txt
python src/train.py
```

The supplied CSV is a local copy generated from scikit-learn's bundled copy of the UCI dataset, so the main experiment does not require an internet connection.

### Important note
This is an academic ML project and **not a clinical diagnostic system**. Model predictions should not be used for medical decision-making.
