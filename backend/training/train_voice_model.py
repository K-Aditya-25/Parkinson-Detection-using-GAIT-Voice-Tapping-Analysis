"""
Train voice XGBoost model on Oxford Parkinson's Disease Detection Dataset.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
import shap

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    """Load Oxford Parkinson's dataset."""
    # Try ucimlrepo first, then direct download
    try:
        from ucimlrepo import fetch_ucirepo
        parkinsons = fetch_ucirepo(id=174)
        X = parkinsons.data.features
        y = parkinsons.data.targets.values.ravel()
        print("Loaded via ucimlrepo")
    except (ImportError, Exception):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        df = pd.read_csv(url)
        X = df.drop(columns=['name', 'status'])
        y = df['status'].values
        print("Loaded via direct download")

    print(f"Dataset shape: {X.shape}")
    print(f"Healthy: {np.sum(y == 0)}, PD: {np.sum(y == 1)}")
    return X, y


def train_model(X, y):
    """Train XGBoost model."""
    feature_names = list(X.columns) if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
    X = np.array(X)

    # Drop highly correlated features (> 0.95)
    corr = np.corrcoef(X.T)
    to_drop = set()
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if abs(corr[i][j]) > 0.95:
                to_drop.add(j)

    keep_idx = [i for i in range(X.shape[1]) if i not in to_drop]
    X = X[:, keep_idx]
    feature_names = [feature_names[i] for i in keep_idx]
    print(f"Features after correlation filter: {len(feature_names)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train final
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nTest Set Results:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'PD']))

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'shap_voice_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    model_path = os.path.join(MODEL_DIR, 'voice_model.pkl')
    joblib.dump(bundle, model_path)
    print(f"\nModel saved to {model_path}")

    return model, scaler


if __name__ == '__main__':
    print("=" * 60)
    print("Training Voice Model on Oxford Dataset")
    print("=" * 60)

    X, y = load_dataset()
    model, scaler = train_model(X, y)

    print("\nDone!")
