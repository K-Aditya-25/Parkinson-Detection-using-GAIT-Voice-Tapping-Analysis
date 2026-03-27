"""
Train gait XGBoost model on PhysioNet Gait in Parkinson's Disease dataset.
"""
import os
import sys
import re
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from feature_extraction.gait_features import extract_gait_features_from_vgrf, get_feature_names

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'gait-in-parkinsons-disease-1.0.0')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_extract_features():
    subjects = {}
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    print(f"Found {len(files)} data files")

    for filename in sorted(files):
        filepath = os.path.join(DATA_DIR, filename)
        base = filename.replace('.txt', '')
        match = re.match(r'([A-Za-z]+\d+)', base)
        if not match:
            continue
        subject_id = match.group(1)

        if 'Co' in subject_id:
            label = 0
        elif 'Pt' in subject_id:
            label = 1
        else:
            continue

        try:
            data = np.loadtxt(filepath, delimiter='\t')
        except:
            continue

        if data.shape[1] < 19:
            continue

        timestamps = data[:, 0]
        total_force_left = data[:, 17]
        total_force_right = data[:, 18]

        if np.max(total_force_left) < 1 or np.max(total_force_right) < 1:
            continue

        features = extract_gait_features_from_vgrf(timestamps, total_force_left, total_force_right)
        if any(np.isnan(v) for v in features.values()):
            continue

        if subject_id not in subjects:
            subjects[subject_id] = {'label': label, 'features_list': []}
        subjects[subject_id]['features_list'].append(features)

    print(f"Processed {len(subjects)} subjects")

    feature_names = get_feature_names()
    X_list, y_list = [], []

    for sid, info in subjects.items():
        avg = {}
        for fname in feature_names:
            vals = [f[fname] for f in info['features_list']]
            avg[fname] = np.mean(vals)
        X_list.append([avg[f] for f in feature_names])
        y_list.append(info['label'])

    X = np.array(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Feature matrix: {X.shape}, Controls: {np.sum(y==0)}, PD: {np.sum(y==1)}")
    return X, y, feature_names


def train_model(X, y, feature_names):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Manual hyperparameter search (no parallel jobs to avoid Windows memory issue)
    best_score = 0
    best_params = {}

    param_combos = []
    for n_est in [100, 200, 300]:
        for depth in [3, 4, 5, 6]:
            for lr in [0.01, 0.05, 0.1]:
                for sub in [0.7, 0.8, 1.0]:
                    for gamma in [0, 0.1, 0.5]:
                        param_combos.append({
                            'n_estimators': n_est,
                            'max_depth': depth,
                            'learning_rate': lr,
                            'subsample': sub,
                            'gamma': gamma,
                            'colsample_bytree': 0.8,
                            'min_child_weight': 3,
                            'reg_alpha': 0.1,
                            'reg_lambda': 2.0,
                        })

    # Subsample for speed
    np.random.seed(42)
    idx = np.random.choice(len(param_combos), min(80, len(param_combos)), replace=False)
    param_combos = [param_combos[i] for i in idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Searching {len(param_combos)} hyperparameter combinations...")
    for i, params in enumerate(param_combos):
        model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            **params
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(param_combos)}] Best CV: {best_score:.4f}")

    print(f"\nBest params: {best_params}")
    print(f"Best CV Accuracy: {best_score:.4f}")

    # Train final model with best params
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', **best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nTest Set Results:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'PD']))

    # SHAP
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'shap_gait_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("SHAP summary plot saved.")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")

    bundle = {'model': model, 'scaler': scaler, 'feature_names': feature_names}
    model_path = os.path.join(MODEL_DIR, 'gait_model.pkl')
    joblib.dump(bundle, model_path)
    print(f"Model saved to {model_path}")
    return model, scaler


if __name__ == '__main__':
    print("=" * 60)
    print("Training Gait Model on PhysioNet Dataset")
    print("=" * 60)
    X, y, feature_names = load_and_extract_features()
    model, scaler = train_model(X, y, feature_names)
    print("\nDone!")
