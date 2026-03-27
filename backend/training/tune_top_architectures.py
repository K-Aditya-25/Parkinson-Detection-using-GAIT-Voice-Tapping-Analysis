"""Hyperparameter tuning for top-performing architectures (HistGradient, LightGBM)."""
import os
import sys
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, recall_score
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.train_gait_model import build_training_table

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def evaluate_loso_params(X, y, study_ids, scaler_name, model_type, params):
    """Evaluate model hyperparameters using LOSO."""
    studies = sorted(np.unique(study_ids))
    aucs = []
    
    for held_out_study in studies:
        test_mask = study_ids == held_out_study
        train_mask = ~test_mask
        
        X_train_raw = X[train_mask]
        y_train = y[train_mask]
        X_test_raw = X[test_mask]
        y_test = y[test_mask]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        scaler = RobustScaler() if scaler_name == 'robust' else None
        if scaler:
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw
        
        if model_type == 'histgradient':
            model = HistGradientBoostingClassifier(random_state=42, **params)
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
        
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
    
    return float(np.mean(aucs)) if aucs else 0.0


def tune_histgradient(X, y, study_ids, scaler_name='robust', max_configs=40):
    """Tune HistGradientBoosting hyperparameters."""
    print("\nTuning HistGradientBoosting...")
    print("─" * 70)
    
    best_score = 0
    best_params = {}
    
    configs = []
    for max_iter in [100, 150, 200]:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for max_depth in [3, 4, 5, 6]:
                for l2_reg in [0.0, 0.01, 0.1]:
                    configs.append({
                        'max_iter': max_iter,
                        'learning_rate': lr,
                        'max_depth': max_depth,
                        'l2_regularization': l2_reg,
                    })
    
    np.random.seed(42)
    idx = np.random.choice(len(configs), min(max_configs, len(configs)), replace=False)
    configs = [configs[i] for i in idx]
    
    print(f"Testing {len(configs)} configurations...")
    
    for i, params in enumerate(configs):
        score = evaluate_loso_params(X, y, study_ids, scaler_name, 'histgradient', params)
        if score > best_score:
            best_score = score
            best_params = params
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(configs)}] Best LOSO AUC: {best_score:.4f}")
    
    print(f"\nBest HistGradient LOSO AUC: {best_score:.4f}")
    print(f"Best params: {best_params}")
    return best_score, best_params


def tune_lightgbm(X, y, study_ids, scaler_name='robust', max_configs=40):
    """Tune LightGBM hyperparameters."""
    print("\nTuning LightGBM...")
    print("─" * 70)
    
    best_score = 0
    best_params = {}
    
    configs = []
    for n_estimators in [100, 150, 200]:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for num_leaves in [15, 31, 63]:
                for max_depth in [3, 4, 5]:
                    configs.append({
                        'n_estimators': n_estimators,
                        'learning_rate': lr,
                        'num_leaves': num_leaves,
                        'max_depth': max_depth,
                    })
    
    np.random.seed(42)
    idx = np.random.choice(len(configs), min(max_configs, len(configs)), replace=False)
    configs = [configs[i] for i in idx]
    
    print(f"Testing {len(configs)} configurations...")
    
    for i, params in enumerate(configs):
        score = evaluate_loso_params(X, y, study_ids, scaler_name, 'lightgbm', params)
        if score > best_score:
            best_score = score
            best_params = params
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(configs)}] Best LOSO AUC: {best_score:.4f}")
    
    print(f"\nBest LightGBM LOSO AUC: {best_score:.4f}")
    print(f"Best params: {best_params}")
    return best_score, best_params


if __name__ == '__main__':
    print("=" * 70)
    print("TUNED ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    X, y, feature_names, subject_ids, study_ids = build_training_table(mode='subject_mean')
    
    print(f"Data loaded: {X.shape}")
    print(f"Studies: {dict(zip(*np.unique(study_ids, return_counts=True)))}\n")
    
    # Tune both top architectures
    hg_score, hg_params = tune_histgradient(X, y, study_ids, scaler_name='robust', max_configs=40)
    lgb_score, lgb_params = tune_lightgbm(X, y, study_ids, scaler_name='robust', max_configs=40)
    
    # Compare baseline (from previous run)
    xgb_baseline = 0.7687  # From earlier run
    
    print("\n" + "=" * 70)
    print("TUNING RESULTS SUMMARY")
    print("=" * 70)
    print(f"HistGradient (tuned):  {hg_score:.4f}")
    print(f"LightGBM (tuned):      {lgb_score:.4f}")
    print(f"XGBoost (baseline):    {xgb_baseline:.4f}")
    print(f"\nImprovement over baseline:")
    print(f"  HistGradient: +{hg_score - xgb_baseline:.4f}")
    print(f"  LightGBM:     +{lgb_score - xgb_baseline:.4f}")
    
    # Save results
    results = {
        'comparison': 'tuned_architectures',
        'baseline_xgboost': xgb_baseline,
        'histgradient': {
            'loso_auc': hg_score,
            'best_params': hg_params,
        },
        'lightgbm': {
            'loso_auc': lgb_score,
            'best_params': lgb_params,
        },
    }
    
    output_path = os.path.join(MODEL_DIR, 'gait_architecture_tuning_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
