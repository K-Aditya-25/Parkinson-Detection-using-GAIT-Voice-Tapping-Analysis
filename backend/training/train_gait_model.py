"""
Train gait XGBoost model on PhysioNet Gait in Parkinson's Disease dataset.
"""
import argparse
import os
import sys
import re
import json
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report,
                             balanced_accuracy_score, average_precision_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from feature_extraction.gait_features import extract_gait_features_from_vgrf, get_feature_names

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'gait-in-parkinsons-disease-1.0.0')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Model factory for creating different architectures
def create_model(model_type, params=None, class_weight=None):
    """Create a classifier of the specified type with given parameters."""
    if params is None:
        params = {}
    
    if model_type == 'xgboost':
        p = {
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        if class_weight is not None:
            p['scale_pos_weight'] = class_weight
        p.update(params)
        return xgb.XGBClassifier(**p)
    
    elif model_type == 'catboost':
        p = {
            'random_state': 42,
            'iterations': 100,
            'depth': 4,
            'learning_rate': 0.1,
            'verbose': False,
            'allow_writing_files': False,
        }
        if class_weight is not None:
            p['scale_pos_weight'] = class_weight
        p.update(params)
        return cb.CatBoostClassifier(**p)
    
    elif model_type == 'lightgbm':
        p = {
            'random_state': 42,
            'num_leaves': 31,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': -1,
            'verbose': -1,
        }
        if class_weight is not None:
            p['scale_pos_weight'] = class_weight
        p.update(params)
        return lgb.LGBMClassifier(**p)
    
    elif model_type == 'histgradient':
        p = {
            'random_state': 42,
            'max_iter': 100,
            'learning_rate': 0.1,
            'max_depth': 4,
        }
        p.update(params)
        return HistGradientBoostingClassifier(**p)
    
    elif model_type == 'randomforest':
        p = {
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 10,
            'n_jobs': -1,
        }
        if class_weight is not None:
            p['class_weight'] = 'balanced' if class_weight else None
        p.update(params)
        return RandomForestClassifier(**p)
    
    elif model_type == 'extratrees':
        p = {
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 10,
            'n_jobs': -1,
        }
        if class_weight is not None:
            p['class_weight'] = 'balanced' if class_weight else None
        p.update(params)
        return ExtraTreesClassifier(**p)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_trial_level_rows():
    rows = []
    feature_names = get_feature_names()
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    print(f"Found {len(files)} data files")

    for filename in sorted(files):
        filepath = os.path.join(DATA_DIR, filename)
        base = filename.replace('.txt', '')
        trial_match = re.match(r'([A-Za-z]+\d+)(?:_(\d+))?$', base)
        if not trial_match:
            continue

        subject_id = trial_match.group(1)
        trial_num = int(trial_match.group(2)) if trial_match.group(2) else 1
        study_id = subject_id[:2]

        match = re.match(r'([A-Za-z]+\d+)', subject_id)
        if not match:
            continue

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

        row = {
            'subject_id': subject_id,
            'study_id': study_id,
            'trial_num': trial_num,
            'is_dual_task': 1 if trial_num == 10 else 0,
            'label': label,
        }
        row.update(features)
        rows.append(row)

    print(f"Processed {len(rows)} valid trials")
    return rows, feature_names


def _build_subject_mean_table(df, base_feature_names):
    grouped = df.groupby('subject_id', sort=True)
    rows = []

    for subject_id, group in grouped:
        out = {
            'subject_id': subject_id,
            'study_id': group['study_id'].iloc[0],
            'label': int(group['label'].iloc[0]),
            'trial_count': int(len(group)),
        }
        for name in base_feature_names:
            out[name] = float(group[name].mean())
        rows.append(out)

    table = pd.DataFrame(rows)
    feature_names = list(base_feature_names)
    return table, feature_names


def _build_trial_table(df, base_feature_names):
    table = df.copy()
    feature_names = list(base_feature_names) + ['is_dual_task', 'trial_num']
    return table, feature_names


def _build_subject_hierarchical_table(df, base_feature_names):
    grouped = df.groupby('subject_id', sort=True)
    rows = []
    dual_delta_keys = ['stride_time_cv', 'cadence', 'step_regularity', 'stride_regularity']

    for subject_id, group in grouped:
        out = {
            'subject_id': subject_id,
            'study_id': group['study_id'].iloc[0],
            'label': int(group['label'].iloc[0]),
            'trial_count': int(len(group)),
            'dual_task_fraction': float(group['is_dual_task'].mean()),
        }

        for name in base_feature_names:
            vals = group[name].to_numpy(dtype=float)
            out[f'{name}_mean'] = float(np.mean(vals))
            out[f'{name}_median'] = float(np.median(vals))
            out[f'{name}_std'] = float(np.std(vals)) if len(vals) > 1 else 0.0

        normal_group = group[group['is_dual_task'] == 0]
        dual_group = group[group['is_dual_task'] == 1]
        for key in dual_delta_keys:
            if len(normal_group) > 0 and len(dual_group) > 0:
                out[f'delta_dual_{key}'] = float(dual_group[key].mean() - normal_group[key].mean())
            else:
                out[f'delta_dual_{key}'] = 0.0

        rows.append(out)

    table = pd.DataFrame(rows)
    feature_names = [
        c for c in table.columns
        if c not in ['subject_id', 'study_id', 'label']
    ]
    return table, feature_names


def build_training_table(mode='subject_mean'):
    rows, base_feature_names = load_trial_level_rows()
    df = pd.DataFrame(rows)
    raw_trial_counts = df.groupby('subject_id').size().to_numpy(dtype=float)

    if mode == 'subject_mean':
        table, feature_names = _build_subject_mean_table(df, base_feature_names)
    elif mode == 'trial':
        table, feature_names = _build_trial_table(df, base_feature_names)
    elif mode == 'subject_hierarchical':
        table, feature_names = _build_subject_hierarchical_table(df, base_feature_names)
    else:
        raise ValueError(f'Unsupported table mode: {mode}')

    X = table[feature_names].to_numpy(dtype=float)
    y = table['label'].to_numpy(dtype=int)
    subject_ids = table['subject_id'].to_numpy(dtype=str)
    study_ids = table['study_id'].to_numpy(dtype=str)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nTable mode: {mode}")
    print(f"Feature matrix: {X.shape}, Controls: {np.sum(y==0)}, PD: {np.sum(y==1)}")
    study_counts = pd.Series(study_ids).value_counts().to_dict()
    print(f"Studies: {study_counts}")
    print(f"Avg valid trials/subject: {np.mean(raw_trial_counts):.2f}")

    return X, y, feature_names, subject_ids, study_ids


def _compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics['auc_roc'] = None
    return metrics


def _evaluate_params_auc(X_train, y_train, groups_train, params, model_type='xgboost'):
    """Evaluate model hyperparameters using group-stratified cross-validation."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    for tr_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Recompute class weighting in each fold.
        n_pos = int(np.sum(y_tr == 1))
        n_neg = int(np.sum(y_tr == 0))
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        model = create_model(model_type, params, class_weight=scale_pos_weight)
        model.fit(X_tr, y_tr)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        if len(np.unique(y_val)) < 2:
            continue
        fold_aucs.append(roc_auc_score(y_val, y_val_prob))

    if not fold_aucs:
        return 0.0
    return float(np.mean(fold_aucs))


def _evaluate_params_loso_auc(X, y, study_ids, scaler_name, params, model_type='xgboost'):
    """Evaluate model using leave-one-study-out protocol."""
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

        scaler = RobustScaler() if scaler_name == 'robust' else StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        model = create_model(model_type, params, class_weight=scale_pos_weight)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))

    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def train_model(
    X,
    y,
    feature_names,
    subject_ids,
    study_ids,
    table_mode='subject_mean',
    scaler_name='robust',
    max_combos=80,
    selection_target='cv_auc',
    save_model=True,
    update_deploy_bundle=True,
    generate_shap=True,
):
    split_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(split_cv.split(X, y, groups=subject_ids))

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = subject_ids[train_idx]

    use_robust_scaler = scaler_name == 'robust'
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    if set(subject_ids[train_idx]).intersection(set(subject_ids[test_idx])):
        raise RuntimeError("Subject leakage detected between train and test splits")

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
    idx = np.random.choice(len(param_combos), min(max_combos, len(param_combos)), replace=False)
    param_combos = [param_combos[i] for i in idx]

    print(f"Searching {len(param_combos)} hyperparameter combinations...")
    for i, params in enumerate(param_combos):
        if selection_target == 'loso_auc':
            mean_score = _evaluate_params_loso_auc(X, y, study_ids, scaler_name, params)
        else:
            mean_score = _evaluate_params_auc(X_train, y_train, groups_train, params)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            if (i + 1) % 20 == 0:
                metric_lbl = 'LOSO AUC' if selection_target == 'loso_auc' else 'CV AUC'
                print(f"  [{i+1}/{len(param_combos)}] Best {metric_lbl}: {best_score:.4f}")

    print(f"\nBest params: {best_params}")
    metric_lbl = 'LOSO AUC-ROC' if selection_target == 'loso_auc' else 'CV AUC-ROC'
    print(f"Best {metric_lbl}: {best_score:.4f}")

    # Train final model with best params
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        **best_params,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    holdout_metrics = _compute_metrics(y_test, y_pred, y_prob)
    holdout_cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Set Results:")
    print(f"Accuracy:          {holdout_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {holdout_metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {holdout_metrics['precision']:.4f}")
    print(f"Recall:            {holdout_metrics['recall']:.4f}")
    print(f"F1 Score:          {holdout_metrics['f1']:.4f}")
    print(f"PR-AUC:            {holdout_metrics['pr_auc']:.4f}")
    print(f"AUC-ROC:           {holdout_metrics['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:\n{holdout_cm}")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'PD']))

    # Leave-one-study-out evaluation for cross-cohort robustness.
    loso_results = []
    print("\nLeave-One-Study-Out Evaluation:")
    for held_out_study in sorted(np.unique(study_ids)):
        test_mask = study_ids == held_out_study
        train_mask = ~test_mask

        X_loso_train_raw = X[train_mask]
        y_loso_train = y[train_mask]
        X_loso_test_raw = X[test_mask]
        y_loso_test = y[test_mask]

        if len(np.unique(y_loso_train)) < 2 or len(np.unique(y_loso_test)) < 2:
            print(f"  {held_out_study}: skipped (single-class split)")
            continue

        loso_scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        X_loso_train = loso_scaler.fit_transform(X_loso_train_raw)
        X_loso_test = loso_scaler.transform(X_loso_test_raw)

        loso_n_pos = int(np.sum(y_loso_train == 1))
        loso_n_neg = int(np.sum(y_loso_train == 0))
        loso_spw = (loso_n_neg / loso_n_pos) if loso_n_pos > 0 else 1.0

        loso_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=loso_spw,
            **best_params,
        )
        loso_model.fit(X_loso_train, y_loso_train)
        y_loso_pred = loso_model.predict(X_loso_test)
        y_loso_prob = loso_model.predict_proba(X_loso_test)[:, 1]
        loso_metrics = _compute_metrics(y_loso_test, y_loso_pred, y_loso_prob)
        loso_metrics['study'] = held_out_study
        loso_metrics['n_test'] = int(np.sum(test_mask))
        loso_results.append(loso_metrics)

        auc_text = f"{loso_metrics['auc_roc']:.4f}" if loso_metrics['auc_roc'] is not None else "N/A"
        print(
            f"  {held_out_study}: AUC={auc_text}, "
            f"BalancedAcc={loso_metrics['balanced_accuracy']:.4f}, "
            f"Recall={loso_metrics['recall']:.4f}"
        )

    if loso_results:
        loso_auc = [r['auc_roc'] for r in loso_results if r['auc_roc'] is not None]
        loso_balacc = [r['balanced_accuracy'] for r in loso_results]
        print("\nLOSO Summary:")
        if loso_auc:
            print(f"  Mean AUC-ROC:        {np.mean(loso_auc):.4f}")
            print(f"  AUC-ROC Std:         {np.std(loso_auc):.4f}")
        print(f"  Mean Balanced Acc:   {np.mean(loso_balacc):.4f}")
        print(f"  Balanced Acc Std:    {np.std(loso_balacc):.4f}")

    if generate_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.tight_layout()
            shap_path = os.path.join(MODEL_DIR, f'shap_gait_summary_{table_mode}_{scaler_name}.png')
            plt.savefig(shap_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to {shap_path}")
        except Exception as e:
            print(f"SHAP plot skipped: {e}")

    model_path = None
    deploy_path = os.path.join(MODEL_DIR, 'gait_model.pkl')
    if save_model:
        bundle = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'table_mode': table_mode,
            'scaler_name': scaler_name,
        }
        model_path = os.path.join(MODEL_DIR, f'gait_model_{table_mode}_{scaler_name}.pkl')
        joblib.dump(bundle, model_path)
        print(f"Model saved to {model_path}")

        # Keep production path compatible with current inference schema.
        if update_deploy_bundle and table_mode == 'subject_mean':
            shutil.copyfile(model_path, deploy_path)
            print(f"Deploy model updated at {deploy_path}")

    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'table_mode': table_mode,
        'selection_metric': 'auc_roc',
        'selection_target': selection_target,
        'scaler': scaler.__class__.__name__,
        'scaler_name': scaler_name,
        'best_params': best_params,
        'best_cv_auc_roc': best_score,
        'holdout': {
            **holdout_metrics,
            'confusion_matrix': holdout_cm.tolist(),
            'n_test': int(len(y_test)),
        },
        'loso': loso_results,
        'model_path': model_path,
        'deploy_model_path': deploy_path if (save_model and update_deploy_bundle and table_mode == 'subject_mean') else None,
    }
    report_path = os.path.join(MODEL_DIR, f'gait_training_report_{table_mode}_{scaler_name}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Training report saved to {report_path}")

    if save_model and table_mode == 'subject_mean':
        latest_report = os.path.join(MODEL_DIR, 'gait_training_report.json')
        shutil.copyfile(report_path, latest_report)
        print(f"Latest report updated at {latest_report}")

    return report


def run_architecture_comparison(table_mode='subject_mean', scaler_name='robust', max_combos=80):
    """Compare different model architectures on the same data using LOSO evaluation."""
    model_types = ['xgboost', 'catboost', 'lightgbm', 'histgradient', 'randomforest', 'extratrees']
    
    X, y, feature_names, subject_ids, study_ids = build_training_table(mode=table_mode)
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON: Testing 6 Model Families")
    print("=" * 70)
    print(f"Data: {table_mode} | Scaler: {scaler_name} | Data shape: {X.shape}")
    print(f"Testing {len(model_types)} architectures with LOSO evaluation\n")
    
    architecture_results = []
    default_params = {
        'xgboost': {}, 'catboost': {}, 'lightgbm': {},
        'histgradient': {}, 'randomforest': {}, 'extratrees': {},
    }
    
    for model_type in model_types:
        print(f"\n{'─' * 70}")
        print(f"Testing: {model_type.upper()}")
        print(f"{'─' * 70}")
        
        studies = sorted(np.unique(study_ids))
        loso_aucs = []
        loso_details = []
        
        for held_out_study in studies:
            test_mask = study_ids == held_out_study
            train_mask = ~test_mask
            
            X_train_raw = X[train_mask]
            y_train = y[train_mask]
            X_test_raw = X[test_mask]
            y_test = y[test_mask]
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            scaler = RobustScaler() if scaler_name == 'robust' else StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
            
            n_pos = int(np.sum(y_train == 1))
            n_neg = int(np.sum(y_train == 0))
            scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            
            model = create_model(model_type, default_params[model_type], class_weight=scale_pos_weight)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            loso_aucs.append(auc)
            
            y_pred = model.predict(X_test)
            recall = recall_score(y_test, y_pred)
            n_test = len(y_test)
            
            loso_details.append({
                'study': held_out_study,
                'auc_roc': float(auc),
                'recall': float(recall),
                'n_test': int(n_test),
            })
            print(f"  {held_out_study}: AUC={auc:.4f}, Recall={recall:.4f}, n={n_test}")
        
        if loso_aucs:
            mean_loso = float(np.mean(loso_aucs))
            std_loso = float(np.std(loso_aucs))
            print(f"  Mean LOSO AUC: {mean_loso:.4f} +/- {std_loso:.4f}")
        else:
            mean_loso = None
            std_loso = None
            print(f"  Mean LOSO AUC: N/A")
        
        architecture_results.append({
            'model_type': model_type,
            'mean_loso_auc': mean_loso,
            'std_loso_auc': std_loso,
            'loso_details': loso_details,
        })
    
    architecture_results.sort(
        key=lambda r: -1e9 if r['mean_loso_auc'] is None else -r['mean_loso_auc']
    )
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE RANKING (by LOSO mean AUC)")
    print("=" * 70)
    for rank, result in enumerate(architecture_results, start=1):
        auc_str = f"{result['mean_loso_auc']:.4f} +/- {result['std_loso_auc']:.4f}" if result['mean_loso_auc'] is not None else "N/A"
        print(f"  {rank}. {result['model_type']:20s} LOSO AUC: {auc_str}")
    
    summary_path = os.path.join(MODEL_DIR, 'gait_architecture_comparison.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'architecture_comparison': architecture_results}, f, indent=2)
    print(f"\nArchitecture comparison saved to {summary_path}")
    
    return architecture_results


def run_benchmark(max_combos=60):
    configs = [
        {'table_mode': 'subject_mean', 'scaler_name': 'robust'},
        {'table_mode': 'trial', 'scaler_name': 'robust'},
        {'table_mode': 'subject_hierarchical', 'scaler_name': 'robust'},
        {'table_mode': 'subject_mean', 'scaler_name': 'standard'},
    ]

    benchmark_results = []
    for i, cfg in enumerate(configs, start=1):
        print("\n" + "-" * 60)
        print(f"Benchmark run {i}/{len(configs)}: {cfg['table_mode']} + {cfg['scaler_name']}")
        print("-" * 60)
        X, y, feature_names, subject_ids, study_ids = build_training_table(mode=cfg['table_mode'])
        report = train_model(
            X,
            y,
            feature_names,
            subject_ids,
            study_ids,
            table_mode=cfg['table_mode'],
            scaler_name=cfg['scaler_name'],
            max_combos=max_combos,
            selection_target='cv_auc',
            save_model=True,
            update_deploy_bundle=(cfg['table_mode'] == 'subject_mean' and cfg['scaler_name'] == 'robust'),
            generate_shap=False,
        )

        loso_auc = [row['auc_roc'] for row in report['loso'] if row['auc_roc'] is not None]
        report['mean_loso_auc_roc'] = float(np.mean(loso_auc)) if loso_auc else None
        report['std_loso_auc_roc'] = float(np.std(loso_auc)) if loso_auc else None
        benchmark_results.append(report)

    benchmark_results.sort(
        key=lambda r: (
            -1e9 if r['mean_loso_auc_roc'] is None else -r['mean_loso_auc_roc'],
            -r['holdout']['auc_roc'] if r['holdout']['auc_roc'] is not None else 1e9,
        )
    )

    print("\nBenchmark Ranking (by LOSO mean AUC):")
    for rank, row in enumerate(benchmark_results, start=1):
        loso_mean = 'N/A' if row['mean_loso_auc_roc'] is None else f"{row['mean_loso_auc_roc']:.4f}"
        holdout_auc = 'N/A' if row['holdout']['auc_roc'] is None else f"{row['holdout']['auc_roc']:.4f}"
        print(
            f"  {rank}. {row['table_mode']} + {row['scaler_name']} | "
            f"LOSO AUC={loso_mean} | Holdout AUC={holdout_auc}"
        )

    summary_path = os.path.join(MODEL_DIR, 'gait_experiment_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'experiments': benchmark_results}, f, indent=2)
    print(f"Benchmark summary saved to {summary_path}")

    return benchmark_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train gait model with strict evaluation options.')
    parser.add_argument(
        '--table-mode',
        default='subject_mean',
        choices=['subject_mean', 'trial', 'subject_hierarchical'],
        help='Data table strategy for training.',
    )
    parser.add_argument(
        '--scaler',
        default='robust',
        choices=['robust', 'standard'],
        help='Feature scaler used before training.',
    )
    parser.add_argument(
        '--max-combos',
        type=int,
        default=80,
        help='Maximum hyperparameter combinations sampled from the search grid.',
    )
    parser.add_argument(
        '--selection-target',
        default='cv_auc',
        choices=['cv_auc', 'loso_auc'],
        help='Hyperparameter selection target: group-CV AUC or leave-one-study-out AUC.',
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run comparative experiments across table modes/scalers.',
    )
    parser.add_argument(
        '--architecture-comparison',
        action='store_true',
        help='Compare different model architectures (XGBoost, CatBoost, LightGBM, etc.).',
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Training Gait Model on PhysioNet Dataset")
    print("=" * 60)

    if args.benchmark:
        run_benchmark(max_combos=args.max_combos)
        print("\nDone!")
        sys.exit(0)
    
    if args.architecture_comparison:
        run_architecture_comparison(
            table_mode=args.table_mode,
            scaler_name=args.scaler,
            max_combos=args.max_combos,
        )
        print("\nDone!")
        sys.exit(0)

    X, y, feature_names, subject_ids, study_ids = build_training_table(mode=args.table_mode)
    train_model(
        X,
        y,
        feature_names,
        subject_ids,
        study_ids,
        table_mode=args.table_mode,
        scaler_name=args.scaler,
        max_combos=args.max_combos,
        selection_target=args.selection_target,
        save_model=True,
        update_deploy_bundle=True,
        generate_shap=True,
    )
    print("\nDone!")
