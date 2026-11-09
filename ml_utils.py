"""Reusable validation, evaluation, and class-specific diagnostics.

All metrics are computed on held-out labels; probabilities drive multi-class AUC.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, confusion_matrix, classification_report,
)
from config import FEATURES, TARGET, LABELS

METRIC_COLUMNS = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']


def validate_test_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept a numeric feature matrix and integer labels 0..9, in any column order."""
    if frame.empty:
        raise ValueError('The uploaded file is empty.')
    absent = [col for col in [*FEATURES, TARGET] if col not in frame.columns]
    if absent:
        short = ', '.join(absent[:8]) + (' ...' if len(absent) > 8 else '')
        raise ValueError(f'Missing {len(absent)} required columns: {short}')
    clean = frame.loc[:, [*FEATURES, TARGET]].copy()
    for col in FEATURES:
        clean[col] = pd.to_numeric(clean[col], errors='coerce')
    if clean[FEATURES].isna().any().any():
        raise ValueError('Feature columns must contain finite numbers and no blank values.')
    if not np.isfinite(clean[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Feature columns contain non-finite values.')
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors='coerce')
    if clean[TARGET].isna().any() or not clean[TARGET].isin(LABELS).all():
        raise ValueError('The digit column must contain numeric labels from 0 through 9.')
    clean[TARGET] = clean[TARGET].astype(int)
    return clean


def evaluate(model, X, y):
    """Comparable one-vs-rest macro AUC and macro P/R/F1 over all ten classes."""
    y_hat = model.predict(X)
    scores = model.predict_proba(X)
    out = {
        'Accuracy': float(accuracy_score(y, y_hat)),
        'Precision': float(precision_score(y, y_hat, average='macro', labels=LABELS, zero_division=0)),
        'Recall': float(recall_score(y, y_hat, average='macro', labels=LABELS, zero_division=0)),
        'F1': float(f1_score(y, y_hat, average='macro', labels=LABELS, zero_division=0)),
        'MCC': float(matthews_corrcoef(y, y_hat)),
    }
    # AUC is undefined when the evaluation CSV omits some classes.
    # All other metrics remain calculable; the UI signals that AUC is N/A.
    try:
        if set(np.unique(y)) != set(LABELS):
            raise ValueError('All ten classes needed for comparable macro AUC')
        out['AUC'] = float(roc_auc_score(y, scores, multi_class='ovr', average='macro', labels=LABELS))
    except ValueError:
        out['AUC'] = float('nan')
    return out, y_hat


def diagnostics(y, y_hat):
    return (
        confusion_matrix(y, y_hat, labels=LABELS),
        classification_report(y, y_hat, labels=LABELS,
                              target_names=[str(v) for v in LABELS],
                              zero_division=0, output_dict=True),
    )
