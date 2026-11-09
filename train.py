"""Reproducible training script. Run from any folder: python train.py."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from config import (DATA_PATH, TEST_PATH, RESULTS_PATH, MODEL_DIR,
                    FEATURES, TARGET, SEED, TEST_SIZE, MODEL_FILES)
from ml_utils import evaluate, validate_test_csv


def make_models():
    """All five models use the identical median-impute + standardize pipeline."""
    estimators = {
        'Logistic Regression': LogisticRegression(C=1.2, max_iter=1600, random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Gaussian Naive Bayes': GaussianNB(var_smoothing=1e-8),
        'Support Vector Machine': SVC(C=3.0, kernel='rbf', probability=True, random_state=SEED),
    }
    return {
        name: Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
            ('classifier', estimator),
        ])
        for name, estimator in estimators.items()
    }


def main():
    original = validate_test_csv(pd.read_csv(DATA_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        original[FEATURES], original[TARGET],
        test_size=TEST_SIZE, random_state=SEED, stratify=original[TARGET]
    )
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test
    test_frame.to_csv(TEST_PATH, index=False)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True, parents=True)
    outcomes = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        result, y_hat = evaluate(model, X_test, y_test)
        result = {'Model': name, **result}
        outcomes.append(result)
        path = MODEL_DIR / MODEL_FILES[name]
        joblib.dump(model, path, compress=3)
        assert (joblib.load(path).predict(X_test.iloc[:10]) == y_hat[:10]).all()
        print(name.ljust(24), ' '.join(f'{k}={v:.4f}' for k, v in result.items() if k != 'Model'))
    table = pd.DataFrame(outcomes)
    table.to_csv(RESULTS_PATH, index=False)
    print(f'\nData: {len(original)} rows; training {len(X_train)}; test {len(X_test)}')
    print('Saved:', RESULTS_PATH, TEST_PATH, MODEL_DIR)


if __name__ == '__main__':
    main()
