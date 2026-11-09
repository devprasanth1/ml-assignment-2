"""Shared, auditable project constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'uci_digits.csv'
TEST_PATH = ROOT / 'test_data.csv'
MODEL_DIR = ROOT / 'model'
RESULTS_PATH = ROOT / 'results' / 'metrics.csv'
TARGET = 'digit'
SEED = 73
TEST_SIZE = 0.25
FEATURES = [f'pixel_r{r}_c{c}' for r in range(8) for c in range(8)]
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Gaussian Naive Bayes': 'gaussian_naive_bayes.joblib',
    'Support Vector Machine': 'support_vector_machine.joblib',
}
LABELS = list(range(10))
