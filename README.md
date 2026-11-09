# DigitScope: Ten-Class Handwritten Digit Classification

> **Independent-learning project starter.** This project was prepared to help you implement and explain an assignment. Review and customize the work, execute it in the required environment, and use only your own screenshots, URLs, and genuine development history. Do not submit it without reviewing the code and results.

**Repository:** `ADD_YOUR_OWN_GITHUB_REPOSITORY_URL`  
**Live Streamlit application:** `ADD_YOUR_OWN_STREAMLIT_URL`  
**Course:** Machine Learning, Assignment 2

## a. Problem statement

The goal is to identify one of ten handwritten digits (0-9) using 64 numerical features representing an 8 x 8 handwritten character grid. We compare five established supervised classifiers on the *same untouched, stratified 25% holdout set*, then inspect the metrics and the kinds of digits each approach misclassifies.

## b. Dataset description

- **Origin:** [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), by E. Alpaydin and C. Kaynak (1998); DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49).
- **Exact subset used:** 1,797 labeled records distributed with `sklearn.datasets.load_digits()`. This is a copy of the *original UCI test subset*, not the complete 5,620-row original UCI collection.
- **Feature count:** 64 pixel-intensity aggregate features named `pixel_r0_c0` ... `pixel_r7_c7` (integers 0 to 16).
- **Target:** `digit`, one of the ten integer labels 0-9.
- **Data-quality check:** 0 missing fields and 0 duplicate records in the supplied subset.
- **Data split:** 1,347 training records / 450 held-out test records; stratified 75/25; `random_state=73`.
- **Reuse:** The upstream UCI dataset is released under CC BY 4.0. The original dataset credit is retained in `DATASET_SOURCE.txt`.

This task measures performance *within this subset*, not a deployment-ready optical character recognition system for arbitrary handwriting or photographs.

## Preprocessing and experimental design

1. Read `data/uci_digits.csv`. Validate feature names, numeric values, and the target.
2. Call `train_test_split(..., stratify=y, test_size=0.25, random_state=73)` only once for a consistent evaluation.
3. Use the **same** pipeline structure for all models: median imputation, then `StandardScaler`, then the classifier. These transforms are fitted on **training data only**, preventing test-set leakage.
4. Train the five classifiers and persist the **entire fitted pipelines** as `.joblib` files.
5. Save the held-out, labeled `test_data.csv` and evaluate every model using it.
6. Streamlit loads persisted pipelines; the application does not fit or retrain on uploads.

## c. GitHub repository link

Replace this line with your actual, working public repository URL after publishing.

## d. Models used and evaluation

**Definitions:** Accuracy = correct fraction; AUC = multiclass one-vs-rest macro average of predicted probabilities; Precision / Recall / F1 = macro averages over labels 0-9; MCC = multiclass Matthews correlation coefficient. For class-limited uploaded files, the app reports AUC as unavailable rather than presenting a misleading value.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9600 | 0.9982 | 0.9607 | 0.9595 | 0.9595 | 0.9557 |
| Decision Tree | 0.8244 | 0.9154 | 0.8240 | 0.8239 | 0.8232 | 0.8051 |
| KNN | 0.9711 | 0.9960 | 0.9721 | 0.9711 | 0.9714 | 0.9680 |
| Gaussian Naive Bayes | 0.7933 | 0.9636 | 0.8395 | 0.7914 | 0.7868 | 0.7759 |
| Support Vector Machine | 0.9778 | 0.9996 | 0.9789 | 0.9776 | 0.9778 | 0.9754 |

The table reports **executed reference results** from `python train.py` with scikit-learn 1.8.0 on the bundled 450-row holdout file. Re-run training in your own environment and update the table if your saved-model results differ.

### Model observations based on the confusion matrices

| Model | Observations on held-out examples |
|---|---|
| Logistic Regression | 18 of 450 test digits were misclassified. Per-digit recall ranged from 0.860 (digit 8) to 1.000 (digit 0). Macro F1 = 0.9595; MCC = 0.9557. |
| Decision Tree | 79 of 450 test digits were misclassified. Per-digit recall ranged from 0.674 (digit 8) to 0.956 (digit 0). Macro F1 = 0.8232; MCC = 0.8051. |
| KNN | 13 of 450 test digits were misclassified. Per-digit recall ranged from 0.933 (digit 4) to 1.000 (digit 0). Macro F1 = 0.9714; MCC = 0.9680. |
| Gaussian Naive Bayes | 93 of 450 test digits were misclassified. Per-digit recall ranged from 0.341 (digit 2) to 1.000 (digit 0). Macro F1 = 0.7868; MCC = 0.7759. |
| Support Vector Machine | 10 of 450 test digits were misclassified. Per-digit recall ranged from 0.933 (digit 4) to 1.000 (digit 0). Macro F1 = 0.9778; MCC = 0.9754. |

**Overall winner:** Support Vector Machine on this split, with accuracy 0.9778, one-vs-rest macro AUC 0.9996, macro F1 0.9778, and MCC 0.9754. KNN is a close runner-up in accuracy and F1. These results are specific to the evaluation protocol above; model superiority cannot be guaranteed on new collections.

## e. Streamlit application

The app offers a CSV upload (or bundled holdout), saved-model dropdown, six performance metrics, confusion matrix, per-digit classification report, predicted-label download, all-model comparison table, and a view of one 8 x 8 digit matrix. An uploaded evaluation file **must** contain the 64 feature columns and a labeled `digit` target. Column order can vary. **No `.fit()` call is used in `app.py`.**

## Repository contents

```text
DigitScope_ML_Assignment2_Starter/
├── app.py                       # prediction-only Streamlit interface
├── train.py                     # trains and exports all five pipelines
├── ml_utils.py                  # validation + consistent metric definitions
├── config.py                    # shared paths, label and feature schema
├── ML_Assignment2_DigitScope.ipynb
├── requirements.txt
├── requirements-report.txt
├── generate_report.py            # composes PDF using your verified links/screenshot
├── QUICK_START.txt
├── README.md
├── test_data.csv                # 450 held-out labeled examples
├── data/uci_digits.csv          # full 1,797-row dataset subset
├── model/                       # five fitted .joblib pipelines
├── results/metrics.csv
├── DATASET_SOURCE.txt
├── docs/REPORT_TEMPLATE.pdf     # requires your own proof/links
└── tests/test_project.py
```

## How to reproduce

Python 3.11 or 3.12 is recommended. In a terminal opened inside the repository:

```bash
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Open the local Streamlit address (usually `http://localhost:8501`) and compare each model on bundled `test_data.csv`.

For tests (if pytest is installed), run `python -m pytest -q`.

**To create your final PDF:** install optional report dependencies with `python -m pip install -r requirements-report.txt`, then use `python generate_report.py --help` for parameters. Supply your own name, BITS ID, real GitHub and Streamlit URLs, and a genuine BITS Virtual Lab screenshot. `docs/REPORT_TEMPLATE.pdf` is intentionally incomplete; **do not submit the unfilled template**.

## Deployment and lab verification

1. Publish this project to a **new GitHub repository** with your own considered commits. Do not copy another student's commit history.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), deploy branch `main`, entry file `app.py`, using a Python version compatible with `requirements.txt` and the joblib files.
3. Check the deployed app by selecting *all five* models on the bundled test file; the numbers should match the README / `results/metrics.csv` (assuming matching dependencies).
4. Run the assignment **inside BITS Virtual Lab** as faculty directs; capture an authentic screenshot with the browser/terminal and relevant results.
5. Put the repository link, live link, your real lab screenshot, and this README content into one final PDF in that order.

**Academic integrity note:** The official assignment permits AI for learning support, not direct copy/paste submissions. Explain preprocessing choices and metrics in your own words, alter or extend the interface meaningfully, and update observations after executing your version.
