

DigitScope compares five machine learning algorithms for recognizing handwritten digits from numerical pixel data. The project includes model evaluation in Jupyter Notebook and a Streamlit application that displays predictions and performance on labeled test data.

**GitHub repository:** https://github.com/devprasanth1/ml-assignment-2  
**Course:** Machine Learning – Assignment 2

## a. Problem statement

The goal is to classify handwritten digits from 0 to 9 using an 8 × 8 grid of pixel values. I trained five classification models on the same dataset and compared their results to understand which approach works best for this task.

## b. Dataset description

- **Source:** [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), using the 1,797-sample subset available through scikit-learn's `load_digits()`.
- **Records:** 1,797
- **Input features:** 64 numeric pixel values per image
- **Target:** `digit`, with 10 classes (0–9)
- **Missing values:** None
- **Train/test split:** 1,347 training records (75%) and 450 test records (25%), stratified by digit, with `random_state=73`

I used a preprocessing pipeline with median imputation and feature scaling. The preprocessing steps were fitted only on the training data to avoid data leakage. The same test set was used to evaluate all five models.

## c. GitHub repository link

https://github.com/devprasanth1/ml-assignment-2

The repository contains the notebook, Python scripts, saved trained models, labeled test data, and the Streamlit application.

## d. Models used and evaluation results

I compared Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).

The metrics are Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). Precision, Recall, and F1 use macro averaging; AUC uses multiclass one-vs-rest macro averaging.

| ML model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9600 | 0.9981 | 0.9607 | 0.9595 | 0.9595 | 0.9557 |
| Decision Tree | 0.8244 | 0.9154 | 0.8240 | 0.8239 | 0.8232 | 0.8051 |
| KNN | 0.9711 | 0.9960 | 0.9721 | 0.9711 | 0.9714 | 0.9680 |
| Gaussian Naive Bayes | 0.7933 | 0.9636 | 0.8395 | 0.7914 | 0.7868 | 0.7759 |
| **Support Vector Machine** | **0.9778** | **0.9996** | **0.9789** | **0.9776** | **0.9778** | **0.9754** |

### Observations

- **Logistic Regression:** A strong baseline, with 96.00% accuracy. It classified most digits correctly but was slightly less accurate than KNN and SVM.
- **Decision Tree:** Reached 82.44% accuracy. Its lower performance suggests that a single tree did not capture the digit patterns as effectively as the stronger models.
- **KNN:** Achieved 97.11% accuracy and performed particularly well on this pixel-based dataset. It was the second-best model overall.
- **Gaussian Naive Bayes:** Had the lowest accuracy, 79.33%. Its independence assumption may be limiting because neighboring pixel values can be related.
- **Support Vector Machine:** Performed best, with 97.78% accuracy and the highest scores across all six evaluation metrics.

**Overall winner: Support Vector Machine (SVM).** It correctly classified 440 of the 450 test samples. These results apply to this particular train/test split and dataset.

## e. Streamlit application

The Streamlit app loads the five saved models and evaluates labeled CSV test data **without retraining**. It provides:

- A CSV upload option and bundled test dataset
- A dropdown to select a model
- All six evaluation metrics
- A confusion matrix and per-digit classification report
- A comparison of the five models and downloadable predictions

The uploaded CSV must include the 64 input features and the `digit` target column.

## How to run the project

From the project folder, run:

```bash
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Open the local app at `http://localhost:8501`. The saved models are in `model/`, the full dataset is in `data/uci_digits.csv`, and the test data is in `test_data.csv`.

**Dataset reference:** Alpaydin, E., and Kaynak, C. (1998). *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49
