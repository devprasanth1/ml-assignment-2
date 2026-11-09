"""DigitScope Streamlit benchmark. Inference only: saved pipelines are never fitted here."""
import io
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from config import ROOT, TEST_PATH, MODEL_DIR, MODEL_FILES, FEATURES, TARGET, LABELS
from ml_utils import METRIC_COLUMNS, validate_test_csv, evaluate, diagnostics

st.set_page_config(page_title='DigitScope | ML Benchmark', page_icon='🔎', layout='wide')
st.markdown('''<style>
 .block-container{padding-top:1.65rem;max-width:1280px}
 h1,h2,h3{color:#14304A}
 [data-testid="stMetric"]{background:#eef5f9;border-radius:10px;padding:14px;border:1px solid #d4e3eb}
 .intro{font-size:1rem;color:#4A6175;margin-top:-8px}
 .tag{background:#dff3ef;color:#17685a;border-radius:16px;padding:4px 11px;font-size:13px}
 </style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def fetch_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])

st.title('DigitScope')
st.markdown('<p class="intro">Compare five classifiers on 8 × 8 handwriting features. Ten classes, one untouched test set, zero retraining in this app.</p>', unsafe_allow_html=True)
st.caption('UCI Optical Recognition of Handwritten Digits · scikit-learn 1,797-row subset · Educational model evaluation')

with st.sidebar:
    st.header('Test set')
    uploaded = st.file_uploader('Upload labeled evaluation CSV', type=['csv'])
    sample_csv = TEST_PATH.read_bytes() if TEST_PATH.exists() else b''
    st.download_button('Download test_data.csv', data=sample_csv, file_name='test_data.csv', mime='text/csv', disabled=not bool(sample_csv))
    use_bundled = st.checkbox('Use bundled held-out test data', value=True)
    st.divider()
    chosen = st.selectbox('Model to examine', list(MODEL_FILES))
    st.caption('A valid CSV contains 64 columns pixel_r0_c0 to pixel_r7_c7 and a digit label (0–9). Extra columns are ignored. Predictions come from saved .joblib files.')

try:
    if uploaded is not None:
        frame = validate_test_csv(pd.read_csv(uploaded))
        file_label = uploaded.name
    elif use_bundled:
        frame = validate_test_csv(pd.read_csv(TEST_PATH))
        file_label = 'bundled test_data.csv'
    else:
        st.info('Choose a CSV file or enable the bundled test set to begin.')
        st.stop()
except (ValueError, OSError, pd.errors.ParserError, UnicodeError) as exc:
    st.error(f'Cannot use this CSV: {exc}')
    st.stop()

X, y = frame[FEATURES], frame[TARGET]
st.caption(f'Active evaluation data: {file_label} · {len(frame)} labeled rows · {len(FEATURES)} numeric predictors · {y.nunique()} represented digits')

try:
    model = fetch_model(chosen)
    scores, predictions = evaluate(model, X, y)
except Exception as exc:
    st.error(f'The saved model could not evaluate this file: {exc}')
    st.stop()

eval_tab, compare_tab, inspect_tab = st.tabs(['Model evaluation', 'All-model comparison', 'Inspect a digit'])

with eval_tab:
    st.subheader(chosen)
    cols = st.columns(6)
    for col, metric in zip(cols, METRIC_COLUMNS):
        number = scores[metric]
        col.metric(metric, f'{number:.4f}' if np.isfinite(number) else 'N/A')
    if not np.isfinite(scores['AUC']):
        st.info('AUC is undefined for this file because at least one digit class is missing. The other metrics still use macro averaging over digits 0–9.')
    matrix, report = diagnostics(y, predictions)
    left, right = st.columns([1.05, 0.95], gap='large')
    with left:
        st.markdown('**Confusion matrix**')
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d'
        )
        ax.set_title('Actual vs. predicted digit')
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with right:
        st.markdown('**Per-digit classification report**')
        detail = pd.DataFrame(report).T
        st.dataframe(detail.round(3), use_container_width=True)
    predictions_out = frame.copy()
    predictions_out['predicted_digit'] = predictions
    st.download_button('Download predictions CSV', predictions_out.to_csv(index=False).encode(),
                       'digit_predictions.csv', 'text/csv')

with compare_tab:
    st.subheader('Same uploaded data, five saved pipelines')
    rows = []
    try:
        for name in MODEL_FILES:
            summary, _ = evaluate(fetch_model(name), X, y)
            rows.append({'Model': name, **summary})
    except Exception as exc:
        st.error(f'Comparison could not be calculated: {exc}')
        st.stop()
    score_table = pd.DataFrame(rows).set_index('Model')
    st.dataframe(score_table.style.format('{:.4f}', na_rep='N/A'), use_container_width=True)
    metric_choice = st.selectbox('Compare by', METRIC_COLUMNS, index=4)
    st.bar_chart(score_table[metric_choice].sort_values(), horizontal=True)
    if score_table['MCC'].notna().any():
        st.info('MCC leader on this file: ' + score_table['MCC'].idxmax() +
                '. Compare F1, class-level errors and AUC before choosing a final model.')

with inspect_tab:
    st.subheader('What do the 64 predictors represent?')
    n = st.number_input('Row number in current test set', min_value=0, max_value=len(frame)-1, value=0, step=1)
    pixel_grid = X.iloc[int(n)].to_numpy(dtype=float).reshape(8, 8)
    c1, c2 = st.columns([1, 2])
    with c1:
        fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
        ax2.imshow(pixel_grid, cmap='gray_r', interpolation='nearest', vmin=0, vmax=16)
        ax2.axis('off')
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)
    with c2:
        st.metric('Ground-truth digit', str(y.iloc[int(n)]))
        st.metric(f'Predicted by {chosen}', str(predictions[int(n)]))
        st.caption('Pixel values are 0–16. This visualizes a row already supplied in the CSV; it does not accept new photographs.')

st.caption('Educational demonstration only. All metrics are calculated from the active labeled CSV; the app never updates the fitted models.')
