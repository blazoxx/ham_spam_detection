# Spam Email Detector

AI-powered spam email classification system using NLP and Machine Learning.

## Features

- Spam/Ham detection
- TF-IDF vectorization
- Naive Bayes classifier
- Confidence score for predictions
- Confusion matrix and ROC visualizations

## Tech stack

- Python 3.8+
- scikit-learn, pandas, nltk, matplotlib, seaborn

## Quick start

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Command Prompt
.\.venv\Scripts\activate.bat
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare dataset:

- Place your CSV at [data/spam_ham_dataset.csv](data/spam_ham_dataset.csv). The expected columns are `label` and `text`.

4. Train the model (from project root):

```bash
python app/train.py
```

5. Evaluate the trained model (generates visuals in `outputs/`):

```bash
python app/evaluate.py
```

6. Interactive prediction (CLI):

```bash
python main.py
```

## Files of interest

- [app/preprocess.py](app/preprocess.py): text cleaning utilities
- [app/train.py](app/train.py): training script (saves models to `models/`)
- [app/evaluate.py](app/evaluate.py): evaluation and plotting (writes `outputs/`)
- [app/predict.py](app/predict.py): prediction helper used by `main.py`
- [notebooks/experimentation.ipynb](notebooks/experimentation.ipynb): exploratory analysis
- [requirements.txt](requirements.txt): Python dependencies

## Outputs and models

- Trained model: [models/spam_model.pkl](models/spam_model.pkl)
- Vectorizer: [models/vectorizer.pkl](models/vectorizer.pkl)
- Plots: [outputs/confusion_matrix.png](outputs/confusion_matrix.png), [outputs/roc_curve.png](outputs/roc_curve.png)

## Notes & tips

- Run the scripts from the project root so relative paths resolve correctly.
- If you see ModuleNotFoundError, ensure the virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- To reproduce results, keep `random_state=42` in `app/train.py` and `app/evaluate.py`.