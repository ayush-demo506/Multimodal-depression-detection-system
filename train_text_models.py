import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_DIR = Path('data_for_text')
MODELS_DIR = Path('models/text')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}_comprehensive.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    df = pd.read_csv(path)
    # Combine text_1..text_10 into single field
    text_cols = [f"text_{i}" for i in range(1, 11)]
    df['text_all'] = df[text_cols].fillna('').agg('\n'.join, axis=1)
    return df[['text_all', 'label']]


def train_and_eval(model_name: str, pipeline: Pipeline, X_train, y_train, X_val, y_val, X_test, y_test):
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODELS_DIR / f"{model_name}.joblib")

    def eval_split(split_name, X, y):
        y_pred = pipeline.predict(X)
        acc = accuracy_score(y, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
        return {
            'accuracy': round(float(acc), 4),
            'precision': round(float(p), 4),
            'recall': round(float(r), 4),
            'f1': round(float(f1), 4),
            'report': classification_report(y, y_pred, target_names=['Not Depressed', 'Depressed'], zero_division=0)
        }

    return {
        'val': eval_split('val', X_val, y_val),
        'test': eval_split('test', X_test, y_test)
    }


def main():
    print('Loading datasets...')
    train_df = load_split('train')
    val_df = load_split('val')
    test_df = load_split('test')

    X_train, y_train = train_df['text_all'].tolist(), train_df['label'].astype(int).values
    X_val, y_val = val_df['text_all'].tolist(), val_df['label'].astype(int).values
    X_test, y_test = test_df['text_all'].tolist(), test_df['label'].astype(int).values

    print('Training Model A: TF-IDF + LogisticRegression')
    model_a = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), lowercase=True)),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=None))
    ])
    results_a = train_and_eval('tfidf_logreg', model_a, X_train, y_train, X_val, y_val, X_test, y_test)

    print('Training Model B: TF-IDF + LinearSVC')
    model_b = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), lowercase=True)),
        ('clf', LinearSVC())
    ])
    results_b = train_and_eval('tfidf_linearsvc', model_b, X_train, y_train, X_val, y_val, X_test, y_test)

    summary = {
        'models_dir': str(MODELS_DIR.resolve()),
        'models': {
            'tfidf_logreg': results_a,
            'tfidf_linearsvc': results_b
        }
    }
    out_path = MODELS_DIR / 'models_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('\nSaved models to:', MODELS_DIR)
    print('Report ->', out_path)


if __name__ == '__main__':
    main()


