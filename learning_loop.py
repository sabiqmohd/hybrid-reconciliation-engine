"""
Learning Loop
Trains a simple classifier on validated matches and re-scores predictions.
Shows how the system improves as more confirmed pairs become available.
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict

FEATURE_COLS = ['amount_diff', 'date_diff_days', 'desc_similarity', 'type_match']


def collect_all_matches(bank_df, reg_df, unique_matches, ml_results):
    """Merges results from both matching stages into one unified feature table."""
    rows = []

    for _, m in unique_matches.iterrows():
        rows.append({
            'transaction_id_bank': m['transaction_id_bank'],
            'transaction_id_reg': m['transaction_id_reg'],
            'amount_diff': 0.0,
            'date_diff_days': abs(m['date_diff_days']),
            'desc_similarity': m['desc_similarity'],
            'type_match': 1 if m['type_match'] else 0,
            'rule_confidence': m['confidence_score'],
            'source': 'unique_match',
        })

    for _, m in ml_results.iterrows():
        if m['status'] == 'NO_CANDIDATE':
            continue
        rows.append({
            'transaction_id_bank': m['transaction_id_bank'],
            'transaction_id_reg': m['transaction_id_reg'],
            'amount_diff': m.get('amount_diff', 0.0) or 0.0,
            'date_diff_days': abs(m.get('date_diff_days', 0) or 0),
            'desc_similarity': m.get('desc_similarity', 0.0) or 0.0,
            'type_match': 1,
            'rule_confidence': m['confidence'],
            'source': 'ml_match',
        })

    return pd.DataFrame(rows)


def create_training_data(all_matches_df, bank_df, reg_df, high_thresh=0.9, n_negatives=1, seed=42):
    """
    Builds a labeled dataset for the classifier.
    Positives come from high-confidence matches. Negatives are synthetic —
    deliberately wrong pairings so the model sees both classes.
    """
    rng = np.random.RandomState(seed)
    df = all_matches_df.copy()

    positives = df[df['rule_confidence'] >= high_thresh].copy()
    positives['label'] = 1

    known_pairs = set(zip(df['transaction_id_bank'], df['transaction_id_reg']))
    reg_ids = reg_df['transaction_id'].values
    neg_rows = []

    for _, pos_row in positives.iterrows():
        b_id = pos_row['transaction_id_bank']
        b_data = bank_df[bank_df['transaction_id'] == b_id].iloc[0]

        generated, attempts = 0, 0
        while generated < n_negatives and attempts < 20:
            wrong_r_id = rng.choice(reg_ids)
            attempts += 1
            if (b_id, wrong_r_id) in known_pairs:
                continue

            r_data = reg_df[reg_df['transaction_id'] == wrong_r_id].iloc[0]
            amt_diff = abs(b_data['rounded_amount'] - r_data['rounded_amount'])
            date_diff = abs((b_data['date_day'] - r_data['date_day']).days)
            desc_sim = SequenceMatcher(
                None, str(b_data['normalized_description']), str(r_data['normalized_description'])
            ).ratio()
            type_match = 1 if b_data['type'] == r_data['type'] else 0

            neg_rows.append({
                'transaction_id_bank': b_id,
                'transaction_id_reg': wrong_r_id,
                'amount_diff': round(amt_diff, 2),
                'date_diff_days': date_diff,
                'desc_similarity': round(desc_sim, 4),
                'type_match': type_match,
                'rule_confidence': 0.0,
                'source': 'synthetic_negative',
                'label': 0,
            })
            generated += 1

    negatives = pd.DataFrame(neg_rows)
    return pd.concat([positives, negatives], ignore_index=True)


def train_model(labeled_df):
    """Fits a Logistic Regression on the labeled pairs. Returns the trained model, features, and labels."""
    train_data = labeled_df[labeled_df['label'].isin([0, 1])].copy()

    if len(train_data) < 5:
        return None, None, None

    X = train_data[FEATURE_COLS].values
    y = train_data['label'].values

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model, X, y


def ml_rescore(all_matches_df, model):
    """Replaces rule-based confidence with the trained model's predicted probability."""
    df = all_matches_df.copy()
    X = df[FEATURE_COLS].values
    df['ml_confidence'] = model.predict_proba(X)[:, 1].round(4)
    return df


def compare_performance(labeled_df, model):
    """
    Side-by-side comparison of rule-based vs ML-based scoring.
    Uses cross-validation on the labeled set to get fair ML metrics.
    """
    train_data = labeled_df[labeled_df['label'].isin([0, 1])].copy()
    X = train_data[FEATURE_COLS].values
    y = train_data['label'].values

    rule_preds = (train_data['rule_confidence'] >= 0.5).astype(int).values
    p_r, r_r, f1_r, _ = precision_recall_fscore_support(y, rule_preds, average='binary', zero_division=0)

    n_splits = min(5, len(train_data))
    if n_splits >= 2 and len(np.unique(y)) > 1:
        ml_preds = cross_val_predict(
            LogisticRegression(random_state=42, max_iter=1000), X, y, cv=n_splits, method='predict'
        )
    else:
        ml_preds = model.predict(X)

    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y, ml_preds, average='binary', zero_division=0)

    return {
        'rule': {'precision': round(p_r, 4), 'recall': round(r_r, 4), 'f1': round(f1_r, 4)},
        'ml': {'precision': round(p_m, 4), 'recall': round(r_m, 4), 'f1': round(f1_m, 4)},
    }
