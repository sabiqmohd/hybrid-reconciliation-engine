import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Config ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATE_WINDOW = 5
AMOUNT_TOLERANCE = 5.0
CONFIDENCE_THRESHOLD = 0.5

W_DESC = 0.40
W_AMT  = 0.30
W_DATE = 0.20
W_TYPE = 0.10


def get_candidates(bank_row, reg_df, date_window=DATE_WINDOW, amount_tol=AMOUNT_TOLERANCE):
    """Filters register transactions to plausible candidates — same type, close amount, close date."""
    type_mask = reg_df['type'] == bank_row['type']
    amt_mask = (reg_df['rounded_amount'] - bank_row['rounded_amount']).abs() <= amount_tol
    date_diff = (bank_row['date_day'] - reg_df['date_day']).dt.days
    date_mask = date_diff.abs() <= date_window
    return reg_df[type_mask & amt_mask & date_mask].copy()


def compute_embeddings(descriptions, model):
    """Encodes description strings into vector embeddings using the sentence transformer."""
    return model.encode(descriptions.tolist(), show_progress_bar=False)


def score_candidate(bank_row, bank_emb, reg_row, reg_emb,
                    max_amt_diff=AMOUNT_TOLERANCE, max_date_diff=DATE_WINDOW):
    """
    Scores a bank-register pair using a weighted combination of:
    description cosine similarity, amount closeness, date closeness, type match.
    """
    desc_sim = float(cosine_similarity(
        bank_emb.reshape(1, -1), reg_emb.reshape(1, -1)
    )[0, 0])
    desc_sim = max(0.0, desc_sim)

    amt_diff = abs(bank_row['rounded_amount'] - reg_row['rounded_amount'])
    amt_score = max(0.0, 1.0 - (amt_diff / max_amt_diff)) if max_amt_diff > 0 else 1.0

    date_diff = abs((bank_row['date_day'] - reg_row['date_day']).days)
    date_score = max(0.0, 1.0 - (date_diff / max_date_diff)) if max_date_diff > 0 else 1.0

    type_score = 1.0 if bank_row['type'] == reg_row['type'] else 0.0

    confidence = W_DESC * desc_sim + W_AMT * amt_score + W_DATE * date_score + W_TYPE * type_score

    return {
        'desc_similarity': round(desc_sim, 4),
        'amount_score': round(amt_score, 4),
        'date_score': round(date_score, 4),
        'type_score': type_score,
        'confidence': round(confidence, 4),
        'amount_diff': round(amt_diff, 2),
        'date_diff_days': date_diff,
    }


def ml_match_remaining(bank_df, reg_df, matched_bank_ids, matched_reg_ids,
                       confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Handles non-unique transactions that couldn't be matched by exact amount.
    Loads embeddings, retrieves candidates, scores each pair, and picks the best match.
    """
    bank_remaining = bank_df[~bank_df['transaction_id'].isin(matched_bank_ids)].copy()
    reg_remaining = reg_df[~reg_df['transaction_id'].isin(matched_reg_ids)].copy()

    if bank_remaining.empty or reg_remaining.empty:
        return pd.DataFrame()

    model = SentenceTransformer(EMBEDDING_MODEL)

    reg_embeddings = compute_embeddings(reg_remaining['normalized_description'].fillna(''), model)
    reg_idx_map = {idx: pos for pos, idx in enumerate(reg_remaining.index)}

    bank_embeddings = compute_embeddings(bank_remaining['normalized_description'].fillna(''), model)
    bank_idx_map = {idx: pos for pos, idx in enumerate(bank_remaining.index)}

    results = []
    matched_reg_ids_round = set()

    for idx, bank_row in bank_remaining.iterrows():
        bank_emb = bank_embeddings[bank_idx_map[idx]]
        candidates = get_candidates(bank_row, reg_remaining)
        candidates = candidates[~candidates['transaction_id'].isin(matched_reg_ids_round)]

        if candidates.empty:
            results.append({
                'transaction_id_bank': bank_row['transaction_id'],
                'transaction_id_reg': None,
                'confidence': 0.0,
                'status': 'NO_CANDIDATE',
            })
            continue

        scored = []
        for c_idx, reg_row in candidates.iterrows():
            reg_emb = reg_embeddings[reg_idx_map[c_idx]]
            scores = score_candidate(bank_row, bank_emb, reg_row, reg_emb)
            scores['transaction_id_reg'] = reg_row['transaction_id']
            scored.append(scores)

        scored_df = pd.DataFrame(scored).sort_values('confidence', ascending=False)
        best = scored_df.iloc[0]

        if best['confidence'] >= confidence_threshold:
            matched_reg_ids_round.add(best['transaction_id_reg'])
            results.append({
                'transaction_id_bank': bank_row['transaction_id'],
                'transaction_id_reg': best['transaction_id_reg'],
                'confidence': best['confidence'],
                'desc_similarity': best['desc_similarity'],
                'amount_diff': best['amount_diff'],
                'date_diff_days': best['date_diff_days'],
                'num_candidates': len(scored_df),
                'status': 'MATCHED',
            })
        else:
            results.append({
                'transaction_id_bank': bank_row['transaction_id'],
                'transaction_id_reg': best['transaction_id_reg'],
                'confidence': best['confidence'],
                'desc_similarity': best.get('desc_similarity', None),
                'amount_diff': best.get('amount_diff', None),
                'date_diff_days': best.get('date_diff_days', None),
                'num_candidates': len(scored_df),
                'status': 'LOW_CONFIDENCE',
            })

    return pd.DataFrame(results)
