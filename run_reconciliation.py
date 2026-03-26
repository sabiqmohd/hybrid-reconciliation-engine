"""
Financial Reconciliation System
================================
End-to-end pipeline: loads transaction data, matches records using
rule-based and ML approaches, runs a learning loop, and exports results.

Usage:
    python run_reconciliation.py
"""

import os
import pandas as pd
from data_loader import load_and_clean_data
from matcher import unique_amount_matching
from ml_matcher import ml_match_remaining
from learning_loop import (collect_all_matches, create_training_data,train_model, ml_rescore, compare_performance)

BANK_PATH = "data/bank_statements.csv"
REGISTER_PATH = "data/check_register.csv"
OUTPUT_DIR = "output"


def run():
    """Runs the full reconciliation pipeline end to end."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ── Phase 1: Load and preprocess ────────────────────────────────
    bank, reg = load_and_clean_data(BANK_PATH, REGISTER_PATH)
    print(f"      Bank records:     {len(bank)}")
    print(f"      Register records: {len(reg)}")

    # ── Phase 2: Unique amount matching ────────────────────────────
    unique_matches = unique_amount_matching(bank, reg)
    matched_bank_ids = set(unique_matches['transaction_id_bank'])
    matched_reg_ids = set(unique_matches['transaction_id_reg'])
    print(f"      Matched: {len(unique_matches)} pairs")

    # ── Phase 3: ML similarity matching for the rest ───────────────
    remaining_bank = len(bank) - len(matched_bank_ids)
    remaining_reg = len(reg) - len(matched_reg_ids)

    ml_results = ml_match_remaining(bank, reg, matched_bank_ids, matched_reg_ids)

    ml_matched = ml_low = ml_none = pd.DataFrame()
    if not ml_results.empty:
        ml_matched = ml_results[ml_results['status'] == 'MATCHED']
        ml_low = ml_results[ml_results['status'] == 'LOW_CONFIDENCE']
        ml_none = ml_results[ml_results['status'] == 'NO_CANDIDATE']
        print(f"      Matched:        {len(ml_matched)}")
        print(f"      Low confidence: {len(ml_low)}")
        print(f"      No candidate:   {len(ml_none)}")

    # ── Phase 4: Learning loop ─────────────────────────────────────
    all_matches = collect_all_matches(bank, reg, unique_matches, ml_results)
    labeled = create_training_data(all_matches, bank, reg, high_thresh=0.9)

    pos_count = int((labeled['label'] == 1).sum())
    neg_count = int((labeled['label'] == 0).sum())

    model, X, y = train_model(labeled)
    metrics = None

    if model is not None:
        rescored = ml_rescore(all_matches, model)
        metrics = compare_performance(labeled, model)
        print(f"      Model trained successfully")
    else:
        rescored = all_matches.copy()
        rescored['ml_confidence'] = rescored['rule_confidence']
        print(f"      Skipped — not enough labeled data")

    # ── Phase 5: Export results ────────────────────────────────────
    final_rows = []

    for _, row in unique_matches.iterrows():
        final_rows.append({
            'bank_transaction_id': row['transaction_id_bank'],
            'register_transaction_id': row['transaction_id_reg'],
            'amount': row['rounded_amount'],
            'confidence_score': row['confidence_score'],
            'match_method': 'unique_amount',
            'status': 'MATCHED',
        })

    if not ml_results.empty:
        for _, row in ml_results.iterrows():
            final_rows.append({
                'bank_transaction_id': row['transaction_id_bank'],
                'register_transaction_id': row.get('transaction_id_reg'),
                'amount': row.get('amount_diff'),
                'confidence_score': row['confidence'],
                'match_method': 'ml_similarity',
                'status': row['status'],
            })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "matched_transactions.csv"), index=False)

    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'method'
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"))

    # ── Summary ────────────────────────────────────────────────────
    total = len(bank)
    matched_total = len(unique_matches) + len(ml_matched)

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total transactions:    {total}")
    print(f"  Matched (high conf):   {matched_total}")
    print(f"  Low confidence:        {len(ml_low)}")
    print(f"  Unmatched:             {len(ml_none)}")
    print(f"  Match rate:            {matched_total/total*100:.1f}%")

    if metrics:
        print(f"\n  {'Metric':<20} {'Rule-Based':>12} {'ML-Based':>12}")
        print(f"  {'-'*48}")
        for m in ['precision', 'recall', 'f1']:
            print(f"  {m.capitalize():<20} {metrics['rule'][m]:>12.4f} {metrics['ml'][m]:>12.4f}")

    print(f"\n  Output saved to: {OUTPUT_DIR}/")
    print(f"    - matched_transactions.csv")
    print(f"    - evaluation_metrics.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
