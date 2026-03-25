import pandas as pd
from difflib import SequenceMatcher


def string_similarity(a, b):
    """Quick ratio between two strings. Returns 0.0 if either is missing."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a), str(b)).ratio()


def unique_amount_matching(bank_df, reg_df):
    """
    Matches transactions where the amount appears exactly once in both datasets.
    Computes date diff, description similarity, type compatibility for each pair
    and assigns a confidence score. Flags anything suspicious.
    """
    bank_amt_counts = bank_df['rounded_amount'].value_counts()
    reg_amt_counts = reg_df['rounded_amount'].value_counts()

    unique_bank = bank_amt_counts[bank_amt_counts == 1].index
    unique_reg = reg_amt_counts[reg_amt_counts == 1].index
    common_unique = unique_bank.intersection(unique_reg)

    b_unique = bank_df[bank_df['rounded_amount'].isin(common_unique)].copy()
    r_unique = reg_df[reg_df['rounded_amount'].isin(common_unique)].copy()

    matches = pd.merge(b_unique, r_unique, on='rounded_amount', suffixes=('_bank', '_reg'))

    if matches.empty:
        return matches

    matches['type_match'] = matches['type_bank'] == matches['type_reg']
    matches['date_diff_days'] = (matches['date_day_bank'] - matches['date_day_reg']).dt.days
    matches['desc_similarity'] = matches.apply(
        lambda row: string_similarity(
            row['normalized_description_bank'], row['normalized_description_reg']
        ),
        axis=1
    )

    def evaluate_match(row):
        score = 1.0
        flags = []

        if row['date_diff_days'] < 0 or row['date_diff_days'] > 5:
            score -= 0.3
            flags.append(f"Date gap: {row['date_diff_days']} days")

        if not row['type_match']:
            score -= 0.4
            flags.append(f"Type mismatch: {row['type_bank']} vs {row['type_reg']}")

        if row['desc_similarity'] < 0.3:
            score -= 0.2
            flags.append(f"Low desc similarity: {row['desc_similarity']:.2f}")

        return pd.Series(
            [max(0.0, score), ' | '.join(flags) if flags else 'OK'],
            index=['confidence', 'flags']
        )

    evals = matches.apply(evaluate_match, axis=1)
    matches['confidence_score'] = evals['confidence']
    matches['flags'] = evals['flags']

    columns = [
        'transaction_id_bank', 'transaction_id_reg', 'rounded_amount',
        'date_day_bank', 'date_day_reg', 'date_diff_days',
        'normalized_description_bank', 'normalized_description_reg', 'desc_similarity',
        'type_bank', 'type_reg', 'type_match',
        'confidence_score', 'flags'
    ]

    return matches[columns].sort_values('confidence_score', ascending=False)
