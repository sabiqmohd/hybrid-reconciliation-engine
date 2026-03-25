import pandas as pd


def clean_dataframe(df):
    """Takes raw transaction data, lowercases text, strips punctuation, standardizes types, and converts dates."""
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    df['normalized_description'] = (
        df['description']
        .astype(str)
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    if 'type' in df.columns:
        df['type'] = (
            df['type']
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({'dr': 'debit', 'cr': 'credit'})
        )

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['date_day'] = df['date'].dt.floor('D')
    df['rounded_amount'] = df['amount'].round(2)

    return df


def load_and_clean_data(bank_csv_path, register_csv_path):
    """Reads both CSVs and returns cleaned, normalized dataframes ready for matching."""
    bank_df = pd.read_csv(bank_csv_path)
    register_df = pd.read_csv(register_csv_path)
    return clean_dataframe(bank_df), clean_dataframe(register_df)
