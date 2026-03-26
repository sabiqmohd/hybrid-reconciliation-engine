# Financial Reconciliation System

## 1. Project Summary

This project builds an automated financial reconciliation system that matches transactions between two data sources — a **bank statement** and a **check register** — where the records don't line up neatly. Dates can differ by up to 5 days, descriptions are written differently, transaction types use different codes, and some amounts have slight rounding differences.

The system uses a **hybrid approach**:

1. **Rule-based matching** handles the easy wins first — transactions where the amount is unique across both datasets get matched instantly with high confidence.
2. **ML/similarity-based matching** tackles the harder cases using sentence-transformer embeddings for description similarity, combined with numerical features like amount closeness and date proximity.
3. **A learning loop** trains a Logistic Regression on validated matches to produce calibrated confidence scores that improve as more labeled data becomes available.

**Key outcome:** The system matches **307 out of 308** transactions (99.7%) with high confidence. The remaining transaction is flagged for manual review rather than forced into a bad match.

---

## 2. Performance Analysis

### 2.1 Metrics

| Metric     | Rule-Based | ML-Based (After Learning) |
|------------|:----------:|:-------------------------:|
| Precision  | 1.0000     | 0.9932                    |
| Recall     | 1.0000     | 1.0000                    |
| F1 Score   | 1.0000     | 0.9966                    |

**Matching coverage breakdown:**

| Stage                        | Matches Found |
|------------------------------|:-------------:|
| Unique Amounts               | 286           |
| ML Similarity                | 21            |
| Low Confidence / Unmatched   | 1             |

### 2.2 Observations

The ML learning loop pushed average confidence from **0.67 → 0.9995**, correctly validating those harder matches that the rule-based scorer was conservatively rating.

### 2.3 Hardest Transactions to Match

**Date mismatches (0–5 days):** 

**Description variations:** The bank says `"ONLINE PMT WATER"`, the register says `"Water bill"`. These describe the same transaction but share almost no words. Sentence-transformer embeddings (`all-MiniLM-L6-v2`) capture semantic similarity rather than lexical overlap, which is exactly why they were chosen over simple string matching. 

**Slight amount differences:** A bank records $147.29 but the register shows $147.33 (a $0.04 rounding difference). The system tolerates up to $5.00 of difference in candidate retrieval

---

## 3. Design Decisions

### Choice of Approach

The system combines **pre-trained sentence embeddings** with **Numerical features** in a weighted scoring scheme. This was chosen for three reasons:

1. **Simplicity:** A 4-feature Logistic Regression is easy to debug, explain, and trust. When a match scores 0.58, you can look at the individual components (amount score, date score, etc.) and understand exactly why.

2. **Speed:** Pre-computing embeddings for ~300 transactions takes under a second. The entire pipeline runs in about 10 seconds including model loading.

3. **Practicality:** With only ~300 transactions per source, deep learning or complex graph-based methods would be overkill.

---

## 4. Limitations

**Description quality dependency.** The system relies heavily on descriptions for disambiguation. If both sources use cryptic codes (e.g., `"TXN#4827"` vs `"REF-4827"`), the sentence embeddings won't capture much similarity. The current dataset has reasonably descriptive text, but this wouldn't hold for all financial institutions.

**Limited dataset size.** 308 transactions is small. The patterns learned (e.g., "amount differences above $2 are likely non-matches") may not generalize to datasets with larger natural variance. The model hasn't seen enough edge cases to be robust against adversarial or highly noisy data.

**Static embeddings.** The `all-MiniLM-L6-v2` model was used off-the-shelf. Fine-tuning on financial transaction descriptions would likely improve description similarity scores, especially for domain-specific terms.

---

## 5. Future Improvements

**Supervised model with real labels.** Replace pseudo-labels with human-validated match/non-match annotations. Even 200 manually reviewed pairs would significantly improve the Logistic Regression's decision boundary.

**ANN search with FAISS.** For datasets with 10K+ transactions, computing all-pairs similarity becomes expensive. FAISS indexing on the description embeddings would enable sub-linear candidate retrieval.

**Fine-tuned embeddings.** Train or fine-tune a sentence model on financial transaction descriptions to improve semantic matching for domain-specific terminology.

---

## 6. Edge Case Handling

| Edge Case                  | How It's Handled                                                                                             |
|----------------------------|--------------------------------------------------------------------------------------------------------------|
| **Missing descriptions**   | `NaN` descriptions default to empty string for embedding. Similarity score falls to ~0, but amount/date/type features can still carry the match. |
| **Large date differences** | Candidate retrieval uses a hard ±5 day cutoff. Matches beyond this window are not considered, and any candidate at the boundary gets a date score near 0. |
| **Slight amount mismatches** | Up to $5 tolerance in candidate retrieval. Amount score decays linearly — a $0.04 difference scores 0.992, while $4.99 scores ~0.001. |
| **Type code differences**  | Normalized during preprocessing: `DR`→`debit`, `CR`→`credit`, `DEBIT`→`debit`, `CREDIT`→`credit`. All comparisons use standardized codes. |
| **Case/punctuation noise** | Descriptions are lowercased, punctuation stripped, and whitespace normalized before any matching.            |
| **Anomalous transactions** | Transactions with no plausible candidate (all scores below threshold) are flagged as `NO_CANDIDATE` or `LOW_CONFIDENCE` for manual review. |

---

## Project Structure

```
Financial Reconciliation System/
├── data/
│   ├── bank_statements.csv 
│   └── check_register.csv 
├── output/                        
│   ├── matched_transactions.csv   
│   └── evaluation_metrics.csv    
├── data_loader.py         
├── matcher.py    
├── ml_matcher.py   
├── learning_loop.py   
├── run_reconciliation.py 
├── requirements.txt  
└── README.md
```

---

## How to Run

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone or navigate to the project directory
cd "Financial Reconciliation System"

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_reconciliation.py
```
