from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Split ─────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TRAIN_SIZE   = 0.70
VAL_SIZE     = 0.15
TEST_SIZE    = 0.15

# ── Columns ───────────────────────────────────────────────────────────────────
TEXT_COL    = "text"
TARGET_COL  = "target"
STEMMED_COL = "preprocessed_text"
LEMMA_COL   = "clean_text"
TOKENS_COL  = "tokens"
KW_RATE_COL = "kw_disaster_rate"

# ── TF-IDF ────────────────────────────────────────────────────────────────────
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MAX_FEATURES = 10_000
TFIDF_MIN_DF       = 3
TFIDF_SUBLINEAR_TF = True

# ── Features: XGBoost (keep all — trees handle redundancy natively) ───────────
FEATURES_ALL: list[str] = [
    "kw_disaster_rate",
    "has_numbers",
    "text_length",
    "has_url",
    "exclamation_count",
    "question_count",
    "has_location",
    "word_count",
    "token_count",
    "unique_tokens",
    "preprocessed_len",
    "ttr",
    "compression_ratio",
    "avg_word_length",
    "uppercase_ratio",
    "is_retweet",
]

# ── Features: LR / SVM (drop columns collinear with text_length) ──────────────
FEATURES_LINEAR: list[str] = [
    "kw_disaster_rate",
    "has_numbers",
    "text_length",
    "has_url",
    "exclamation_count",
    "question_count",
    "has_location",
]

# ── Metric ────────────────────────────────────────────────────────────────────
PRIMARY_METRIC = "f1_macro"
