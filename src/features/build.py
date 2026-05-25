import pandas as pd

from src.data.preprocess import clean_text, preprocess_data, tokenize


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to df in-place. Returns df."""
    # ── Text length & structure ───────────────────────────────────────────────
    df["text_length"]     = df["text"].str.len().astype("float64")
    df["word_count"]      = df["text"].str.split().str.len().astype("float64")
    df["avg_word_length"] = df["text"].apply(
        lambda t: round(sum(len(w) for w in str(t).split()) / max(len(str(t).split()), 1), 2)
    )

    # ── Binary surface signals ────────────────────────────────────────────────
    df["has_url"]    = df["text"].str.contains(r"https?://", regex=True, na=False).astype(int)
    df["is_retweet"] = df["text"].str.startswith("RT ", na=False).astype(int)
    df["has_numbers"] = df["text"].str.contains(r"\d", regex=True, na=False).astype(int)

    # ── Emotional / stylistic markers ─────────────────────────────────────────
    df["uppercase_ratio"]   = df["text"].apply(
        lambda t: round(sum(c.isupper() for c in str(t)) / max(len(str(t)), 1), 3)
    )
    df["exclamation_count"] = df["text"].str.count(r"!")
    df["question_count"]    = df["text"].str.count(r"\?")

    # ── Location flag ─────────────────────────────────────────────────────────
    df["has_location"] = (~df["location"].isna()).astype(int)

    # ── Lemmatized tokens (for SHAP) ──────────────────────────────────────────
    df["clean_text"]    = df["text"].apply(clean_text)
    df["tokens"]        = df["text"].apply(tokenize)
    df["token_count"]   = df["tokens"].str.len().astype("float64")
    df["unique_tokens"] = df["tokens"].apply(lambda t: len(set(t))).astype("float64")
    df["ttr"]           = (
        df["unique_tokens"] / df["token_count"].where(df["token_count"] > 0)
    ).round(3)

    # ── Stemmed text for TF-IDF ───────────────────────────────────────────────
    df["preprocessed_text"] = df["text"].apply(preprocess_data)
    df["preprocessed_len"]  = df["preprocessed_text"].str.split().str.len().astype("float64")
    df["compression_ratio"] = (
        df["preprocessed_len"] / df["word_count"].where(df["word_count"] > 0)
    ).round(3)

    return df


def add_kw_disaster_rate(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """
    Compute kw_disaster_rate on train only, then map to all splits.
    Unknown keywords in val/test receive the global train disaster rate.
    Returns (train, *others) with the new column added in-place.
    """
    kw_rate = train.groupby("keyword")["target"].mean().rename("kw_disaster_rate").round(3)
    global_rate = round(float(train["target"].mean()), 3)

    train["kw_disaster_rate"] = train["keyword"].map(kw_rate)
    for split in others:
        split["kw_disaster_rate"] = split["keyword"].map(kw_rate).fillna(global_rate)

    return (train, *others)
