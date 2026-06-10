"""
Stage DVC 1 — preprocess
Charge les données depuis OpenML, applique le feature engineering,
découpe en 3 splits et sauvegarde les parquets.
Reproduit le notebook 01_02_splitting.ipynb.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from settings.params import RANDOM_STATE, TRAIN_SIZE
from src.data.make_dataset import load_data
from src.features.build import add_features, add_kw_disaster_rate

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_data("disaster-tweets", columns_to_lower=True)

# Supprime les lignes sans texte (même comportement que le notebook 01_02_splitting)
df = df[df["text"].notna() & (df["text"].str.strip() != "")]
df = df.reset_index(drop=True)

add_features(df)

# Split stratifié 70 / 15 / 15
train_df, temp_df = train_test_split(
    df,
    test_size=1 - TRAIN_SIZE,
    stratify=df["target"],
    random_state=RANDOM_STATE,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["target"],
    random_state=RANDOM_STATE,
)

train_df, val_df, test_df = add_kw_disaster_rate(train_df, val_df, test_df)

train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
val_df.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

print(f"train : {len(train_df)} lignes | disaster_rate={train_df['target'].mean():.3f}")
print(f"val   : {len(val_df)} lignes | disaster_rate={val_df['target'].mean():.3f}")
print(f"test  : {len(test_df)} lignes | disaster_rate={test_df['target'].mean():.3f}")
