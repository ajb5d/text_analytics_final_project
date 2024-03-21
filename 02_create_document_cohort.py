#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
JOURNALS = ["jamia_open", "jamia"]
ARTICLES_PER_YEAR = 20

all_articles = [pd.read_parquet(DATA_DIR / f"{x}.parquet") for x in JOURNALS]
all_articles = pd.concat(all_articles)
all_articles['sortdate'] = pd.to_datetime(all_articles['sortdate'])
all_articles['pubyear'] = all_articles['sortdate'].dt.year

print(f"Total articles: {len(all_articles)}")
all_articles = all_articles[all_articles['pubyear'].between(2000, 2023)]
print(f"Total articles (2000-2023): {len(all_articles)}")

year_span = list(range(all_articles['pubyear'].min(), all_articles['pubyear'].max()+1))

sampled_df = all_articles.groupby('pubyear').sample(n=ARTICLES_PER_YEAR, random_state=0)
print(f"Sampled articles: {len(sampled_df)}")

sampled_df.to_parquet(DATA_DIR / "sampled_articles.parquet")