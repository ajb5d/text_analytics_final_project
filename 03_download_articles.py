#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import s3fs

DATA_DIR = Path("data")
DOCUMENT_DIR = DATA_DIR / "documents"
S3_BASE = "s3://pmc-oa-opendata"

s3 = s3fs.S3FileSystem(anon=True)
articles_df = pd.read_parquet(DATA_DIR / "all_articles.parquet")

if not DOCUMENT_DIR.exists():
    DOCUMENT_DIR.mkdir(parents=True)

for record in tqdm(articles_df.to_dict(orient="records")):
    if (DOCUMENT_DIR / f"{record['pmcid']}.xml").exists():
        continue
    s3.get(f"{S3_BASE}/{record['Key']}", str(DOCUMENT_DIR / f"{record['pmcid']}.xml"))
    