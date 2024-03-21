#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data")
JOURNALS = ["jamia_open", "jamia"]
SUBSET_SIZE = 250

## To get the article manifests from the public s3 bucket:
# aws --no-sign-request s3 cp s3://pmc-oa-opendata/author_manuscript/xml/metadata/csv/author_manuscript.filelist.csv data/ 
# aws --no-sign-request s3 cp s3://pmc-oa-opendata/oa_noncomm/xml/metadata/csv/oa_noncomm.filelist.csv data/
# aws --no-sign-request s3 cp s3://pmc-oa-opendata/oa_comm/xml/metadata/csv/oa_comm.filelist.csv data/

if not (DATA_DIR / "article_manifests.parquet").exists():
    article_manifests = []
    for subset in tqdm(["author_manuscript", "oa_noncomm", "oa_comm"]):
        article_manifests.append(
            pd.read_csv(
                DATA_DIR / f"{subset}.filelist.csv",
                usecols=["AccessionID", "Key", "PMID"]
            )
        )
    pd.concat(article_manifests).to_parquet(DATA_DIR / "article_manifests.parquet")

article_manifests = pd.read_parquet(DATA_DIR / "article_manifests.parquet").set_index("AccessionID")

all_articles = [pd.read_parquet(DATA_DIR / f"{x}.parquet") for x in JOURNALS]
all_articles = pd.concat(all_articles)
all_articles['sortdate'] = pd.to_datetime(all_articles['sortdate'])
all_articles['pubyear'] = all_articles['sortdate'].dt.year

print(f"Total articles: {len(all_articles)}")
all_articles = all_articles.merge(article_manifests, left_on="pmcid", right_index=True)
print(f"Total articles available in download subset: {len(all_articles)}")

sampled_df = all_articles.sample(n=SUBSET_SIZE, random_state=0)
print(f"Sampled articles: {len(sampled_df)}")

all_articles.to_parquet(DATA_DIR / "all_articles.parquet")
sampled_df.to_parquet(DATA_DIR / "sampled_articles.parquet")