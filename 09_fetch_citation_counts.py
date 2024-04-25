#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
import requests

ARTICLES = pd.read_parquet("data/all_articles.parquet")

BASE_URL = "https://opencitations.net/index/api/v2/citation-count/pmid:"
result = {}
for row in tqdm(ARTICLES.itertuples(), total=len(ARTICLES)):
    target = f"{BASE_URL}{row.PMID}"
    response = requests.get(target)

    response_obj = response.json()

    if len(response_obj) == 0:
        result[row.pmcid] = 0
    else:
        result[row.pmcid] = response_obj[0]["count"]
for key in result:
    result[key] = int(result[key])
pd.DataFrame(result.items(), columns=["pmcid", "citation_count"]).to_parquet("data/citation_count.parquet")