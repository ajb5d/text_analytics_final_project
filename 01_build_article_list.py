#!/usr/bin/env python3

from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd

DATA_DIR = Path("data")
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FIELDS = ['uid', 'title', 'pubdate', 'source', 'sortdate']
JOURNALS = {
    'jamia_open': "JAMIA Open",
    'jamia': "J Am Med Inform Assoc",
    # 'aci': "Appl Clin Inform",
}
CHUNK_SIZE = 500

for journal, journal_name in JOURNALS.items():
    params = {
        "db": "pmc",
        "term": f"{journal_name}[Journal]",
        "usehistory": "y",
        "retmode": "json",
    }

    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params)
    resp.raise_for_status()
    response_body = resp.json()

    webenv = response_body["esearchresult"]["webenv"]
    query_key = response_body["esearchresult"]["querykey"]
    count = int(response_body["esearchresult"]["count"])

    print(f"{journal}: Found {count} articles")
    starts = list(range(0, count, CHUNK_SIZE))
    records = []
    for start in tqdm(starts, desc=journal):
        params = {
            "db": "pmc",
            "query_key": query_key,
            "WebEnv": webenv,
            "retmax": CHUNK_SIZE,
            "retstart": start,
            "retmode": "json",
        }

        resp = requests.get(f"{BASE_URL}/esummary.fcgi", params=params)
        resp.raise_for_status()
        response_body = resp.json()

        for uid in response_body["result"]["uids"]:
            new_record = {}
            for field in FIELDS:
                new_record[field] = response_body["result"][uid][field]
            for id_entry in response_body["result"][uid]["articleids"]:
                new_record[id_entry["idtype"]] = id_entry["value"]
            records.append(new_record)

    df = pd.DataFrame(records)
    df.to_parquet(DATA_DIR / f"{journal}.parquet")