#!/usr/bin/env python3
import pandas as pd
import sqlite3
from gensim.models import word2vec
from sklearn.manifold import TSNE
import numpy as np

WINDOW = 5
VECTOR_SIZE = 384
MIN_COUNT = 200
N_LARGEST = 1_000
db = sqlite3.connect("data/corpus_f3.sqlite")
DOCS = pd.read_sql("SELECT * FROM doc", db)
TOKENS = pd.read_sql("SELECT * FROM token", db)
TERMS = pd.read_sql("SELECT * FROM terms", db)
db.close()
SENTENCE_CORPUS = (
    TOKENS
        .groupby(["title", "chapter", "paragraph", "sentence"])
        ["term_str"]
        .apply(lambda x: x.tolist())
        .tolist()
)


model = word2vec.Word2Vec(SENTENCE_CORPUS, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=4)
top_terms = (
    TERMS[
        TERMS["term_str"].isin(model.wv.index_to_key) & 
        (TERMS["term_str"].str.len() > 1) &
        (TERMS["is_stopword"] == 0) &
        (TERMS["is_num"] == 0)]
    .nlargest(N_LARGEST, "count")
    ["term_str"]
    .tolist()
)

top_terms = [term for term in top_terms if term in model.wv.index_to_key]

label = pd.Series(top_terms, name = "term")
vector = pd.Series([model.wv[x] for x in top_terms], name = "vector")
coords = pd.concat([label, vector], axis = 1)
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
tsne_values = tsne_model.fit_transform(np.array(coords['vector'].to_list()))
coords['x'] = tsne_values[:,0]
coords['y'] = tsne_values[:,1]
coords.to_parquet("data/corpus_f5_word2vec.parquet", index = False)
