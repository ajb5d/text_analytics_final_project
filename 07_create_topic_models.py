#!/usr/bin/env python3
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

N_TERMS = 5_000
N_TOPICS = 10
MAX_ITER = 10

db = sqlite3.connect("data/corpus_f3.sqlite")
DOCS = pd.read_sql("SELECT * FROM doc", db)
TOKENS = pd.read_sql("SELECT * FROM token", db)
TERMS = pd.read_sql("SELECT * FROM terms", db)
db.close()

TFIDF = pd.read_parquet("data/corpus_f4_tfidf.parquet")
## Make the data set smaller by removing:
## - numbers
## - stopwords
## - punctuation
## - proper nouns
## And only keep the top 10,000 terms by frequency
## We do this to reduce the computational complixity of the PCA

small_tfidf_terms = (
    TERMS[
        (TERMS["is_num"] == 0) &
        (TERMS["is_stopword"] == 0) &
        (TERMS["is_punctuation"] == 0) &
        (TERMS["pos"] != "NNP") &
        (TERMS["pos"] != "NNPS")]
    .sort_values("count", ascending=False)
    .head(10_000)
    ["term_id"].tolist()
)

SMALL_TFIDF = TFIDF[small_tfidf_terms]


pca = make_pipeline(
    Normalizer(), #Normalize along each row for document length
    StandardScaler(with_std=False), #Center the data
    PCA(n_components=10)
)
big_pca_result = pd.DataFrame(
    pca.fit_transform(TFIDF),
    index = TFIDF.index
)

small_pca_result = pd.DataFrame(
    pca.fit_transform(SMALL_TFIDF),
    index = SMALL_TFIDF.index
)
big_pca_result.to_parquet("../data/corpus_f5_pca.parquet")
small_pca_result.to_parquet("../data/corpus_f5_small_pca.parquet")

DOCUMENT_CORPUS = (
    TOKENS[TOKENS['term_id'].isin(small_tfidf_terms)]
        .groupby(["title"])
        ["term_str"]
        .apply(lambda x: ' '.join(x))
        .to_frame()
)
lda_pipeline = make_pipeline(
    CountVectorizer(max_features=N_TERMS, stop_words="english"),
    LDA(n_components=N_TOPICS, max_iter=MAX_ITER, random_state=42)
)

lda_pipeline.fit(DOCUMENT_CORPUS["term_str"])

THETA = pd.DataFrame(lda_pipeline.transform(DOCUMENT_CORPUS["term_str"]), index=DOCUMENT_CORPUS.index)
PHI = pd.DataFrame(lda_pipeline.steps[1][1].components_, columns=lda_pipeline.steps[0][1].get_feature_names_out())

PHI.index.name = "topic_id"
THETA.to_parquet("../data/corpus_f5_theta.parquet")
PHI.to_parquet("../data/corpus_f5_phi.parquet")