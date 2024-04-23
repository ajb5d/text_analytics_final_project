#!/usr/bin/env python3
import pandas as pd
import sqlite3
import numpy as np

db = sqlite3.connect("data/corpus_f3.sqlite")
DOCS = pd.read_sql("SELECT * FROM doc", db)
TOKENS = pd.read_sql("SELECT * FROM token", db)
TERMS = pd.read_sql("SELECT * FROM terms", db)
db.close()

def tfidf(token_df, bag_level, count_method="n", tf_method="sum", idf_method="standard", tf_norm_k=0.5):
    """
    Compute TF-IDF matrix from a token table.
    Args:
    token_df: DataFrame
        A table of tokens with term_id and doc_id columns.
    bag_level: list of str
        A list of column names to group by.
    count_method: str
        'n' or 'c' for number of tokens or number of distinct tokens.
    tf_method: str
        'sum', 'max', 'log', 'double_norm', 'raw', 'binary'
    idf_method: str
        'standard', 'max', 'smooth'
    tf_norm_k: float
        The value of K in double_norm formula.
    """

    bag_of_words = (
        token_df
            .groupby(bag_level+['term_id'])['term_id']
            .value_counts()
            .to_frame(name="n")
    )
    bag_of_words['c'] = 1

    document_term_count_matrix = bag_of_words[count_method].unstack().fillna(0)

    match tf_method:
        case 'sum':
            TF = document_term_count_matrix.T / document_term_count_matrix.T.sum()
        case 'max':
            TF = document_term_count_matrix.T / document_term_count_matrix.T.max()
        case 'log':
            TF = np.log10(1 + document_term_count_matrix.T)
        case 'raw':
            TF = document_term_count_matrix.T
        case 'double_norm':
            TF = document_term_count_matrix.T / document_term_count_matrix.T.max()
            TF = tf_norm_k + (1 - tf_norm_k) * TF[TF > 0] # EXPLAIN; may defeat purpose of norming
        case 'binary':
            TF = document_term_count_matrix.T.astype('bool').astype('int')
        case _:
            raise ValueError(f"tf_method '{tf_method}' not recognized")
        
    TF = TF.T
    DF = document_term_count_matrix[document_term_count_matrix > 0].count()
    N = document_term_count_matrix.shape[0]

    match idf_method:
        case 'standard':
            IDF = np.log10(N / DF)
        case 'max':
            IDF = np.log10(DF.max() / DF) 
        case 'smooth':
            IDF = np.log10((1 + N) / (1 + DF)) + 1 # Correct?
        case _:
            raise ValueError(f"idf_method '{idf_method}' not recognized")

    TFIDF = TF * IDF

    return TFIDF

TFIDF = tfidf(TOKENS, ["title"])

TFIDF_SUM = TFIDF.sum()
TFIDF_SUM.name = "tfidf_sum"
TFIDF_SUM = TFIDF_SUM.to_frame()

TOKENS = TOKENS.merge(TFIDF_SUM, left_on="term_id", right_index=True, how="left")
TERMS = TERMS.merge(TFIDF_SUM, left_on="term_id", right_index=True, how="left")

dbout = sqlite3.connect("data/corpus_f4.sqlite")
DOCS.to_sql("doc", dbout, if_exists='replace', index=False)
TOKENS.to_sql("tokens", dbout, if_exists='replace', index=False)
TERMS.to_sql("terms", dbout, if_exists='replace')
dbout.commit()
dbout.close()