#!/usr/bin/env python3
import pandas as pd
import sqlite3
import nltk
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer

db = sqlite3.connect("data/corpus_f1.sqlite")

PARAGRAPHS = pd.read_sql_query("SELECT * FROM paragraphs", db)
SENTENCES = pd.read_sql_query("SELECT * FROM sentences", db)
WORDS = pd.read_sql_query("SELECT * FROM words", db)

DOC = SENTENCES
WORDS['term_str'] = WORDS['token_str'].str.lower()
TERM = WORDS['term_str'].value_counts().reset_index()
TERM.index.name = 'term_id'
WORDS = WORDS.merge(TERM['term_str'].reset_index())

dbout = sqlite3.connect("data/corpus_f2.sqlite")
DOC.to_sql("doc", dbout, if_exists='replace', index=False)
WORDS.to_sql("token", dbout, if_exists='replace', index=False)
TERM.to_sql("terms", dbout, if_exists='replace')
dbout.commit()
dbout.close()

stemmer = PorterStemmer()
stop_words = nltk.corpus.stopwords.words("english")

TERM["p_stem"] = TERM.term_str.apply(stemmer.stem)
TERM['is_stopword'] = TERM['term_str'].isin(stop_words)
TERM["is_num"] = TERM['term_str'].str.match("\d+")
TERM['is_punctuation'] = TERM['term_str'].str.match("\W+")
TERM['is_number'] = TERM['term_str'].str.match("\d+")

token_str = []
term_str = []
pos = []
for row in tqdm(DOC.itertuples(), total=DOC.shape[0]):
    sent_tokens = nltk.word_tokenize(row.token_str)
    sent_pos = nltk.pos_tag(sent_tokens)

    token_str.extend(sent_tokens)
    term_str.extend([t[0].lower() for t in sent_pos])
    pos.extend([x[1] for x in sent_pos])


pos_labels = pd.DataFrame({'term_str': term_str, 'pos': pos})
pos_labels_most_common = pos_labels.groupby('term_str')['pos'].agg(pd.Series.mode)

TERM = TERM.merge(pos_labels_most_common, on='term_str', how='left')
TERM.index.name = 'term_id'

dbout = sqlite3.connect("data/corpus_f3.sqlite")
DOC.to_sql("doc", dbout, if_exists='replace', index=False)
WORDS.to_sql("token", dbout, if_exists='replace', index=False)
TERM.to_sql("terms", dbout, if_exists='replace')
dbout.commit()
dbout.close()