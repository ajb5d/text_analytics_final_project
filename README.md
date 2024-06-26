This source is available at https://github.com/ajb5d/text_analytics_final_project

The processed data is available at https://virginia.box.com/s/3ejoovqm72vc64trwk9fa8a3aqt4szut

# Data Manifest
Provenance: These documents came from articles deposited from two journals (JAMIA and ACI) into the [Pub Med Central](https://www.ncbi.nlm.nih.gov/pmc/) archive.

Location: [UVA Box](https://virginia.box.com/s/a23iqqcla8q8fi5ek8g8gqjsnqcivsa7)

Description: This is a collection of articles about clinical informatics from two journals (JAMIA and ACI)

Format: XML in the 'Journal Archiving and Interchange DTD'

# Steps

1. `01_build_article_list.py` uses the [NCBI Entrez APIs](https://www.ncbi.nlm.nih.gov/books/NBK25501/) to build a list of articles from candidate journals that are available in [Pub Med Central](https://www.ncbi.nlm.nih.gov/pmc/). 

1. `02_create_document_cohort.py` uses the article lists from the previous step, merges it with the metadata of available articles, and samples the articles.

1. `03_download_articles.py` takes the articles from the previous step and downloads them from the public s3 bucket. This is the F0 format.

1. `04_parse_articles.py` converts the F0 documents into F1 (paragraphs, sentences, words) tables in the `data/corpus_f1.sqlite` sqlite database

1. `05_create_stadm.py` takes the F1 corpus, transforms it into F2, and annotates it (F3). The F2 and F3 results are saved into sqlite files.

1. `06_add_tfidf.py` computes TF-IDF on the F3 corpus and annotates the TOKEN and TERM tables to create a F4 corpus saved into `data/corpus_f4.sqlite`

1. `07_create_topic_models.py` computes document level PCA vectors and LDA based topic models.

1. `08_create_word_embeddings.py` computes word2vec embeddings for the entire corpus.

# Explorations

* `00_baseline_eda.ipynb` basic statistics about the coropus
* `01_tfidf_pca.ipynb` explorations of TF-IDF based features
* `02_topic_models.ipynb` visualizations of topic models
* `03_word_embeddings.ipynb` word2vec embeddings of the corpus
