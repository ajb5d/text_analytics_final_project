# Steps

1. `01_build_article_list.py` uses the [NCBI Entrez APIs](https://www.ncbi.nlm.nih.gov/books/NBK25501/) to build a list of articles from candidate journals that are available in [Pub Med Central](https://www.ncbi.nlm.nih.gov/pmc/). 

1. `02_create_document_cohort.py` uses the article lists from the previous step, merges it with the metadata of available articles, and samples the articles.

1. `03_download_articles.py` takes the articles from the previous step and downloads them from the public s3 bucket. This is the F0 format.

1. `04_parse_articles.py` converts the F0 documents into F1 (paragraphs, sentences, words) tables in the `data/corpus_f1.sqlite` sqlite database


