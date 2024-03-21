# Steps

1. `01_build_article_list.py` uses the [NCBI Entrez APIs](https://www.ncbi.nlm.nih.gov/books/NBK25501/) to build a list of articles from candidate journals that are available in [Pub Med Central](https://www.ncbi.nlm.nih.gov/pmc/). 

1. `02_create_document_cohort.py` uses the article lists from the previous step and samples the articles stratified by publication year.