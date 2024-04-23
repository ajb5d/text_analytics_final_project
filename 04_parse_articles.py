#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from lxml import etree
import sqlite3
import nltk

def textify(element):
    if len(element) == 0:
        return element.text or ''
    
    text_content = element.text or ''
    for child in element:
        text_content += textify(child)
        if child.tail:
            text_content += child.tail
    return text_content

def sanitize_document(body):
    STRIP_ELEMENTS = ["boxed-text", "fig", "table-wrap", "xref", "sup", "sub", "title", "disp-quote"]
    for e in STRIP_ELEMENTS:
        etree.strip_elements(body, e, with_tail=False)
    STRIP_TAGS = ["italic", "bold", "underline", "list", "list-item"]
    for e in STRIP_TAGS:
        etree.strip_tags(body, e)

def extract_grafs(body, title):
    header_section = []
    tail_section = []
    sections = []
    for e in body:
        if e.tag != "sec":
            if len(sections) == 0:
                header_section.append(textify(e))
            else:
                tail_section.append(textify(e))
        if e.tag == "sec":
            section_paragraphs = []
            for child in e:
                section_paragraphs.append(textify(child))
            sections.append(section_paragraphs)
    if len(header_section) > 0:
        sections.insert(0, header_section)
    if len(tail_section) > 0:
        sections.append(tail_section)

    doc = []
    for chap_num, chap in enumerate(sections):
        for para_num, para in enumerate(chap):
            doc.append({
                "title": title,
                "chapter": chap_num,
                "paragraph": para_num,
                'token_str': para,
            })
            
    return (
        pd.DataFrame(doc)
            .set_index(["title", "chapter", "paragraph"])
    )

docs = []
for document in tqdm(list(Path("data/documents").glob("*.xml"))):
    with open(document, "r") as file:
        tree = etree.parse(file)
    bodies = tree.xpath("/article/body")
    if len(bodies) == 0:
        continue
    body = bodies[0]
    sanitize_document(body)
    result = extract_grafs(body, document.stem)
    docs.append(result)

PARAS = pd.concat(docs)

SENTS = (
    PARAS["token_str"]
    .apply(lambda x: pd.Series(nltk.sent_tokenize(x)))
    .stack()
)

SENTS.index.names = ["title", "chapter", "paragraph", "sentence"]
SENTS.name = "token_str"

WORDS = (
    SENTS
    .apply(lambda x: pd.Series(nltk.word_tokenize(x)))
    .stack()
)
WORDS.index.names = ["title", "chapter", "paragraph", "sentence", "word"]
WORDS.name = "token_str"

db = sqlite3.connect("data/corpus_f1.sqlite")
PARAS.to_sql("paragraphs", db, if_exists="replace")
SENTS.to_sql("sentences", db, if_exists="replace")
WORDS.to_sql("words", db, if_exists="replace")
