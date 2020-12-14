from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from rapidfuzz import fuzz
from pathlib import Path
import sqlite3
import unicodedata
import os

DB_PATH = os.path.join("data", "wiki_docs_skimmed.db")
WIKI_IDS_PATH = os.path.join("data", "wiki_doc_skimmed_ids.obj")

class DocDB(object):
    """Sqlite backed document storage.
    Implements get_doc_text(doc_id).
    Credit: This portion of the code was taken from Facebook's DrQA
    project under the retrieval library (doc_db.py)
    """

    def __init__(self, db_path=None):
        self.path = db_path or DEFAULTS['db_path']
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (unicodedata.normalize('NFD', doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


def _clean_text(text):
    """
    Basic clean text utility where RRB and LRB tags are removed and underscores are replaced with
    spaces and text is converted to all lowercase and punctuation is removed. Used for partial ratio
    filter preprocessing
    :param text:
    :return text: string - Cleaned text
    """
    rrb = re.compile("-RRB-")
    lrb = re.compile("-LRB-")
    new_text = re.sub(rrb, " ", text)
    new_text = re.sub(lrb, " ", new_text)

    punct = re.compile(r'[_?!.,]')
    new_text = re.sub(punct, " ", new_text)

    new_text = str(new_text).lower()
    return new_text


def _clean_text_similarity(text):
    """
    Basic clean text utility where RRB and LRB tags are removed and underscores are replaced with
    spaces. Punctuation is kept for sentence embeddings, which will futher tokenize text.
    :param text:
    :return text: string - Cleaned text
    """
    rrb = re.compile("-RRB-")
    lrb = re.compile("-LRB-")

    new_text = re.sub(rrb, ")", text)
    new_text = re.sub(lrb, "(", new_text)

    new_text = re.sub(r"_", " ", new_text)
    return new_text


def _get_cleaned_first_line_similarity(db, doc_id):
    """

    Parameters
    ----------
    db - DocDB database object (sqlite3)
    doc_id - string that represents the id of a document in the database

    Returns
    -------
    first_line: string - cleaned text
    """
    doc = db.get_doc_text(doc_id)
    sent_delim = re.compile(r'\s+\.\s+')
    first_line = re.split(sent_delim, doc)[0]
    return _clean_text_similarity(first_line + ".")


def _rank(claim_docs):
    """
        Takes the documents filtered based on partial ratio threshold of > 75, and ranks the docs
        according to their cosine similarity with the claim when encoded using BERT sentence embeddings
        (https://huggingface.co/sentence-transformers/LaBSE), and returns the top 5 ranked documents.
        During training/tuning using random sampling of small subsets, this approach performed with a 83.8% recall.

        :param claim_docs: Tuple that holds (claim, retrieved_docs)
                claim: string
                        the raw claim read in from the data
                retrieved_docs: list
                        doc ids of the filtered docs from filter 1

        :return filtered_docs: list
                doc ids of the documents that had the top 5 cosine similarity values with the claim
    """

    model = SentenceTransformer('LaBSE')
    # this second model looks better suited for task, but not enough testing to conclude that it is better than LaBSE
    #model = SentenceTransformer('distilroberta-base-msmarco-v2')

    db = DocDB(DB_PATH)
    claim, retrieved_docs = claim_docs
    sentences = [_clean_text_similarity(claim)]

    # looks at doc id and first line of doc
    sentences.extend([(_clean_text_similarity(doc_id)+" "+_get_cleaned_first_line_similarity(db, doc_id)) for doc_id in retrieved_docs])
    db.close()

    embeddings = model.encode(sentences)
    docs_similarities = []
    for i in range(1, len(retrieved_docs)+1):
        # find cosine similarity of the claim and this ith doc
        cos_sim = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[i].reshape(1, -1))[0][0]
        doc = retrieved_docs[i-1]
        docs_similarities.append((doc, cos_sim))

    ordered_docs = sorted(docs_similarities, key=lambda pair: pair[1], reverse=True)
    filtered_docs = [pair[0] for pair in ordered_docs]
    return filtered_docs[:5]


def get_docs(claim: str):
    """
    Simple key_word matching, where we look at the claim and doc title (doc id) and doing
    fuzz partial ratio matching to filter docs, and return the top 5 ranked documents.

    :param claim: string
                  the raw claim read in from the data
    :return docs: list
                  doc ids of the top 5 documents that matched with claim
    """
    with open(WIKI_IDS_PATH, "rb") as file:
        ids = pickle.load(file)
    docs = []
    compare_claim = _clean_text(claim)
    for doc_id in ids:
        title = _clean_text(doc_id)
        similarity = fuzz.partial_ratio(compare_claim, title)
        if similarity > 75:  # this threshold ensures less docs returned inorder for sentence embedding filter to be tractable
            docs.append(doc_id)
    top_five_docs = _rank((claim, docs))
    return top_five_docs


if __name__ == "__main__":
    claim = "Robert J. O'Neill was born April 10, 1976."
    print(get_docs(claim))
