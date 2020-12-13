from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

import pickle

import re

from rapidfuzz import fuzz

from pathlib import Path


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
    #tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    #model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    model = SentenceTransformer('LaBSE')

    claim, retrieved_docs = claim_docs
    sentences = [_clean_text_similarity(claim)]
    sentences.extend([_clean_text_similarity(doc_id) for doc_id in retrieved_docs])

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
    with open(Path(__file__).parent / "../../data/wiki_doc_skimmed_ids.obj", "rb") as file:
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
    # It does pretty poorly on this example :(
    # but better than doc_retrieval_keyword.py in terms of articles with name Robert in them :)
    claim = "Robert J. O'Neill was born April 10, 1976."
    print(get_docs(claim))
