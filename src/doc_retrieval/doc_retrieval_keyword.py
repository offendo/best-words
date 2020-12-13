import re
import pickle
from rapidfuzz import fuzz
from pathlib import Path


def _clean_text(text):
    """
    Basic clean text utility where RRB and LRB tags are removed and underscores are replaced with
    spaces and text is converted to all lowercase and punctuation is removed.
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


def _rank(claim_docs):
    """
    Takes the documents filtered based on partial ratio threshold of > 50, and ranks the docs
    according to their partial ratios with the claims, and returns the top 5 ranked documents.
    During training/tuning using random sampling of small subsets, this approach performed with a 72% recall.

    :param claim_docs: Tuple that holds (claim, retrieved_docs)
            claim: string
                    the raw claim read in from the data
            retrieved_docs: list
                    doc ids of the filtered docs from filter 1

    :return filtered_docs: list
            doc ids of the documents that had the top 5 partial ratios
    """

    claim, retrieved_docs = claim_docs
    compare_claim = _clean_text(claim)
    partial_ratios = [(doc_id, fuzz.partial_ratio(compare_claim, _clean_text(doc_id))) for doc_id in retrieved_docs]
    ordered_partial_ratios = sorted(partial_ratios, key=lambda pair: pair[1], reverse=True)
    filtered_docs = [pair[0] for pair in ordered_partial_ratios]
    if len(ordered_partial_ratios) > 5:
        filtered_docs = filtered_docs[:5]
    return filtered_docs


def get_docs(claim: str, threshold: int = 50):
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
        if similarity >= threshold:  # tunable parameter, depends on how high recall should be in this stage
            docs.append(doc_id)
    # top_five_docs = _rank((claim, docs))
    # return top_five_docs
    return docs

if __name__ == "__main__":
    claim = "Peggy Sue Got Married is a Egyptian film released in 1986."
    print(get_docs(claim))
