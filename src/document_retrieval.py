import os
import pandas as pd
import data
import re
from sklearn.model_selection import train_test_split
import sqlite3
import unicodedata
import pickle
from rapidfuzz import fuzz
from multiprocessing import Pool

import numpy as np
seed = np.random.seed(2)

from tqdm import tqdm


class DocDB(object):
    """Sqlite backed document storage.
    Implements get_doc_text(doc_id).
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

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    #  Added in for test purposes
    def get_many_doc_ids(self, count):
        """Fetch ids of one doc stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchmany(count)]
        cursor.close()
        return results

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


def extract_key_words(sent):
    """
    A simple heuristic to remove stop words from a sentence and return relevant tokens or use NER to extract entities.
    TODO: Not too sure of this, I have a feeling a good stop word list would be sufficient.
    :param sent: The cleaned input string, usually the claim
    :return key_words: List of words in claim that are not stop words, or words that are recognized as entities by Spacy
    """
#    stopwords = nlp.Defaults.stop_words
#    key_words = [x.text for x in nlp(sent).ents]
#    if len(key_words) == 0:
    stop_words = ["the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
                  "few", "little", "much", "lot", "of", "most", "some", "any", "enough", "other", "another", "such",
                  "what", "rather", "quite", "and", "be", "in", "can", "been", "has", "on", "only", "is", "was", "with",
                  "at", "to", "where", "will"]

    key_words = [tok for tok in sent.split() if tok.lower() not in stop_words]
    return key_words


def clean_text(text):
    """
    Basic clean text utility where RRB and LRB tags are replaced with parentheses and underscores are replaced with
    spaces
    :param text:
    :return text: string - Cleaned text
    """
    rrb = re.compile("-RRB-")
    lrb = re.compile("-LRB-")
    text = re.sub(rrb, ")", text)
    text = re.sub(lrb, "(", text)
    text = re.sub("_", " ", text)
    text = str(text).lower()
    return text


def get_cleaned_first_line(db, doc_id):
    doc = db.get_doc_text(doc_id)
    sent_delim = re.compile(r'\s+\.\s+')
    first_line = re.split(sent_delim, doc)[0]
    return clean_text(first_line)


def key_word_match_filter(claim_gold_pair):
    """
    A simple key_word matching program, where we are (for now) looking at claim and doc title (doc id) and doing
    fuzz partial ratio matching to filter docs.

    :param claim_gold_pair: Tuple that holds (claim, gold_evidence)
    claim: List of important words to look for in documents
    gold_evidence: The evidence set from training, just for purposes of keeping order for multiprocessing
    :return (docs, gold_evidence)
    docs: List of doc ids of the documents that matched claims
    gold_evidence: The evidence set from training, just for purposes of keeping order for multiprocessing
    """
    claim, gold_evidence = claim_gold_pair
    db = DocDB("../project_data/wiki_docs_skimmed.db")
    with open("../project_data/wiki_doc_skimmed_ids.obj", "rb") as file:
        ids = pickle.load(file)
    docs = []
    compare_claim = claim.lower()
    for doc_id in ids:
        title = doc_id
        # Clean up doc id
        title = clean_text(title)
        more_text = title + " . " + get_cleaned_first_line(db, doc_id) + " . "
        similarity = fuzz.partial_ratio(compare_claim, more_text)
        if similarity > 90:  # Can be tuned to further narrow down, but need to keep recall same
            docs.append(doc_id)
    db.close()
    return docs, gold_evidence


def key_word_match_filter_nonpool(claim):
    """
    A simple key_word matching program, where we are (for now) looking at claim and doc title (doc id) and doing
    fuzz partial ratio matching to filter docs.

    :param claim: List of important words to look for in documents
    gold_evidence: The evidence set from training, just for purposes of keeping order for multiprocessing
    :return docs: List of doc ids of the documents that matched claims
    """
    with open("../project_data/wiki_doc_skimmed_ids.obj", "rb") as file:
        ids = pickle.load(file)
    docs = []
    compare_claim = claim.lower()
    for doc_id in ids:
        title = doc_id
        # Clean up doc id
        rrb = re.compile("-RRB-")
        lrb = re.compile("-LRB-")
        title = re.sub(rrb, ")", title)
        title = re.sub(lrb, "(", title)
        title = re.sub("_", " ", title)
        title = str(title).lower()
        similarity = fuzz.partial_ratio(compare_claim, title)
        if similarity > 50:  # Can be tuned to further narrow down, but need to keep recall same
            docs.append(doc_id)
    return docs


def key_word_score_filter(claim_gold_pair):
    """
    A simple key_word matching program, where we are (for now) looking at claim and doc title (doc id) and based on key
    words extracted from claim, see if at least two of those keywords exist in doc title and first line of doc

    :param claim_gold_pair: Tuple that holds (claim, gold_evidence)
    claim: List of important words to look for in documents
    gold_evidence: The evidence set from training, just for purposes of keeping order for multiprocessing
    :return (docs, gold_evidence)
    docs: List of doc ids of the documents that matched claims
    gold_evidence: The evidence set from training, just for purposes of keeping order for multiprocessing
    """
    claim, gold_evidence = claim_gold_pair
    db = DocDB("../project_data/wiki_docs_skimmed.db")
    with open("../project_data/wiki_doc_skimmed_ids.obj", "rb") as file:
        ids = pickle.load(file)
    docs = []
    compare_claim = claim.lower()
    for doc_id in ids:
        title = doc_id
        # Clean up doc id
        title = clean_text(title)
        more_text = title + " . " + get_cleaned_first_line(db, doc_id) + " . "
        key_words_claims = extract_key_words(compare_claim)
        key_words_doc = extract_key_words(more_text)
        count = 0
        for key in key_words_doc:
            if key in key_words_claims:
                docs.append(doc_id)
                break
        #if count >= len(key_words_claims)/2:  #a heuristic: if at least half of the keywords are found in doc, doc is retrieved
        #    docs.append(doc_id)
    db.close()
    return docs, gold_evidence


def key_word_experiment(df):
    """
    An experiment to check the train data to see if it is sufficient to look for key words (from the claim) in the doc id
    instead of the text of the doc.
    Run fuzz.partial_ratio on claim and doc id
        if ratio is less than 50,
            print the doc id and the sentence number that holds the evidence
                this is to check how much of doc we might need to look at to extract key words in addition to doc id
    TODO: Look at a larger portion of random docs to see how many lines to include
    :return:
    """
    db = DocDB("../project_data/wiki_docs_skimmed.db")
    for i, row in df.iterrows():
        if row["verifiable"] is False:  # Note: "label" field is unreliable
            continue
        claim = row["claim"].lower()
        evidence_set = row["evidence"]
        titles = []
        for st in evidence_set:
            st_titles = []
            for doc in st:
                st_titles.append((doc[0], doc[1]))
            # print(f"One sufficent evidence set for claim:{claim}:")
            for title, line_num in st_titles:
                id = title
                title = clean_text(title)
                first_line = get_cleaned_first_line(db, id)
                more_text = title + " . " + first_line + " . "
                ratio = fuzz.partial_ratio(claim, title)
                #print(ratio)
                #print(f"\n***Claim: {claim}***\ndoc_id: {title}\npartial_ratio: {ratio}\n")
                if ratio < 50:
                    print(
                        f"\n***Claim: {claim}***\ndoc_id: {title}\npartial_ratio: {ratio}\n text {more_text}")
    db.close()


def remove_duplicates(evidence_sets):
    """
    TODO: Should I put this in the data.py as a utility function so it can be used for evaluation of doc retrieval?
    Removes duplicate documents within each evidence set, and remove duplicate evidence sets for the purposes
    of document retrieval
    :param evidence_sets: List of sets of sufficient veracity label-supporting document ids and line numbers
        Ex:
        [
            [
                ["Oliver_Reed", 0],
                ["Oliver_Reed", 10]
            ],
            [
                ["Oliver_Reed", 3],
                ["Gladiator_-LRB-2000_film-RRB-", 0]
            ],
            [
                ["Oliver_Reed", 6],
                ["Gladiator_-LRB-2000_film-RRB-", 10]
            ]
        ]
    Modifies evidence_sets to a list of unique sets of doc ids
        Ex:
        [
            ["Oliver_Reed"],
            ["Oliver_Reed", "Gladiator_-LRB-2000_film-RRB-"]
        ]

    """
    unique_evidence_sets = set()
    for evidence_set in evidence_sets:
        evidences = frozenset([evidence[0] for evidence in evidence_set])
        unique_evidence_sets.add(evidences)
    print(unique_evidence_sets)
    return unique_evidence_sets


def evaluate_recall(pred_gold_pair):
    """
    For a given claim and the doc_ids retrieved by the model, compare to a gold complete set of required documents.
    Using a recall-based metric where first checked for equivalence to complete set(s), then to gold each set calculate
    recall (how many matched with gold set/how many in gold set) and return highest recall.

    :param pred_gold_pair: tuple that holds (retrieved_doc_ids, evidence_sets)
    retrieved_doc_ids: list - a list doc ids that model retrieved for this claim
    evidence_sets: list of gold evidence sets for this claim
    [
            [
                ["<doc_id>", 0]
            ],
            [
                ["<doc_id>", 3],
                ["<doc_id>", 0]
            ]
    ]
    :param claim: string - actual claim text
    :return recall: float - highest recall for retrieved doc ids
    Note: For both recieved: [id1, id2, id3] gold: [id1, id2]
                   recieved: [id1, id2] gold: [id1, id2]
          the recall is 1
    """
    retrieved_doc_ids, evidence_sets = pred_gold_pair

    highest_recall = None
    evidence_sets = remove_duplicates(evidence_sets) # golds
    for e in evidence_sets:
        total = len(e)
        correct = len(e & retrieved_doc_ids)
        recall = correct / total
        if highest_recall is None or recall > highest_recall:
            highest_recall = recall
    return highest_recall


def phase_1(df):
    """
    Prints out stats of recall for key-word match filter on train set
    :param df: The train set of claims and gold evidence
    """

    verifiable = df[df['verifiable'] is not False]
    claim_gold_pairs = list(zip(verifiable['claim'], verifiable['evidence'])) # list of tuples (claim, evidence_set)
    print(f"Looking at a random {len(claim_gold_pairs)} verifiable examples")

    #  Filter documents based on keyword/fuzzy matching
    pred_gold_pairs = []  # list of tuples (retrieved_docs, gold_evidence_set)
    av_cores = 2
    with Pool(av_cores) as pool:
        pred_gold_pairs = pool.map(key_word_score_filter, claim_gold_pairs)
    print([len(pair[0]) for pair in pred_gold_pairs])
    print(f'The average length of the documents retrieved = {sum([len(pair[0]) for pair in pred_gold_pairs])/len(pred_gold_pairs)}')
    #  Evaluate retreived document recall
    recall_scores = []
    with Pool(av_cores) as pool:
        recall_scores = pool.map(evaluate_recall, pred_gold_pairs)
    print(f'Average recall = {sum(recall_scores) / len(recall_scores)}')


def phase_1_evaluate_small_random_samples(df):
    """
    How I randomly sampled data to evaluate the keyword extraction quickly for my computer
    :param df: The dataframe with data
    :return:
    """
    print("Random seed 2")
    for i in range(10):
        print(f"***Iteration {i}***")
        print(f"Average recall:")
        train_set, _ = train_test_split(df, train_size=100, random_state=seed) # need to look into NEI ratios and balance
        phase_1(train_set)


def main():
    train_file = os.path.join('../data', 'train.jsonl')
    df = data.get_train(train_file)

    # Phase 1: Filter docs based on key_word matching
    phase_1_evaluate_small_random_samples(df)
    #  key_word_experiment(claim_set)

    # Phase 2: Sift through retrieved docs using sentence embeddings


if __name__ == '__main__':
    main()
