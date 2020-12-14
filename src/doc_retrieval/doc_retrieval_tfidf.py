import json
import itertools
import os
import sys
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

proj_path = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
sys.path.append(proj_path)
import src.utils.drqa_retriever_utils as utils

TRAIN_PATH = os.path.join('data', 'train.jsonl')
TFIDF_PATH = os.path.join('data', 'wiki_docs_skimmed-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
ranker = utils.TfidfDocRanker(tfidf_path=TFIDF_PATH)


def minmax_normalization(docs, scores, threshold):
    ''' Uses a MinMaxScaer to range scores of docs between [0,1]. Then retrieves all
        documents that pass a certain threshold.
    '''
    if not threshold:
        threshold = 0.1
    mms = MinMaxScaler()
    mms_scores =  mms.fit_transform(scores.reshape(scores.shape[0], -1)).reshape(scores.shape[0],)
    return [doc for doc, score in zip(docs, mms_scores) if score >= threshold]


def softmax_zscore_normalization(docs, scores, threshold):
    ''' Uses a softmax to normalize scores. Takes zscores and retrieves all
        documents that pass a certain threshold.
    '''
    if not threshold:
        threshold = 1
    exp = np.exp(scores)
    softmax_normalized = exp / exp.sum()
    z_scores = zscore(softmax_normalized)
    return [doc for doc, score in zip(docs, z_scores) if score >= threshold]


def get_docs(claim, k=5, norm_type='softmax', threshold=None):
    ''' Retrieves top k documents by some metric. By default, top 10 documents, and softmax normalization.
    '''
    closest_docs, scores = ranker.closest_docs(claim, k=k)

    if norm_type == 'softmax':
        return softmax_zscore_normalization(closest_docs, scores, threshold)
    if norm_type == 'minmax':
        return minmax_normalization(closest_docs, scores, threshold)
    return closest_docs


def evaluate_retrieval(claims, predicted):
    recalls = np.array([])
    for (_, _, gold_groups), pred in zip(claims, predicted):
        best_recall = None
        for gold in gold_groups:
            p = len(gold)
            tp = len(gold & pred)
            local_recall = tp/p
            if best_recall is None or local_recall > best_recall:
                best_recall = local_recall
        recalls = np.append(recalls, best_recall)
    return np.average(recalls)


if __name__ == "__main__":
    with open(TRAIN_PATH, 'r') as fp:
        claims = []
        claims_sets = []
        for jsonl in fp.readlines():
            jsoned = json.loads(jsonl)
            if jsoned['verifiable'] is False:
                continue
            evidence_set = set()
            evidence_groups = set()
            for evidence_batch in jsoned['evidence']:
                evidences = frozenset([evidence[2] for evidence in evidence_batch])
                evidence_groups.add(evidences)
            for evidence in itertools.chain(*jsoned['evidence']):
                if evidence[2] is not None:
                    evidence_set.add(evidence[2])
            claims.append((jsoned['claim'], evidence_set, evidence_groups))

    predicted = []
    for claim, _, _ in claims:
        predicted.append(set(get_docs(claim, k=5)))

    print('AVG Recall of tf-idf: {}%'.format(round(100 * evaluate_retrieval(claims, predicted), 2)))
