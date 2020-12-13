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

train_data = os.path.join('data', 'train.jsonl')
tfidf_path = os.path.join('data', 'wiki_docs_skimmed-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
ranker = utils.TfidfDocRanker(tfidf_path=tfidf_path)


def normalize(prediction_scores):
    exp = prediction_scores
    normalized = exp / np.sum(exp)
    return np.array(normalized > 0.1)


def minmaxnormalize(prediction_scores):
    mms = MinMaxScaler()
    return mms.fit_transform(prediction_scores.reshape(prediction_scores.shape[0], -1)).reshape(prediction_scores.shape[0],) > 0.4


def mms_get_predicted():
    mms_pred = []
    for predicted_row in predicted:
        mms = minmaxnormalize(predicted_row[1][1])
        mms_docset = set(compress(predicted_row[1][0], mms))
        mms_pred.append((predicted_row[0], mms_docset))
    return mms_pred

def get_docs(claim, k=5, include_scores=False, norm_type=None, zscore_threshold=False):
    closest_docs, score = ranker.closest_docs(claim, k=k)
    if zscore_threshold:
        softmax = np.exp(score) / np.exp(score).sum()
        scores = zscore(softmax)
        return [doc for doc, s in zip(closest_docs, scores) if s >= zscore_threshold]

    if include_scores:
        return claim, set(closest_docs)
    return closest_docs

if __name__ == "__main__":
    with open(train_data, 'r') as fp:
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

    
    # predicted = mms_get_predicted()
    
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

    print('AVG Recall of tf-idf: {}%'.format(round(100 * np.average(recalls), 2)))
