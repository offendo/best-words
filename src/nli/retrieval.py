#!/usr/bin/env python3

""" Document retrieval module.

This module contains everything required for solving the first part of the
pipeline - retrieving documents matching a claim.
"""

from sentence_transformers import SentenceTransformer, util
import torch.multiprocessing as mp
import spacy
import pytextrank

def extract_keywords(nlp, text: str, special_tokens: list = None):
    """ [DEPRECATED] Extracts keywords from a string of text

    Parameters
    ----------
    nlp : model
        spaCy English language model

    text : str
        Query text from which to extract keywords

    special_tokens : list
        list of tokens to be automaitcally added

    Returns
    -------
    set :
        set of keywords from `text`
    """

    pos_tag = {"PROPN", "ADJ", "NOUN"}
    doc = nlp(text.lower())

    # return value: keywords extracted from the text
    keywords = set()

    # add all the special tags which appear in the text
    if special_tokens:
        tags = [tag.lower() for tag in special_tokens]
        keywords |= {t.text for t in doc if t.text in tags}

    # extract noun chunks, but filter out the tokens
    # which aren't in the pos_tag list
    for chunk in doc.noun_chunks:
        final_chunk = " ".join([t.text for t in chunk if t.pos_ in pos_tag])
        if final_chunk:
            keywords.add(final_chunk)

    for token in doc:
        # if the token is a stop-word or punctuation, ignore it
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        # otherwise, if the part of speech is in pos_tag, add it to keywords
        if token.pos_ in pos_tag:
            keywords.add(token.text)

    return keywords


class SentEmbed:
    def __init__(self, model='distilbert-base-nli-stsb-mean-tokens'):
        self.model = SentenceTransformer(model)
        self.tokenizer = spacy.load('en_core_web_lg')
        tr = pytextrank.TextRank()
        self.tokenizer.add_pipe(tr.PipelineComponent, name='textrank', last=True)

    def compare(self, claim: str, targets: list):
        claim_embedding = self.embed([claim])
        targets_embedding = self.embed(targets)
        cosine_scores = util.pytorch_cos_sim(claim_embedding, targets_embedding)
        return cosine_scores

    def keywords(self, text: str):
        doc = self.tokenizer(text)
        return ' '.join([p.text for p in doc._.phrases])

    def compare_keywords(self, claim: str, targets: list):
        claim_keywords = self.keywords(claim)
        target_keywords = list(map(self.keywords, targets))
        return self.compare(claim_keywords, target_keywords)

    def embed(self, sentences: list):
        return self.model.encode(sentences, convert_to_tensor=True)
