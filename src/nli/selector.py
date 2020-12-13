#!/usr/bin/env python3

""" Document retrieval module.

This module contains everything required for solving the first part of the
pipeline - retrieving documents matching a claim.
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import torch.nn as nn
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


class Embedder:
    def __init__(self, model="distilbert-base-nli-stsb-mean-tokens"):
        self.model = SentenceTransformer(model)
        self.tokenizer = spacy.load("en_core_web_lg")
        tr = pytextrank.TextRank()
        self.tokenizer.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    def compare(self, claim: str, targets: list):
        claim_embedding = self.embed([claim])
        targets_embedding = self.embed(targets)
        cosine_scores = util.pytorch_cos_sim(claim_embedding, targets_embedding)
        return cosine_scores

    def keywords(self, text: str):
        doc = self.tokenizer(text)
        return " ".join([p.text for p in doc._.phrases])

    def compare_keywords(self, claim: str, targets: list):
        claim_keywords = self.keywords(claim)
        target_keywords = list(map(self.keywords, targets))
        return self.compare(claim_keywords, target_keywords)

    def embed(self, sentences: list):
        return self.model.encode(sentences, convert_to_tensor=True)


class Sequencer:
    def __init__(self, word2vec, dim):
        self.nlp = spacy.load("en_core_web_lg")
        self.w2v = word2vec
        self.w2v.add(["<unk>"], [np.random.uniform(low=-1, high=1.0, size=(dim,))])
        self.tokenizer = lambda text: [t.text for t in self.nlp(text)]
        self.unk_index = self.w2v.vocab.get("<unk>")

    def encode(self, X):
        tokens = self.tokenizer(X.lower())

        seq = []
        for token in tokens:
            seq.append(self.w2v.vocab.get(token, self.unk_index).index)

        return seq

    def create_padded_tensor(self, sequences):
        # Given a list of sequences, pad all to the same length

        max_seq_len = max(len(sequence) for sequence in sequences)
        tensor = torch.full((len(sequences), max_seq_len), 0, dtype=torch.long)

        for i, sequence in enumerate(sequences):
            for j, token in enumerate(sequence):
                tensor[i][j] = token

        return tensor


class Selector:
    """ Sentence selector module """

    def __init__(self, embedder):
        self.embedder = embedder

    def choose_top_n(self, claim, targets, n, concat=True, cosine_sim=True, pad=0):
        """ Chooses top n sentences from targets using cosine similarity to claim

        Parameters
        ----------
        claim : str
            Input claim to match against targets
        targets : list
            List of target sentences
        n : int
            Number of sentences to return
        concat : bool
            If true, the claim will be concatenated with each of the target
            vectors
        cosine_sim : bool
            If true, the cosine similarity will be concatenated with each of the
            target vectors
        pad : int
            If non-zero, adds vectors of pad until output is length `n`

        Returns
        -------
        tensor :
            Tensor of shape `[n, embedding_dim]`, `[n, embedding_dim * 2]`, or
            `[n, 1 + embedding_dim * 2]` if neither, `concat` and `cosine_sim`
            are True, respectively
        """

        embedded = self.embedder.embed([claim] + targets)

        claim = embedded[0]
        # If we have an empty targets list, just return back padding
        if not targets:
            empty = torch.full(size=(n, 1 + (2 * claim.shape[-1])), fill_value=pad)
            return empty, torch.Tensor([])

        targets = embedded[1:]
        k = min(len(targets), n)

        cosines = util.pytorch_cos_sim(claim, targets)
        # filter the targets/scores with top_indices
        top_k = torch.topk(cosines, k)
        top_cosines = top_k.values.T
        top_targets = targets[top_k.indices].squeeze()

        # concatenate the cosine similarity & claims if true
        stacked_claims = torch.stack([claim] * k)
        if concat and cosine_sim:
            result = torch.cat([stacked_claims, top_cosines, top_targets], dim=-1)
        elif concat:
            result = torch.cat([stacked_claims, top_targets], dim=-1)
        elif cosine_sim:
            result = torch.cat([top_cosines, top_targets], dim=-1)

        if pad != 0 and k < n:
            add = n - k
            empty = torch.full(size=(add, result.shape[-1]), fill_value=pad)
            result = torch.cat([result, empty], dim=0)

        return result, top_k.indices.squeeze()


class SelectorNN(nn.Module):
    """Substitute for embedding module during training.

    Selects top N sentences for each claim, and passes on the matrices for those
    specific sentences.
    """

    def __init__(self, num_sents, embeddings):
        super().__init__()
        self.n = num_sents
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)

    def compare_sents(self, s1, s2):
        n, d = s1.shape  # claim
        m, d = s2.shape  # target
        matrix = s1 @ s2.T  # n x m
        scores = torch.softmax(matrix, dim=0)  # softmax along columns
        scores = torch.max(scores, dim=0).values  # collapse the rows into 1 row
        final_score = torch.sum(
            scores
        )  # sum of the softmax values is the 'similarity score'
        return float(final_score), matrix

    def forward(self, claim, targets):
        # claim size: <1, seq_len>
        # targets size: <num_targets, seq_len>
        claim = self.embedding(claim)
        targets = self.embedding(targets)
        # seq len, dim
        # b, m, d = claim.shape
        # b, t, n, d = targets.shape

        # [batch, num_targets, len_claim, len_target]
        batch_U = torch.matmul(claim[:, None, :, :], targets.transpose(-2, -1))
        softmaxed = torch.softmax(batch_U, axis=3)

        # [batch, num_targets, target_len]
        scores = torch.max(softmaxed, axis=2).values

        target_scores = scores.sum(axis=2)  # [batch, num_targets]

        finals = []
        for i, batch in enumerate(target_scores):
            top_k_targets = torch.topk(batch, self.n, dim=0, sorted=False).indices
            top = batch_U[i, top_k_targets, :]
            finals.append(top / top.norm(keepdim=True, dim=2))  # normalize each target sentence

        finals = torch.stack(finals)
        return finals
