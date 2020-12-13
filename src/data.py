from multiprocessing import Pool
from tqdm import tqdm
import sqlite3 as sql

import pandas as pd
import os
import re

import torch
from torch.utils.data import Dataset

# Local imports
from doc_retrieval.doc_retrieval_keyword import get_docs

WIKI_PATH = "../data/wiki-pages"


def _clean_evidence(row: list):
    """
    Removes unnecessary IDs from evidence column

    Parameters
    ----------
    row : list
        Default element in the `evidence` column in training data
        A list of sufficient evidence sets
        [
            [
                [<annotation_id>, <evidence_id>, "Oliver_Reed", 0]
            ],
            [
                [<annotation_id>, <evidence_id>, "Oliver_Reed", 3],
                [<annotation_id>, <evidence_id>, "Gladiator_-LRB-2000_film-RRB-", 0]
            ]
        ]
    Returns
    -------
    list
        Cleaned `evidence` item
         [
            [
                ["Oliver_Reed", 0]
            ],
            [
                ["Oliver_Reed", 3],
                ["Gladiator_-LRB-2000_film-RRB-", 0]
            ]
        ]
    """
    cleaned = []
    for st in row:
        s = [evidence[2:] for evidence in st]
        cleaned.append(s)
    return cleaned


def _unclean_evidence(row: list):
    """This is stupid. Why am I doing this?"""
    uncleaned = []
    for st in row:
        s = [[None, None] + evidence for evidence in st]
        uncleaned.append(s)
    return uncleaned


def _format_evidence(row: list):
    """ Converts list of (article, line) pairs into a dictionary

    Parameters
    ----------
    row : list
        list of `(article, line)` pairs

    Returns
    -------
    dict :
        dictionary of `{article: [line1 line2, ...]}`
    """
    result = {}
    for art, line in row:
        if art is None:
            return {}
        if art in result:
            result[art].append(line)
        else:
            result[art] = [line]
    return result


def _ungroup_evidence(row: list):
    result = {}
    for group in row:
        for art, line in group:
            if art is None:
                return {}
            if art in result:
                result[art].append(line)
            else:
                result[art] = [line]
    return result


def get_train(path: str):
    """
    Reads in training data from `path` into a DataFrame

    Parameters
    ----------
    path : str
        Path to the `train.jsonl` file

    Returns
    -------
    DataFrame
        Easily readable training data with columns:
            1. id
            2. verifiable
            3. label
            4. claim
            5. evidence
    """
    df = pd.read_json(path, lines=True)

    # convert VERIFIABLE to bools
    df["verifiable"] = df["verifiable"].apply(lambda x: x == "VERIFIABLE")

    # map the labels to 2, 1, 0
    label_map = {"SUPPORTS": 2, "NOT ENOUGH INFO": 1, "REFUTES": 0}
    df["label"] = df["label"].apply(lambda x: label_map[x])

    # Clean up the data
    df["evidence"] = df["evidence"].apply(_clean_evidence)

    return df


###############################################################################
#                              Wikipedia Parsing                              #
###############################################################################


def _get_ids(w):
    """ Gets the ID column from a wiki dataframe

    Parameters
    ----------
    w : str, Path
        Path to the wiki file to parse

    Returns
    -------
    tuple :
        Tuple of `(w, ID column)`
    """
    return (w, set(get_wiki(w).index))


def index_wiki(path: str = WIKI_PATH):
    """
    Indexes the wikipedia articles to speed up retrieval

    Parameters
    ---------
    path : str
        Path to the `wiki-pages` directory
    Returns
    -------
    dict
        Dictionary of `wiki_file : {article IDs}`
    """
    index = {}
    wiki_files = [os.path.join(path, p) for p in os.listdir(path)]
    with Pool(8) as pool:
        index = dict(tqdm(pool.imap(_get_ids, wiki_files), total=len(wiki_files)))

    return index


def find_article(index: dict, article: str):
    """
    Queries the wiki files for an article

    Parameters
    ----------
    index : dict
        Dictionary of `wiki-file : {article IDs}`

    article : str
        Article ID to search for

    Returns
    -------
    pd.Series :
        Article data
    """
    for wiki, items in index.items():
        if article in items:
            w = get_wiki(wiki)
            return w.loc[article]


def get_wiki(path: str = WIKI_PATH):
    """
    Loads the wiki `jsonl` file from `path`

    Parameters
    ----------
    path : str
        Path to the wiki file

    Returns
    -------
    pd.DataFrame
        Human readable wikipedia article data
    """
    return pd.read_json(path, lines=True).drop(0).set_index(["id"])


def clean_article(article: str):
    """ Cleans a wikipedia article and splits into lines/sentences

    Parameters
    ----------
    article : str
        Article text from the `lines` column of the wikipedia data

    Returns
    -------
    list :
        Cleaned and split article
    """
    rrb = re.compile("-RRB-")
    lrb = re.compile("-LRB-")

    lines = []
    for i, line in enumerate(article.split("\n")):
        # replace left/right bracket tokens with the actual brackets
        line = re.sub(rrb, ")", line)
        line = re.sub(lrb, "(", line)
        # replace tabs with spaces
        line = re.sub(r"\t", " ", line)
        # delete the number that starts
        line = re.sub(r"^\d+ ", "", line)
        lines.append(line)

    return lines


###############################################################################
#                                Database Stuff                               #
###############################################################################


class WikiDatabase:
    """Read-only connection to wikipedia database"""

    def __init__(self, path):
        self.path = path
        self.conn = None
        self.cursor = None

    def connect(self):
        if self.conn is None:
            self.conn = sql.connect(self.path)
            self.cursor = self.conn.cursor()

    def close(self):
        if self.conn is not None:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()

    def get_one(self, name: str):
        query = " SELECT * FROM documents WHERE id = ?"
        return self.cursor.execute(query, (name,))

    def get_many(self, names: list):
        query = f"SELECT * FROM documents WHERE id in ({', '.join('?'*len(names))})"
        return self.cursor.execute(query, tuple(names))


###############################################################################
#                          Dataset Class for PyTorch                          #
###############################################################################


class SentenceDataset(Dataset):
    def __init__(self, data, embedder, num_procs, wiki_path):
        self.data = data[data["verifiable"]]
        self.embedder = embedder
        self.wiki = WikiDatabase(wiki_path)
        self.input_pad_idx = 0
        self.output_pad_idx = 3
        self.num_procs = num_procs

    def connect_to_db(self):
        self.wiki.connect()

    def __getitem__(self, idx: int):
        """ Returns a claim, label (0, 1, or 2), and list of
        `(article, line number)` pairs

        Parameters
        ----------
        idx : int
            Index of row to grab

        Returns
        -------
        Tuple[str, int, list]
            claim, label, and associated evidence
        """
        row = self.data.iloc[idx]
        claim = row["claim"]
        evidence = _format_evidence(row["evidence"])
        label = row["label"]  # 0 - refute, 1 - neutral, 2 - support

        return claim, label, evidence

    def pad_output_tensor(self, ys):
        """ Creates a padded tensor from a batch of inputs

        Parameters
        ----------
        ys : list
            list of torch Tensors of shape [n_sentences, ]

        Returns
        -------
        torch.Tensor
            tensor of shape [batch_size, max_n_sentences]
        """
        max_n_sents = max(ys, key=lambda x: x.shape[0]).shape[0]

        all_padded = torch.full(
            size=(len(ys), max_n_sents),
            fill_value=self.output_pad_idx,
            dtype=torch.long,
        )

        for i, y in enumerate(ys):
            # create the padded tensor with the pad index
            # fill it up with the text
            n_sents = len(y)
            all_padded[i, :n_sents] = y

        return all_padded

    def pad_input_tensor(self, Xs):
        """ Creates a padded tensor from a batch of inputs

        Parameters
        ----------
        Xs : list
            list of torch Tensors of shape [n_sentences, embedding_dim]

        Returns
        -------
        torch.Tensor
            tensor of shape [batch_size, max_n_sentences, embedding_dim]
        """
        max_n_sents = max(Xs, key=lambda x: x.shape[0]).shape[0]

        _, embedding_dim = Xs[0].shape
        all_padded = torch.full(
            size=(len(Xs), max_n_sents, embedding_dim),
            fill_value=self.output_pad_idx,
            dtype=torch.float,
        )

        for i, x in enumerate(Xs):
            # create the padded tensor with the pad index
            # fill it up with the text
            n_sents, _ = x.shape
            all_padded[i, :n_sents, :] = x

        return all_padded

    def pad(self, batch):
        xs, indices, ys, labels = zip(*batch)
        xs = self.pad_input_tensor(xs)
        ys = self.pad_output_tensor(ys)
        return (xs, indices, ys, labels)

    def __len__(self):
        return len(self.data)

    def split_line_numbers_by_article(line_numbers, starting_indices):
        """ Converts sentence indices (indexing a collection of articles) to
        line numbers (indexing a single article) and its respective article.

        Parameters
        ----------
        line_numbers : list
            A list of numbers indexing a collection of sentences from multiple articles
        starting_indices : dict
            A dict of {index: article} which tells which article starts at which index.

        Returns
        -------
        dict :
            dictionary of {article: [indices]}
        """
        result = {}
        for line in line_numbers:
            index, article = max(
                [(k, v) for k, v in starting_indices.items() if k < line]
            )
            if article in result:
                result[article].append(line - index)
            else:
                result[article] = [line - index]

        return result


class FastDataset(Dataset):
    def __init__(self, data):
        self.data = data[data["verifiable"]]
        self.input_pad_idx = 3
        self.output_pad_idx = 3

    def __getitem__(self, idx: int):
        """ Returns a claim, label (0, 1, or 2), and list of
        `(article, line number)` pairs

        Parameters
        ----------
        idx : int
            Index of row to grab

        Returns
        -------
        Tuple[str, int, list]
            claim, label, and associated evidence
        """
        row = self.data.iloc[idx]
        claim = row["claim"]
        evidence = row["evidence"]
        label = row["label"]  # 0 - refute, 1 - neutral, 2 - support

        return claim, label, evidence

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, data):
        # don't care about verifiable or not
        self.data = data[data["verifiable"]]
        self.input_pad_idx = 3
        self.output_pad_idx = 3

    def __getitem__(self, idx: int):
        """ Returns a claim, label (0, 1, or 2), and list of
        `(article, line number)` pairs

        Parameters
        ----------
        idx : int
            Index of row to grab

        Returns
        -------
        Tuple[str, int, list]
            claim, label, and associated evidence
        """
        row = self.data.iloc[idx]
        claim = row["claim"]
        evidence = row["evidence"]
        label = row["label"]  # 0 - refute, 1 - neutral, 2 - support

        return claim, label, evidence

    def __len__(self):
        return len(self.data)


class OracleDocRetriever:
    """ Gets articles given evidence

    Will always get the correct articles, so it's good to use for training
    """

    def __init__(self, wiki_path):
        self.wiki = WikiDatabase(wiki_path)
        self.connect_to_db()

    def connect_to_db(self):
        self.wiki.connect()

    def retrieve(self, evidence):
        names, lines = zip(*[e for e in evidence.items()])

        # Fetch all the articles in one query
        articles = self.wiki.get_many(names).fetchall()

        # dictionary of {article_name: article_lines}
        name2lines = {k: v.split("<SPLIT>") for k, v in articles}

        return name2lines


class DocRetriever:
    """ Imperfect document retriever

    To adjust, just change the retrieve function from `names = get_docs(claim)`
    to whatever you wish!
    """

    def __init__(self, wiki_path):
        self.wiki = WikiDatabase(wiki_path)
        self.connect_to_db()

    def connect_to_db(self):
        self.wiki.connect()

    def retrieve(self, claim):
        """ Retrieves a list of document IDs given a claim

        Parameters
        ----------
        claim : str
            Input claim

        Returns
        -------
        dict :
            map between article names and contents
        """

        names = get_docs(claim, threshold=80)

        # Fetch all the articles in one query
        articles = self.wiki.get_many(names).fetchall()

        # dictionary of {article_name: article_lines}
        name2lines = {k: v.split("<SPLIT>") for k, v in articles}

        return name2lines


def collate(
    batch, num_sentences, retriever, selector, oracle_doc_ret=True,
):
    """ Collates a batch of evidence into model-readable format

    Parameters
    ----------
    batch : list
        List of items retrieved via `__getitem__`
    num_sentences : int
        Number of sentences to select as evidence to a claim
    retriever : DocRetriever, OracleDocRetriever
        An object implementing a `retriever` function to retrieve articles from DB
    selector : Selector
        Sentence selector object implementing `choose_top_n` function


    Returns
    -------
    Unknown at the moment
    """

    batch_json_items = []
    batch_sentences = []
    batch_labels = []

    for claim, label, ev in batch:
        formatted_ev = _ungroup_evidence(ev)
        # Document retrieval

        # If we're using the oracle, pass in the evidence
        # Otherwise, pass in the claim
        if oracle_doc_ret:
            name2lines = retriever.retrieve(formatted_ev)
        else:
            name2lines = retriever.retrieve(claim)

        # Sentence Selection
        name2idx = {}
        real_indices = []
        i = 0
        cat_lines = []
        for name, lines in name2lines.items():
            name2idx[name] = [i + num for num in range(len(lines))]
            # these are the "correct" indices we have to predict
            real_indices.extend(name2idx[name])
            i += len(lines)
            cat_lines += lines

        sentences, pred_indices = selector.choose_top_n(
            claim, cat_lines, num_sentences, pad=True
        )

        pred_ev = []
        for name, idxs in name2idx.items():
            pred_ev.extend([[name, p] for p in pred_indices.tolist() if p in idxs])
        # assuming we got the label correct; that way we can check our maximum
        # possible fever score after sentence selection
        json_item = {
            "claim": claim,
            "label": label,
            "predicted_label": label,
            "predicted_evidence": pred_ev,
            "evidence": _unclean_evidence(ev),
        }

        batch_json_items.append(json_item)

        # 1 is NEI tag, so that's what we use as default
        y = torch.ones(size=(num_sentences,), dtype=torch.long)
        for i, (pred, real) in enumerate(zip(pred_indices, real_indices)):
            if pred == real:
                y[i] = label

        batch_labels.append(y)
        batch_sentences.append(sentences)

    X = torch.stack(batch_sentences)
    y = torch.stack(batch_labels)

    return X, y, batch_json_items
