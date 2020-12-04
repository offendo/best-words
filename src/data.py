from multiprocessing import Pool
from tqdm import tqdm
import sqlite3 as sql

import time
import pandas as pd
import os
import re

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.nn.functional import cosine_similarity

WIKI_PATH = "../data/wiki-pages"


def _clean_evidence(row: list):
    """
    Removes unnecessary IDs from evidence column

    Parameters
    ----------
    row : list
        Default element in the `evidence` column in training data

    Returns
    -------
    list
        Cleaned `evidence` item
    """
    return [i[2:] for j in row for i in j]


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

    def connect(self):
        self.conn = sql.connect(self.path)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
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
    def __init__(self, data, embedder, wiki_path):
        self.data = data[data["verifiable"]]
        self.embedder = embedder
        self.wiki = WikiDatabase(wiki_path)
        self.wiki.connect()
        self.input_pad_idx = 0
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
        evidence = _format_evidence(row["evidence"])
        label = row["label"]  # 0 - refute, 1 - neutral, 2 - support

        return claim, label, evidence

    def collate(self, batch: list):
        """ Collates a batch of evidence into model-readable format

        Parameters
        ----------
        batch : list
            List of items retrieved via `__getitem__`

        Returns
        -------
        Unknown at the moment
        """

        # claims : list of strings
        # labels : list of ints
        # evidences : list of dicts
        claims, labels, evidences = zip(*batch)

        # flatten the list of dicts and split the keys/values
        # into list of article names (keys) and line numbers
        all_article_names, all_line_numbers = zip(
            *[items for e in evidences for items in e.items()]
        )

        # Fetch all the articles in one query
        all_articles = self.wiki.get_many(all_article_names).fetchall()

        # dictionary of {article_name: article_lines}
        name2lines = {k: v.split("<SPLIT>") for k, v in all_articles}

        ###########################
        #   BOTTLENECK IN SPEED   #
        ###########################
        start = time.time()
        # embed all the article texts
        # name2embed = {k: self.embedder.embed(v) for k, v in name2lines.items()}
        processes = []
        with mp.Pool(processes=self.num_procs) as pool:

            embeddings = pool.map(self.embedder.embed, name2lines.values())
        end = time.time()
        print("Embedding: ", end - start)

        # Go through each element in the batch and build the input/output
        # tensors and the starting indices for each article
        prepared_batch = []
        for claim, label, evidence in batch:
            i = 0
            starting_indices = {}
            target_numbers = []
            claim_embed = self.embedder.embed(claim)
            targets = []
            # get the starting indices for each article
            for article_name, lines in evidence.items():
                # add the embeddings for this article to the list of targets
                embeddings = name2embed[article_name]
                targets.extend(embeddings)
                # mark this article as starting at the current index
                starting_indices[i] = article_name
                # reindex the lines to start at the new index
                # occasionally we get a sentence number that's not right
                reindexed_lines = [num + i for num in lines if num < len(embeddings)]
                for i in lines:
                    if i >= len(embeddings):
                        print("Error:")
                        print(f"Trying to index sentence {i}")
                        print("But '{article_name}' has only {len(embeddings)}")
                        print(f"Claim: {claim}")

                target_numbers.extend(reindexed_lines)
                # increment the index by number of lines
                i += len(lines)

            # Concatenate the claim and target embeddings
            stacked_targets = torch.stack(targets)
            cosine_sims = cosine_similarity(claim_embed, stacked_targets, dim=-1)
            stacked_claims = torch.stack([claim_embed] * len(targets))
            X = torch.cat(
                [stacked_claims, cosine_sims.unsqueeze(1), stacked_targets], dim=-1
            )

            # Convert the sentence indices into a list of support/refute class
            # indices. 1 is default since 'not enough info' is mapped to 1
            y = torch.ones(size=(len(targets),))
            for n in target_numbers:
                y[n] = label

            # Add the elements to the prepared batch
            prepared_batch.append((X, starting_indices, y))
        # pad the tensors
        return self.pad(prepared_batch)

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
            fill_value=self.input_pad_idx,
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
        xs, indices, ys = zip(*batch)
        xs = self.pad_input_tensor(xs)
        ys = self.pad_output_tensor(ys)
        return (xs, indices, ys)

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
