import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm

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
    return [i[0][2:] for i in row]


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

    # map the labels to 1, 0, -1
    label_map = {"SUPPORTS": 1, "NOT ENOUGH INFO": 0, "REFUTES": -1}
    df["label"] = df["label"].apply(lambda x: label_map[x])

    # Clean up the data
    df["evidence"] = df["evidence"].apply(_clean_evidence)

    return df


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
    return (w, set(get_wiki(w)["id"]))


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
            print(article, wiki)
            w = get_wiki(wiki)
            return w[w["id"] == article]


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
    return pd.read_json(path, lines=True).drop(0)
