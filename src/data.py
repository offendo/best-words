import pandas as pd
import os
import json


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


def index_wiki(path: str):
    """
    Indexes the wikipedia articles to speed up retrieval

    Parameters
    ---------
    path : str
        Path to the `wiki-pages` directory
    Returns
    -------
    dict
        Dictionary of `wiki_file : (first_item, last_item)`
    """
    index = {}
    for wiki in os.listdir(path):
        with open(wiki, "r") as f:
            # first line is empty for some reason, index 1 is the first row
            next(f)
            first = f.readline()
            for line in f:
                pass
            last = line
            first_json = json.loads(first)["id"]
            last_json = json.loads(last)["id"]
        index[wiki] = (first_json, last_json)
    return index


def _is_word_in_range(word: str, minmax: tuple):
    """ Compares `word` lexically with `minmax[0]` and `index[1]`

    Parameters
    ---------
    word : str
        Word to query
    minmax : tuple
        Tuple containing the first and last word to search between

    Returns
    -------
    int
        0 if `min < word < max`
        1 if `min < max < word`
        -1 if `word < min < max`
    """
    # make everything lowercase
    first, last = minmax
    first, last = first.lower(), last.lower()
    word = word.lower()

    # compare
    if word < first:
        return -1
    elif word > last:
        return -1
    return 0


def get_wiki(index, keyword):
    """
    Returns the wiki-file which contains `keyword`

    Parameters
    ----------
    index : dict
        Dictionary of `wiki-file : (first_word, last_word)`

    keyword : str
        Word to search for

    Returns
    -------
    str
        Path of the wiki file to retrieve
    """
    for wiki, minmax in index.items():
        if _is_word_in_range(keyword, minmax) == 0:
            return wiki
