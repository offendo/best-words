#!/usr/bin/env python3

import data
import retrieval as ret
import lstm
from trainer import Trainer
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm


if __name__ == '__main__':
    ###########################################################################
    #                                Clean Data                               #
    ###########################################################################
    outdir = "../data/clean/"
    index = data.index_wiki('../data/wiki-pages')
    for file in tqdm(index.keys()):
        wiki = data.get_wiki(file)
        lines = wiki["lines"].apply(lambda l: "<SPLIT>".join(data.clean_article(l)))
        wiki["text"] = lines
        wiki = wiki.drop("lines", axis=1).reset_index()
        new_file = outdir + file.split("/")[-1]
        wiki.to_json(new_file, orient="records", lines=True)
    ###########################################################################
    #                                  Setup                                  #
    ###########################################################################

    # Load the data
    train = data.get_train("../data/train.jsonl")
    train = train.explode("evidence").reset_index()
    train, test = train_test_split(train)

    # Load the model
    embedder = ret.SentEmbed("distilroberta-base-msmarco-v2")

    # Build the dataset objects and loaders
    train_dataset = data.SentenceDataset(train, embedder, "../data/wiki.db", 4)
    test_dataset = data.SentenceDataset(test, embedder, "../data/wiki.db", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=train_dataset.collate,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=test_dataset.collate,
        num_workers=0,  # doesn't work with more than 1 and a sqlite connection
    )

    ###############################################################################
    #                               Model parameters                              #
    ###############################################################################

    # General
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model params
    EMBEDDING_DIM = embedder.model.get_sentence_embedding_dimension()
    HIDDEN_DIM = 100
    OUTPUT_DIM = 3  # refute, not enough info, support
    N_LAYERS = 2
    DROPOUT = 1e-1
    BIDIRECTIONAL = True
    # Loss fn params
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 3
    LR = 1e-3

    # Build the model
    model = lstm.LSTMClassifier(
        None,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        pad_idx=train_dataset.input_pad_idx,
    )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=train_dataset.output_pad_idx,
        reduction="sum",
    )

    trainer = Trainer(model, optimizer, loss_fn, device, log_every_n=5)
    labels = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORT"}

    trainer.fit(
        train_loader=train_loader,
        valid_loader=test_loader,
        labels=labels,
        n_epochs=N_EPOCHS,
    )
