#!/usr/bin/env python

# Local stuff
import data
from nli.selector import Selector, Embedder
import nli.lstm as lstm
import nli.mlp as mlp
from nli.trainer import FastTrainer
from doc_retrieval.doc_retrieval_similarity import get_docs

# Utilities
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Pytorch
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

torch.multiprocessing.set_start_method("spawn", force=True)
np.random.seed(12345)


def prepare(batch):
    NUM_SENTS = 5
    RETRIEVER = data.DocRetriever("../data/wiki.db", get_docs)
    # RETRIEVER = data.OracleDocRetriever("../data/wiki.db")
    em = Embedder()
    sel = Selector(em)
    SELECTOR = sel
    return data.collate(
        batch,
        NUM_SENTS,
        RETRIEVER,
        SELECTOR,
        oracle_doc_ret=isinstance(RETRIEVER, data.OracleDocRetriever),
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###########################################################################
    #                        Setup the datasets/loaders                       #
    ###########################################################################

    train = data.get_train("../data/train.jsonl")
    train, test = train_test_split(train)

    torch.cuda.empty_cache()
    em = Embedder()

    train_dataset = data.FastDataset(train)
    test_dataset = data.TestDataset(test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=prepare,
        num_workers=0,  # doesn't work with more than 1
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=prepare,
        num_workers=0,  # doesn't work with more than 1 and a sqlite connection
    )

    ###########################################################################
    #                       Model Training & Evaluation                       #
    ###########################################################################

    # Model params
    EMBEDDING_DIM = em.model.get_sentence_embedding_dimension()
    HIDDEN_DIM = 100
    HIDDEN_DIMS = [300, 100, 10]
    OUTPUT_DIM = 3  # refute, not enough info, support
    N_LAYERS = 2
    DROPOUT = 1e-1
    BIDIRECTIONAL = True
    # Loss fn params
    WEIGHT_DECAY = 1e-2
    N_EPOCHS = 3
    LR = 1e-2
    LR_DECAY = 1e-3

    # model = lstm.LSTMClassifier(
    #     embedding_dim=EMBEDDING_DIM,
    #     hidden_dim=HIDDEN_DIM,
    #     output_dim=OUTPUT_DIM,
    #     n_layers=N_LAYERS,
    #     dropout=DROPOUT,
    #     bidirectional=BIDIRECTIONAL,
    #     pad_idx=train_dataset.input_pad_idx,
    # )
    model = mlp.MLPClassifier(
        embedding_dim=EMBEDDING_DIM,
        hidden_dims=HIDDEN_DIMS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        pad_idx=train_dataset.input_pad_idx,
    )
    model.to(device)

    # load the pretrained model
    # state_dict = torch.load("../models/mlp-intermediate.pt")
    # model.load_state_dict(state_dict)

    # Optimizer & Loss function
    optimizer = optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY, lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=train_dataset.output_pad_idx, reduction="sum",
    )

    # Evaluate the model on a small subset of stuff
    trainer = FastTrainer(model, optimizer, loss_fn, device, log_every_n=1)
    labels = {0: "REFUTES", 1: "NOT ENOUGH INFO", 2: "SUPPORT"}

    model.share_memory()
    small_train_dataset = data.FastDataset(train.sample(10000))
    small_train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=prepare,
        num_workers=2,  # doesn't work with more than 1
        prefetch_factor=2,
    )
    small_test_dataset = data.TestDataset(test.sample(20))
    small_test_loader = DataLoader(
        small_test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=prepare,
        num_workers=0,  # doesn't work with more than 0
    )
    # trainer.evaluate(small_test_loader, labels)
    trainer.fit(
        small_train_loader,
        small_test_loader,
        labels,
        "/home/offendo/Documents/masters/nlp243/best-words/models/mlp-intermediate.pt",
    )

    torch.save(
        model.state_dict(),
        "/home/offendo/Documents/masters/nlp243/best-words/models/mlp-final.pt",
    )
