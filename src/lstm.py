#!/usr/bin/env python3


import torch.nn as nn
import numpy as np
import torch

np.random.seed(12345)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        embeddings,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        bidirectional,
        pad_idx,
    ):

        super(LSTMClassifier, self).__init__()
        # Embedding layer...don't know if this is necessary since we're doing
        # the embeddings elsewhere. Maybe there's a way to move it to here?
        # self.embedding = nn.Embedding.from_pretrained(
        #     embeddings, freeze=True, padding_idx=pad_idx
        # )

        # Create the lstm layer
        self.lstm = nn.LSTM(
            2 * embedding_dim + 1,
            hidden_dim,
            n_layers,
            bias=True,
            batch_first=True,
            bidirectional=bidirectional
        )

        # dropout
        self.dropout = nn.Dropout(dropout)

        # fully connected at the end
        scale = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * scale, output_dim, bias=True)

    def forward(self, X):
        """ Forward function for the entailment classifier

        Parameters
        ----------
        X : Tensor
            tensor of shape `[batch_size, num targets, 2 * dim + 1]`. Each row is the
            concatenation of `[claim, cosine_sim, target]`

        Returns
        -------
        Tensor :
            Tensor of shape `[batch_size, num targets, num_classes]`
        """

        # LSTM layer
        out, hidden = self.lstm(self.dropout(X))

        # FC layer
        logits = self.fc(self.dropout(out))

        return logits
