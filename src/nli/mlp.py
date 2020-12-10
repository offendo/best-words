#!/usr/bin/env python3

import torch.nn as nn
import numpy as np
import torch
np.random.seed(12345)

class MLPClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dims,
        output_dim,
        dropout,
        pad_idx,
    ):
        super(MLPClassifier, self).__init__()

        # linear layers
        self.fc1 = nn.Linear(1 + embedding_dim * 2, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)
        # self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        # ReLU as activation function
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """ Forward function for the entailment classifier

        Parameters
        ----------
        X : Tensor
            tensor of shape `[batch_size, num targets, 2 * dim]`. Each row is the
            concatenation of `[claim, target]`

        Returns
        -------
        Tensor :
            Tensor of shape `[batch_size, num targets, num_classes]`
        """

        # Reshape to batch_size * num targets, 2*dim
        # X = X.view(-1, X.shape[-1])

        out = self.activation(self.fc1(self.dropout(X)))
        out = self.fc2(self.dropout(out))
#         out = self.fc3(self.dropout(out))

        return out
