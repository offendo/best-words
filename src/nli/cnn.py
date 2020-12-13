#!/usr/bin/env python3


import torch.nn as nn
import numpy as np
import torch
from nli.selector import SelectorNN

np.random.seed(12345)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        num_sents,
        embeddings,
            input_dim,
            channels,
        kernel_sizes,
        hidden_dims,
        output_dim,
        dropout,
    ):
        super(CNNClassifier, self).__init__()

        # linear layers
        self.selector = SelectorNN(num_sents, embeddings)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(input_dim, channels, kernel_size)
                for kernel_size in kernel_sizes
            ]
        )
        self.fc = nn.Linear(len(kernel_sizes) * channels, output_dim)
        # ReLU as activation function
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, claim, targets):
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

        selected = self.selector(claim, targets)

        convs = [self.activation(conv(selected)) for conv in self.convs]

        pools = [F.max_pool2d(conv_out, conv_out.shape[2]).squeeze(2) for conv_out in convs]

        return out
