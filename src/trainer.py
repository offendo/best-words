#!/usr/bin/env python3


import torch.nn as nn
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch

class Trainer:
    """ Class to train models for predicting
    """

    def __init__(self, model, optimizer, loss_fn, device, log_every_n=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.log_every_n = log_every_n

    def _print_summary(self):
        """Print summary of the training method"""
        print(f"model: {self.model}")
        print(f"optimizer: {self.optimizer}")
        print(f"loss_fn: {self.loss_fn}")

    def train(self, loader):
        """ Run a single epoch of training

        Parameters
        ----------
        loader : DataLoader
            DataLoader for a dataset to train

        Returns
        -------
        ([float], [float]) :
            Loss history (for plotting purposes)
        """

        # Run the model in training mode
        self.model.train()

        loss_history = []
        running_loss = 0.0
        running_loss_history = []

        for i, batch in tqdm(enumerate(loader), total=len(loader)):

            # Zero out the gradient
            self.optimizer.zero_grad()

            # Split up the batch
            X, starting_indices, y = batch

            # X : shape [batch_size, num sentences, input dim]
            # y : shape [batch_size, num sentences, output dim]

            # Foward
            logits = self.model(X.to(self.device))

            # logits : shape [batch_size, num_sentences, output dim]

            # Reshape to (num sentences * batch size, output dim)
            logits = logits.view(-1, logits.shape[-1])

            # Compute loss & add to history
            loss = self.loss_fn(logits, y.view(-1).to(self.device))
            loss_history.append(loss.item())

            # Compute a rolling average loss & add to history
            running_loss += (loss_history[-1] - running_loss) / (i + 1)
            running_loss_history.append(running_loss)

            # Log the running loss
            if self.log_every_n and i % self.log_every_n == 0:
                print(f"Loss: {loss}")
                print(f"Running loss: {running_loss}")
                print(f"{logits.shape}")
                # print(f"predictions: {torch.argmax(logits, dim=-1)}")
                # print(f"y: {y.view(-1)}")

            # Backpropogation
            loss.backward()

            # Clip the norms to prevent explosion
            nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

            # update step
            self.optimizer.step()

        print("Epoch completed!")
        print(f"Epoch Loss: {running_loss}")

        return loss_history, running_loss_history

    def evaluate(self, loader, labels):
        """ Evaluate model on validation data

        Parameters
        ----------
        loader : data.DataLoader
            Data loader class containing validation data

        labels : dict
            Index to output class

        Returns
        -------
        ([float], [float]):
            Loss history and running loss history

        """
        self.model.eval()
        batch_wise_true_labels = []
        batch_wise_predictions = []

        loss_history = []
        running_loss = 0.0
        running_loss_history = []

        # don't compute gradient
        with torch.no_grad():
            for i, batch in enumerate(loader):

                # Split up the batch
                X, lengths, y = batch

                # Foward
                logits = self.model(X.to(self.device), lengths)
                og_shape = logits.shape
                print(og_shape)

                # Reshape to be (sent len * batch size, output dim)
                logits = logits.view(-1, logits.shape[-1])

                # Compute loss & add to history
                loss = self.loss_fn(logits, y.view(-1).to(self.device))

                # no backprop
                loss_history.append(loss.item())

                running_loss += (loss_history[-1] - running_loss) / (i + 1)
                running_loss_history.append(running_loss)

                # softmax to normalize probabilities class
                probs = torch.softmax(logits, dim=-1)
                print(probs.shape)

                # get the output class from the probs
                # also, reshape the prediction back to sentences
                prediction = torch.argmax(probs, dim=-1).reshape(og_shape[:-1])

                batch_wise_true_labels.append(y.tolist())
                batch_wise_predictions.append(prediction.tolist())

        all_true_labels = list(chain.from_iterable(batch_wise_true_labels))
        all_predictions = list(chain.from_iterable(batch_wise_predictions))

        print(f"Evaluation loss: {running_loss}")
        print(all_true_labels)
        print(all_predictions)
        # print("Classification report after epoch:")

        # ignore the padding item
        # pad_index = labels[]

        # non_padding_labels = [
        #     [
        #         idx2word[label]
        #         for j, label in enumerate(all_true_labels[i])
        #         if all_true_labels[i][j] != pad_index
        #     ]
        #     for i in range(len(all_true_labels))
        # ]
        # non_padding_predictions = [
        #     [
        #         idx2word[label]
        #         for j, label in enumerate(all_predictions[i])
        #         if all_true_labels[i][j] != pad_index
        #     ]
        #     for i in range(len(all_true_labels))
        # ]

        # # print(non_padding_predictions)
        # # print(non_padding_labels)
        # report = classification_report(
        #     non_padding_labels,
        #     non_padding_predictions,
        #     mode="strict",
        # )
        # print(report)
        return loss_history, running_loss_history, report

    def fit(self, train_loader, valid_loader, labels, n_epochs=10):
        """ Train the model

        Parameters
        ----------
        train_loader : DataLoader
            Class to load training data

        valid_loader : DataLoader
            Class to load validation data

        labels : dict
            Dict of index to output classes

        n_epochs : int
            Number of epochs to run

        Returns
        -------
        None
        """
        self._print_summary()

        train_losses = []
        train_running_losses = []

        valid_losses = []
        valid_running_losses = []

        # start the training
        for i in range(n_epochs):
            print(f"Epoch number {i}")
            # Record the loss from both training and validation
            loss_hist, running_loss_hist = self.train(train_loader)

            valid_loss_hist, valid_running_loss_hist, report = self.evaluate(
                valid_loader, labels
            )

            train_losses.append(loss_hist)
            train_running_losses.append(running_loss_hist)

            valid_losses.append(valid_loss_hist)
            valid_running_losses.append(valid_running_loss_hist)
