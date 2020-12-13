#!/usr/bin/env python3


import torch.nn as nn
from collections import Counter
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch
from fever.scorer import fever_score


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

        progress = tqdm(enumerate(loader), total=len(loader))
        for i, batch in progress:

            # Zero out the gradient
            self.optimizer.zero_grad()

            # Split up the batch
            X, starting_indices, y, labels = batch

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
            running_loss = sum(loss_history[-5:]) / 5
            running_loss_history.append(running_loss)

            # Log the running loss
            progress.set_description(f"Loss: {loss}\tRunning loss: {running_loss}")
            # if self.log_every_n and i % self.log_every_n == 0:
            # print(f"predictions: {torch.argmax(logits, dim=-1).tolist()}")
            # print(f"y: {y.view(-1).tolist()}")

            # Backpropogation
            loss.backward()

            # Clip the norms to prevent explosion
            nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

            # update step
            self.optimizer.step()
            progress.update(1)

        print("Epoch completed!")
        print(f"Epoch Loss: {running_loss}")

        return loss_history, running_loss_history

    def evaluate(self, loader, batch_labels):
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
            for i, batch in tqdm(enumerate(loader), total=len(loader)):
                # Split up the batch
                X, indices, y, _ = batch

                # Foward
                logits = self.model(X.to(self.device))
                og_shape = logits.shape

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

                # get the output class from the probs
                # also, reshape the prediction back to sentences
                prediction = torch.argmax(probs, dim=-1).reshape(og_shape[:-1])

                batch_wise_true_labels.append(y.tolist())
                batch_wise_predictions.append(prediction.tolist())

        all_true_labels = list(chain.from_iterable(batch_wise_true_labels))
        all_predictions = list(chain.from_iterable(batch_wise_predictions))

        print(f"Evaluation loss: {running_loss}")
        print("Classification report after epoch:")

        # ignore the padding item
        pad_index = 3

        true_sequences = [
            [
                label
                for j, label in enumerate(all_true_labels[i])
                if all_true_labels[i][j] != pad_index
            ]
            for i in range(len(all_true_labels))
        ]
        pred_sequences = [
            [
                label
                for j, label in enumerate(all_predictions[i])
                if all_true_labels[i][j] != pad_index
            ]
            for i in range(len(all_true_labels))
        ]

        fever_score = 0
        label_accuracy = 0
        for true, pred in zip(true_sequences, pred_sequences):
            fev, acc = self.fever(true, pred)
            fever_score += fev
            label_accuracy += acc

        # # true/pred sequences
        is_equal = [a == b for a, b in zip(true_sequences, pred_sequences)]
        print(f"Evidence accuracy: {sum(is_equal) / len(is_equal)}")
        print(f"Number correct: {sum(is_equal)} out of {len(is_equal)}")
        print(f"Fever score: {fever_score / len(is_equal)}")
        print(f"Number right: {fever_score} out of {len(is_equal)}")
        print(f"Label accuracy: {label_accuracy / len(is_equal)}")

        # print(Counter(non_padding_predictions))
        # print(Counter(non_padding_labels))
        # print(list(labels.values()))
        # report = classification_report(non_padding_labels, non_padding_predictions,)
        # print(report)
        return loss_history, running_loss_history

    def fever(self, true, pred):
        # only care about the selected sentences
        true_sentences = {i for i, label in enumerate(true) if label != 1}
        pred_sentences = {i for i, label in enumerate(pred) if label != 1}
        # calculate the label
        c = Counter(pred)
        pred_label = 2 if c[2] > c[0] else 0 if c[0] > c[2] else 1
        c = Counter(true)
        true_label = 2 if c[2] > c[0] else 0 if c[0] > c[2] else 1

        is_correct = true_label == pred_label
        has_evidence = true_sentences <= pred_sentences

        return is_correct and has_evidence, is_correct

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

            valid_loss_hist, valid_running_loss_hist = self.evaluate(
                valid_loader, labels
            )

            train_losses.append(loss_hist)
            train_running_losses.append(running_loss_hist)

            valid_losses.append(valid_loss_hist)
            valid_running_losses.append(valid_running_loss_hist)


class FastTrainer:
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

    def train(self, loader, path):
        """ Run a single epoch of training

        Parameters
        ----------
        loader : DataLoader
            DataLoader for a dataset to train
        path : str
            Path to location to save model every n batches

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

        progress = tqdm(enumerate(loader), total=len(loader))
        for i, batch in progress:

            # Zero out the gradient
            self.optimizer.zero_grad()

            # Split up the batch
            X, y, json = batch

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
            running_loss = sum(loss_history) / len(loss_history)
            running_loss_history.append(running_loss)

            # Log the running loss
            progress.set_description(f"Loss: {loss}\tRunning loss: {running_loss}")

            if self.log_every_n and i % self.log_every_n == 0:
                torch.save(
                    self.model.state_dict(),
                    path
                )

            # Backpropogation
            loss.backward()

            # Clip the norms to prevent explosion
            nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)

            # update step
            self.optimizer.step()
            progress.update(1)

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

        jsons = []

        loss_history = []
        running_loss = 0.0
        running_loss_history = []

        # don't compute gradient
        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader), total=len(loader)):
                # Split up the batch
                X, y, json_list = batch

                # Foward
                logits = self.model(X.to(self.device))
                og_shape = logits.shape

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

                # get the output class from the probs
                # also, reshape the prediction back to sentences
                predictions = torch.argmax(probs, dim=-1).reshape(og_shape[:-1])

                for pred, json in zip(predictions.tolist(), json_list):
                    c = Counter(pred)
                    # most common value, or 1 (NEI) if it's a tie
                    most_common = 2 if c[2] > c[0] else 0 if c[0] > c[2] else 1
                    json["predicted_label"] = labels[most_common]
                    json["label"] = labels[json["label"]]
                    jsons.append(json)

        # print(f"Evaluation loss: {running_loss}")
        # print("Classification report after epoch:")
        strict_score, label_accuracy, precision, recall, f1 = fever_score(jsons)
        print(f"Fever score: {strict_score}")
        print(f"Label accuracy: {label_accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        return loss_history, running_loss_history

    def evaluate_sentence_selection(self, loader, labels):
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
        jsons = []
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            X, y, json_list = batch
            for json in json_list:
                json["predicted_label"] = labels[json["predicted_label"]]
                json["label"] = labels[json["label"]]
                jsons.append(json)

        # print(f"Evaluation loss: {running_loss}")
        # print("Classification report after epoch:")
        strict_score, label_accuracy, precision, recall, f1 = fever_score(jsons)
        print(f"Fever score: {strict_score}")
        print(f"Label accuracy: {label_accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        return jsons

    def fit(self, train_loader, valid_loader, labels, path, n_epochs=10):
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
            loss_hist, running_loss_hist = self.train(train_loader, path)

            valid_loss_hist, valid_running_loss_hist = self.evaluate(
                valid_loader, labels
            )

            train_losses.append(loss_hist)
            train_running_losses.append(running_loss_hist)

            valid_losses.append(valid_loss_hist)
            valid_running_losses.append(valid_running_loss_hist)

        return train_running_losses, valid_running_losses
