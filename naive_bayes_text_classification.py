"""Naive Bayes Text Classification using PyTorch and TorchText.

Author: Nandana Chigaterappa HemanthKumar
Version: 1.0
"""

import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from typing import List, Tuple


class NaiveBayesClassifier:
    """Naive Bayes classifier for binary text classification.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        class_log_prior (torch.Tensor): Log prior probabilities for the classes.
        feature_log_prob (torch.Tensor): Log likelihood of features (words) given classes.
    """

    def __init__(self, vocab_size: int):
        """Initializes the Naive Bayes Classifier.

        Args:
            vocab_size (int): The number of unique words in the vocabulary.
        """
        self.vocab_size = vocab_size
        self.class_log_prior = torch.zeros(2)
        self.feature_log_prob = torch.zeros((2, vocab_size))

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Trains the Naive Bayes classifier by calculating class prior and feature probabilities.

        Args:
            X (torch.Tensor): Input feature matrix where each row is a bag-of-words vector.
            y (torch.Tensor): Class labels corresponding to each input sample.

        Raises:
            ValueError: If the training data is empty or the class counts are zero.
        """
        n_samples, n_features = X.shape
        n_classes = 2

        if n_samples == 0 or n_features == 0:
            raise ValueError("Training data must not be empty.")

        class_count = torch.bincount(
            y - 1, minlength=2
        )  # Subtract 1 to convert labels 1,2 to 0,1
        if class_count.sum() == 0:
            raise ValueError("Class count cannot be zero.")

        self.class_log_prior = torch.log(class_count / n_samples)

        feature_count = torch.zeros((n_classes, n_features))
        for i in range(n_samples):
            feature_count[y[i] - 1] += X[i]

        # Add-1 smoothing to avoid zero probabilities
        smoothed_fc = feature_count + 1
        smoothed_cc = class_count.unsqueeze(1) + 2

        self.feature_log_prob = torch.log(smoothed_fc / smoothed_cc)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predicts class labels for given input data.

        Args:
            X (torch.Tensor): Input feature matrix where each row is a bag-of-words vector.

        Returns:
            torch.Tensor: Predicted class labels (1 for negative, 2 for positive).

        Raises:
            ValueError: If the input data is empty.
        """
        if X.numel() == 0 or X.sum() == 0:
            raise ValueError("Input data for prediction cannot be empty.")
        log_prob = self._joint_log_likelihood(X)
        return log_prob.argmax(dim=1) + 1

    def _joint_log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the joint log likelihood of features for each class.

        Args:
            X (torch.Tensor): Input feature matrix.

        Returns:
            torch.Tensor: Joint log likelihood for each class.
        """
        return X @ self.feature_log_prob.T + self.class_log_prior


def build_vocab(data: List[Tuple[int, str]], max_size: int = 25000) -> dict:
    """Builds a vocabulary from the given dataset.

    Args:
        data (List[Tuple[int, str]]): List of (label, text) tuples.
        max_size (int): Maximum size of the vocabulary.

    Returns:
        dict: Mapping from word to unique index in the vocabulary.

    Raises:
        ValueError: If the dataset is empty and no vocabulary can be built.
    """
    counter = Counter()
    tokenizer = get_tokenizer("basic_english")
    for _, text in data:
        counter.update(tokenizer(text.lower()))
    if not counter:
        raise ValueError("Vocabulary could not be built from empty dataset.")
    return {word: idx for idx, (word, _) in enumerate(counter.most_common(max_size))}


def text_to_bow(text: str, vocab: dict) -> torch.Tensor:
    """Converts a text into a Bag of Words (BoW) vector.

    Args:
        text (str): Input text to convert.
        vocab (dict): Mapping of word to unique index in the vocabulary.

    Returns:
        torch.Tensor: Bag-of-Words vector where each index corresponds to the word count in the text.
    """
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(text.lower())
    bow = torch.zeros(len(vocab))
    for token in tokens:
        if token in vocab:
            bow[vocab[token]] += 1
    return bow


def prepare_data(
    data: List[Tuple[int, str]], vocab: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares the dataset for training or testing.

    Args:
        data (List[Tuple[int, str]]): List of (label, text) tuples.
        vocab (dict): Vocabulary dictionary.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of feature matrix (X) and label vector (y).
    """
    X, y = [], []
    for label, text in data:
        X.append(text_to_bow(text, vocab))
        y.append(label)
    return torch.stack(X), torch.tensor(y)


def load_dataset(
    dataset_name: str = "IMDB",
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Loads the specified dataset. Currently supports IMDB.

    Args:
        dataset_name (str): The name of the dataset to load (IMDB).

    Returns:
        Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]: Training and testing datasets.

    Raises:
        ValueError: If the dataset is not supported.
    """
    if dataset_name == "IMDB":
        train_iter = list(torchtext.datasets.IMDB(split="train"))
        test_iter = list(torchtext.datasets.IMDB(split="test"))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return train_iter, test_iter


def train_and_evaluate(dataset_name: str = "IMDB"):
    """Trains the Naive Bayes model and evaluates it on the test dataset.

    Args:
        dataset_name (str): The name of the dataset to train and evaluate on.

    Returns:
        float: The accuracy of the model on the test set.
    """
    print(f"Loading {dataset_name} dataset...")
    train_iter, test_iter = load_dataset(dataset_name)

    print("Building vocabulary...")
    vocab = build_vocab(train_iter)
    print(f"Vocabulary size: {len(vocab)}")

    print("Preparing training data...")
    X_train, y_train = prepare_data(train_iter, vocab)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    print("Preparing test data...")
    X_test, y_test = prepare_data(test_iter, vocab)
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    # Train the model
    print("Training the Naive Bayes model...")
    model = NaiveBayesClassifier(len(vocab))
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")

    return accuracy.item()


if __name__ == "__main__":
    accuracy = train_and_evaluate()
