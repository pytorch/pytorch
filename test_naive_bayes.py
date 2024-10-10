"""Unit tests for NaiveBayesClassifier and helper functions.

Author: Nandana Chigaterappa HemanthKumar
Version: 1.0
"""

import unittest
import torch
from naive_bayes_text_classification import (
    NaiveBayesClassifier,
    text_to_bow,
    build_vocab,
    prepare_data,
    load_dataset,
)


class TestNaiveBayesClassifier(unittest.TestCase):
    """Unit tests for the Naive Bayes Classifier."""

    def setUp(self):
        """Sets up the dataset, vocabulary, and model for the tests."""
        print("\nSetting up dataset, vocabulary, and model...")
        self.train_iter, self.test_iter = load_dataset("IMDB")
        self.vocab = build_vocab(self.train_iter)
        self.model = NaiveBayesClassifier(len(self.vocab))
        self.X_train, self.y_train = prepare_data(self.train_iter, self.vocab)
        self.X_test, self.y_test = prepare_data(self.test_iter, self.vocab)

    def test_empty_vocab(self):
        """Test vocabulary building with empty dataset."""
        print("\nRunning test for empty vocabulary...")
        with self.assertRaises(ValueError):
            build_vocab([])

    def test_empty_text_to_bow(self):
        """Test that empty text returns a BoW vector of zeros."""
        print("\nRunning test for empty text to BoW conversion...")
        bow = text_to_bow("", self.vocab)
        print(f"BoW Vector for empty text: {bow}")
        self.assertTrue(torch.all(bow == 0))

    def test_empty_string_prediction(self):
        """Test that the model handles empty strings gracefully during prediction."""
        print("\nRunning test for empty string prediction...")
        self.model.fit(self.X_train, self.y_train)
        bow = text_to_bow("", self.vocab)
        print(f"BoW Vector for empty string: {bow}")
        try:
            self.model.predict(bow.unsqueeze(0))
        except ValueError as e:
            print(f"Exception raised for empty string: {str(e)}")

    def test_positive_text_to_bow(self):
        """Test that a positive sentence converts to a non-zero BoW vector."""
        print("\nRunning test for positive text to BoW conversion...")
        positive_text = "This movie was amazing, I loved it!"
        bow = text_to_bow(positive_text, self.vocab)
        print(f"BoW Vector for positive text: {bow}")
        self.assertTrue(torch.any(bow > 0))

        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(bow.unsqueeze(0)).item()
        sentiment = "Positive" if pred == 2 else "Negative"
        print(f"Text: '{positive_text}' -> Classified as {sentiment}")

    def test_negative_text_to_bow(self):
        """Test that a negative sentence converts to a non-zero BoW vector."""
        print("\nRunning test for negative text to BoW conversion...")
        negative_text = "This movie was awful, I hated it!"
        bow = text_to_bow(negative_text, self.vocab)
        print(f"BoW Vector for negative text: {bow}")
        self.assertTrue(torch.any(bow > 0))

        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(bow.unsqueeze(0)).item()
        sentiment = "Positive" if pred == 2 else "Negative"
        print(f"Text: '{negative_text}' -> Classified as {sentiment}")

    def test_combined_text_to_bow(self):
        """Test that combined positive and negative words convert to a non-zero BoW vector."""
        print("\nRunning test for combined text to BoW conversion...")
        combined_text = "I loved the movie, but the ending was awful."
        bow = text_to_bow(combined_text, self.vocab)
        print(f"BoW Vector for combined text: {bow}")
        self.assertTrue(torch.any(bow > 0))

        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(bow.unsqueeze(0)).item()
        sentiment = "Positive" if pred == 2 else "Negative"
        print(f"Text: '{combined_text}' -> Classified as {sentiment}")

    def test_long_text_to_bow(self):
        """Test that a long piece of text can be converted to a BoW vector."""
        print("\nRunning test for long text to BoW conversion...")
        long_text = (
            "This movie had an intriguing plot and a cast of complex characters, "
            "but the pacing was slow, and I found myself losing interest halfway through. "
            "The cinematography was beautiful, but the lack of coherent storytelling made the movie hard to follow. "
            "Overall, it was a mixed experience—some parts were great, while others felt underdeveloped and tedious."
        )
        bow = text_to_bow(long_text, self.vocab)
        print(f"BoW Vector for long text: {bow}")
        self.assertTrue(torch.any(bow > 0))

        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(bow.unsqueeze(0)).item()
        sentiment = "Positive" if pred == 2 else "Negative"
        print(f"Text: '{long_text[:50]}...' -> Classified as {sentiment}")

    def test_training_process(self):
        """Test the model training process and log probabilities."""
        print("\nRunning test for model training process...")
        self.model.fit(self.X_train, self.y_train)
        print(f"Class Log Prior: {self.model.class_log_prior}")
        print(f"Feature Log Prob Shape: {self.model.feature_log_prob.shape}")
        self.assertEqual(self.model.class_log_prior.shape[0], 2)
        self.assertEqual(self.model.feature_log_prob.shape, (2, len(self.vocab)))

    def test_predictions_on_valid_data(self):
        """Test that predictions return valid class labels (1 or 2)."""
        print("\nRunning test for predictions on valid data...")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print(f"Predicted Class Labels: {y_pred}")
        self.assertTrue(torch.all((y_pred == 1) | (y_pred == 2)))

    def test_prediction_shape(self):
        """Test that predictions have the same shape as the input labels."""
        print("\nRunning test for prediction shape...")
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print(f"Prediction Shape: {y_pred.shape}, Expected Shape: {self.y_test.shape}")
        self.assertEqual(y_pred.shape, self.y_test.shape)


if __name__ == "__main__":
    unittest.main()
