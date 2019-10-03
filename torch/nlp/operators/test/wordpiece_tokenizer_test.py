#!/usr/bin/env python3

import ctypes
import logging
import os

import numpy as np
import torch
from common_utils import TestCase


lib_path = "//caffe2/torch/nlp/operators:wordpiece_tokenizer"
torch.ops.load_library(lib_path)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def to_utf8_list(texts):
    return [s.encode("utf-8") for s in texts]


def read_vocab(vocab_path):
    d = {}
    with open(vocab_path) as f:
        for i, line in enumerate(f.readlines()):
            d[line.strip()] = i
    return d


class WordpieceTokenizerTest(TestCase):
    def test_wordpiece_tokenizer(self):
        # Use in TorchScript
        @torch.jit.script
        def fn(text, my_dict, max_seq_len):
            # type: (List[str], Dict[str, int], int) -> List[Tensor]
            return torch.ops.internal.wordpiece_tokenizer(text, my_dict, max_seq_len)

        test_input = to_utf8_list(
            [
                "unknownword Ernie and^   bert   ",
                "Computer Repair. Viewed by 0 people Do you have a old computer just laying around?",
                "Earrings and necklace 2018. Earrings and necklace...",
            ]
        )
        max_seq_len = 17
        vocab_path = (
            "caffe2/torch/nlp/operators/resources/"
            "test_wordpiece_tokenizer_ops_test_vocab.txt"
        )
        true_tokens = [
            "[CLS] [UNK] Ernie and ^ b ##ert [SEP] " + " ".join(["[PAD]"] * 9),
            "[CLS] Computer Rep ##air . View ##ed by 0 people Do you have a old computer [SEP]",
            "[CLS] E ##ar ##rings and necklace 2018 . E ##ar ##rings and necklace . . . [SEP]",
        ]

        log.info(fn.graph)

        vocab = read_vocab(vocab_path)
        vocab_inverted_idx = {v: k for k, v in vocab.items()}

        tokens_numberized, token_masks, segmentids = fn(test_input, vocab, max_seq_len)
        tokens_numberized = tokens_numberized.numpy()
        tokens = [[vocab_inverted_idx[i] for i in list(x)] for x in tokens_numberized]
        for i in range(len(test_input)):
            log.info("decoded output: {}".format(" ".join(tokens[i])))
            self.assertTrue(" ".join(tokens[i]) == true_tokens[i])

    def test_wordpiece_pairwise_classification_tokenizer(self):
        # Use in TorchScript
        @torch.jit.script
        def fn(text_a, text_b, my_dict, max_seq_len):
            # type: (List[str], List[str], Dict[str, int], int) -> List[Tensor]
            return torch.ops.internal.wordpiece_pairwise_classification_tokenizer(
                text_a, text_b, my_dict, max_seq_len
            )

        test_input_a = to_utf8_list(["Do you have a old computer"])
        test_input_b = to_utf8_list(["Earrings and necklace"])
        max_seq_len = 16
        vocab_path = (
            "caffe2/torch/nlp/operators/resources/"
            "test_wordpiece_tokenizer_ops_test_vocab.txt"
        )
        true_tokens = ["[CLS] Do you have a old computer [SEP] E ##ar ##rings and necklace [SEP] [PAD] [PAD]"]
        true_input_ids = torch.tensor([[36, 11, 12, 13, 14, 15, 16, 37, 22, 24, 26, 27, 28, 37, 0, 0]])
        true_input_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
        true_segment_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]])

        log.info(fn.graph)

        vocab = read_vocab(vocab_path)
        vocab_inverted_idx = {v: k for k, v in vocab.items()}

        tokens_numberized, token_masks, segmentids = fn(
            test_input_a, test_input_b, vocab, max_seq_len
        )

        log.info(tokens_numberized)
        log.info(token_masks)
        log.info(segmentids)

        self.assertEqual(true_input_ids, tokens_numberized)
        self.assertEqual(token_masks, true_input_mask)
        self.assertEqual(segmentids, true_segment_ids)

        log.info(tokens_numberized)
        log.info(token_masks)
        log.info(segmentids)
        tokens_numberized = tokens_numberized.numpy()
        tokens = [[vocab_inverted_idx[i] for i in list(x)] for x in tokens_numberized]
        for i in range(len(test_input_a)):
            log.info("decoded output: {}".format(" ".join(tokens[i])))
            self.assertTrue(" ".join(tokens[i]) == true_tokens[i])
