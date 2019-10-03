#!/usr/bin/env python3
import regex
import torch
from common_utils import TestCase

lib_path = "//caffe2/torch/nlp/operators:gpt2_bpe_tokenizer"
torch.ops.load_library(lib_path)


class GPT2BPT2(TestCase):

    def test_gpt2_bpe_tokenizer(self):
        # Use in TorchScript
        @torch.jit.script
        def tokenize(text: str):
            return torch.ops.internal.gpt2_bpe_tokenizer(text)

        def py_tokenize(text):
            pat = regex.compile(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            )
            return regex.findall(pat, text)

        vocab_path = (
            "caffe2/torch/nlp/operators/resources/"
            "test_gpt2_bpe_tokenizer_input.txt"
        )
        with open(vocab_path, "r") as f:
            for line in f:
                pcre_tokens = tokenize(line[:-1])
                py_tokens = py_tokenize(line[:-1])
                self.assertEqual(pcre_tokens, py_tokens)
