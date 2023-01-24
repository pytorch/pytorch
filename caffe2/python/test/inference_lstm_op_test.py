#!/usr/bin/env python3

import hypothesis.strategies as st
import numpy as np
import torch
from caffe2.python import core
from caffe2.python.test_util import TestCase
from hypothesis import given, settings
from torch import nn


class TestC2LSTM(TestCase):
    @given(
        bsz=st.integers(1, 5),
        seq_lens=st.integers(1, 6),
        emb_lens=st.integers(5, 10),
        hidden_size=st.integers(3, 7),
        num_layers=st.integers(1, 4),
        has_biases=st.booleans(),
        is_bidirectional=st.booleans(),
        batch_first=st.booleans(),
    )
    @settings(deadline=10000)
    def test_c2_lstm(
        self,
        bsz,
        seq_lens,
        emb_lens,
        hidden_size,
        num_layers,
        has_biases,
        is_bidirectional,
        batch_first,
    ):
        net = core.Net("test_net")
        num_directions = 2 if is_bidirectional else 1
        py_lstm = nn.LSTM(
            emb_lens,
            hidden_size,
            batch_first=batch_first,
            bidirectional=is_bidirectional,
            bias=has_biases,
            num_layers=num_layers,
        )

        hx = np.zeros((num_layers * num_directions, bsz, hidden_size), dtype=np.float32)

        if batch_first:
            inputs = np.random.randn(bsz, seq_lens, emb_lens).astype(np.float32)
        else:
            inputs = np.random.randn(seq_lens, bsz, emb_lens).astype(np.float32)

        py_results = py_lstm(torch.from_numpy(inputs))
        lstm_in = [
            torch.from_numpy(inputs),
            torch.from_numpy(hx),
            torch.from_numpy(hx),
        ] + [param.detach() for param in py_lstm._flat_weights]

        c2_results = torch.ops._caffe2.InferenceLSTM(
            lstm_in, num_layers, has_biases, batch_first, is_bidirectional
        )

        np.testing.assert_array_almost_equal(
            py_results[0].detach().numpy(), c2_results[0].detach().numpy()
        )
        np.testing.assert_array_almost_equal(
            py_results[1][0].detach().numpy(), c2_results[1].detach().numpy()
        )
        np.testing.assert_array_almost_equal(
            py_results[1][1].detach().numpy(), c2_results[2].detach().numpy()
        )
