#!/usr/bin/env python3
import hypothesis.strategies as st
import numpy as np
import torch
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
from hypothesis import given
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

    def test_c2_lstm_empty_input(self):
        seq_lens = 0
        bsz = 1
        emb_lens = 200
        hidden_size = 128
        num_layers = 2
        is_bidirectional = True
        num_directions = 2

        net = core.Net("test_net")
        py_lstm = nn.LSTM(emb_lens, hidden_size, num_layers=num_layers)
        hx = np.zeros((num_layers * num_directions, bsz, hidden_size), dtype=np.float32)
        params = py_lstm._flat_weights
        inputs = np.random.randn(seq_lens, bsz, emb_lens).astype(np.float32)

        inputs_blob = net.AddExternalInput("inputs")
        workspace.FeedBlob(str(inputs_blob), inputs)
        hx_blob = net.AddExternalInput("hx")
        workspace.FeedBlob(str(hx_blob), hx)
        lstm_in_blobs = [inputs_blob, hx_blob, hx_blob]
        for i, param in enumerate(params):
            param_blob = net.AddExternalInput("param_{}".format(i))
            workspace.FeedBlob(str(param_blob), param.detach().numpy())
            lstm_in_blobs.append(param_blob)
        outputs, hidden, cell = net.InferenceLSTM(
            lstm_in_blobs,
            bidirectional=is_bidirectional,
            num_layers=num_layers,
            outputs=3,
        )
        workspace.CreateNet(net)
        with self.assertRaises(RuntimeError):
            workspace.RunNet(net)
