from __future__ import absolute_import, division, print_function, unicode_literals
from test_pytorch_common import TestCase, run_tests

import torch
import torch.onnx
from torch.onnx import utils

import io


class TestUtilityFuns(TestCase):

    def test_is_in_onnx_export(self):
        test_self = self

        class MyModule(torch.nn.Module):
            def forward(self, x):
                test_self.assertTrue(torch.onnx.is_in_onnx_export())
                raise ValueError
                return x + 1

        x = torch.randn(3, 4)
        f = io.BytesIO()
        try:
            torch.onnx.export(MyModule(), x, f)
        except ValueError:
            self.assertFalse(torch.onnx.is_in_onnx_export())

    def test_constant_fold_lstm(self):
        class GruNet(torch.nn.Module):
            def __init__(self):
                super(GruNet, self).__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            def forward(self, input, initial_state):
                return self.mygru(input, initial_state)

        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)
        graph, _, __ = utils._model_to_graph(GruNet(), (input, h0), None,
                                             do_constant_folding=True)
        for node in graph.nodes():
            assert node.kind != "onnx::Slice"
            assert node.kind != "onnx::Concat"
            assert node.kind != "onnx::Unsqueeze"
        assert len(list(graph.nodes())) == 3

    def test_constant_fold_transpose(self):
        class MatMulNet(torch.nn.Module):
            def __init__(self):
                super(MatMulNet, self).__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            def forward(self, A):
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        model = MatMulNet()
        A = torch.randn(2, 3)
        graph, _, __ = utils._model_to_graph(MatMulNet(), (A), None,
                                             do_constant_folding=True)
        for node in graph.nodes():
            assert node.kind != "onnx::Transpose"
        assert len(list(graph.nodes())) == 1

if __name__ == '__main__':
    run_tests()
