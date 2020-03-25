from __future__ import absolute_import, division, print_function, unicode_literals
from test_pytorch_common import TestCase, run_tests

import torch
import torch.onnx
from torch.onnx import utils, OperatorExportTypes
from torch.onnx.symbolic_helper import _set_opset_version, _set_operator_export_type

import onnx

import io
import copy
import unittest


skip = unittest.skip


class TestUtilityFuns(TestCase):
    opset_version = 9

    def setUp(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

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
            torch.onnx.export(MyModule(), x, f, opset_version=self.opset_version)
        except ValueError:
            self.assertFalse(torch.onnx.is_in_onnx_export())

    def test_validate_dynamic_axes_invalid_input_output_name(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils._validate_dynamic_axes({'input1': {}, 'output': {},
                                         'invalid_name1': {}, 'invalid_name2': {}},
                                         None, ['input1', 'input2'], ['output'])
            messages = [str(warning.message) for warning in w]
        assert "Provided key invalid_name1 for dynamic axes is not a valid input/output name" in messages
        assert "Provided key invalid_name2 for dynamic axes is not a valid input/output name" in messages
        assert len(messages) == 2

    def test_constant_fold_transpose(self):
        class TransposeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = torch.transpose(a, 1, 0)
                return b + x

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        x = torch.ones(3, 2)
        graph, _, __ = utils._model_to_graph(TransposeModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Transpose"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_slice(self):
        class NarrowModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = torch.narrow(a, 0, 0, 1)
                return b + x

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        x = torch.ones(1, 3)
        graph, _, __ = utils._model_to_graph(NarrowModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_slice_index_exceeds_dim(self):
        class SliceIndexExceedsDimModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = a[1:10]         # index exceeds dimension
                return b + x

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        x = torch.ones(1, 3)
        graph, _, __ = utils._model_to_graph(SliceIndexExceedsDimModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True,
                                             operator_export_type=OperatorExportTypes.ONNX)

        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_slice_negative_index(self):
        class SliceNegativeIndexModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = a[0:-1]        # index relative to the end
                return b + x

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        x = torch.ones(1, 3)
        graph, _, __ = utils._model_to_graph(SliceNegativeIndexModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_unsqueeze(self):
        class UnsqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
                b = torch.unsqueeze(a, 0)
                return b + x

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        x = torch.ones(1, 2, 3)
        graph, _, __ = utils._model_to_graph(UnsqueezeModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Unsqueeeze"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_concat(self):
        class ConcatModule(torch.nn.Module):
            def forward(self, x):
                # Why did I insert a Cast here?  There appears to be intentional
                # behavior in ONNX constant folding where constant tensors which
                # are not attached to any known to be foldable onnx
                # operations don't get extracted into the initializer graph.  So
                # without these casts, we will actually fail to pull out one of
                # the constants, thus failing constant folding.  I think the
                # test is wrong but I don't have time to write a more correct
                # test (I think the right way to go about the test is to setup
                # a predicate for what invariant graphs should hold after
                # constant folding, and then verify this predicate holds.
                # I think the asserts below are an attempt at this predicate,
                # but it is not right!)
                #
                # More commentary at
                # https://github.com/pytorch/pytorch/pull/18698/files#r340107552
                a = torch.tensor([[1., 2., 3.]]).to(torch.float)
                b = torch.tensor([[4., 5., 6.]]).to(torch.float)
                c = torch.cat((a, b), 0)
                d = b + c
                return x + d

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        x = torch.ones(2, 3)
        graph, _, __ = utils._model_to_graph(ConcatModule(), (x, ),
                                             do_constant_folding=True,
                                             _disable_torch_constant_prop=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Concat"
            assert node.kind() != "onnx::Cast"
            assert node.kind() != "onnx::Constant"
        assert len(list(graph.nodes())) == 2

    def test_constant_fold_lstm(self):
        class GruNet(torch.nn.Module):
            def __init__(self):
                super(GruNet, self).__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            def forward(self, input, initial_state):
                return self.mygru(input, initial_state)

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)
        graph, _, __ = utils._model_to_graph(GruNet(), (input, h0),
                                             do_constant_folding=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Slice"
            assert node.kind() != "onnx::Concat"
            assert node.kind() != "onnx::Unsqueeze"
        assert len(list(graph.nodes())) == 3

    def test_constant_fold_transpose_matmul(self):
        class MatMulNet(torch.nn.Module):
            def __init__(self):
                super(MatMulNet, self).__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            def forward(self, A):
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        A = torch.randn(2, 3)
        graph, _, __ = utils._model_to_graph(MatMulNet(), (A),
                                             do_constant_folding=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Transpose"
        assert len(list(graph.nodes())) == 1

    # TODO we need to figure out the root cause and fix the problem
    @skip("causing segmentation fault")
    def test_constant_fold_reshape(self):
        class ReshapeModule(torch.nn.Module):
            def __init__(self, ):
                super(ReshapeModule, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                b = self.weight.reshape(1, -1, 1, 1)
                return x * b

        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        x = torch.randn(4, 5)
        graph, _, __ = utils._model_to_graph(ReshapeModule(), (x, ), do_constant_folding=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Reshape"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_div(self):
        class Module(torch.nn.Module):
            def __init__(self, ):
                super(Module, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                div = self.weight.div(torch.tensor([1, 2, 3, 4, 5]))
                return div * x

        x = torch.randn(2, 5)
        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        graph, _, __ = utils._model_to_graph(Module(), (x, ), do_constant_folding=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Div"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_mul(self):
        class Module(torch.nn.Module):
            def __init__(self, ):
                super(Module, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                mul = self.weight.mul(torch.tensor([1, 2, 3, 4, 5]))
                return mul / x

        x = torch.randn(2, 5)
        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        graph, _, __ = utils._model_to_graph(Module(), (x, ), do_constant_folding=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Mul"
        assert len(list(graph.nodes())) == 1

    def test_constant_fold_sqrt(self):
        class Module(torch.nn.Module):
            def __init__(self, ):
                super(Module, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                sqrt = torch.sqrt(self.weight)
                return sqrt / x

        x = torch.randn(2, 5)
        _set_opset_version(self.opset_version)
        _set_operator_export_type(OperatorExportTypes.ONNX)
        graph, _, __ = utils._model_to_graph(Module(), (x, ), do_constant_folding=True,
                                             operator_export_type=OperatorExportTypes.ONNX)
        for node in graph.nodes():
            assert node.kind() != "onnx::Sqrt"
        assert len(list(graph.nodes())) == 1

    def test_strip_doc_string(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                return torch.exp(input)
        x = torch.randn(3, 4)

        def is_model_stripped(f, strip_doc_string=None):
            if strip_doc_string is None:
                torch.onnx.export(MyModule(), x, f, opset_version=self.opset_version)
            else:
                torch.onnx.export(MyModule(), x, f, strip_doc_string=strip_doc_string,
                                  opset_version=self.opset_version)
            model = onnx.load(io.BytesIO(f.getvalue()))
            model_strip = copy.copy(model)
            onnx.helper.strip_doc_string(model_strip)
            return model == model_strip

        # test strip_doc_string=True (default)
        self.assertTrue(is_model_stripped(io.BytesIO()))
        # test strip_doc_string=False
        self.assertFalse(is_model_stripped(io.BytesIO(), False))

    # NB: remove this test once DataParallel can be correctly handled
    def test_error_on_data_parallel(self):
        model = torch.nn.DataParallel(torch.nn.ReflectionPad2d((1, 2, 3, 4)))
        x = torch.randn(1, 2, 3, 4)
        f = io.BytesIO()
        with self.assertRaisesRegex(ValueError,
                                    'torch.nn.DataParallel is not supported by ONNX '
                                    'exporter, please use \'attribute\' module to '
                                    'unwrap model from torch.nn.DataParallel. Try '):
            torch.onnx.export(model, x, f, opset_version=self.opset_version)


# opset 10 tests
TestUtilityFuns_opset10 = type(str("TestUtilityFuns_opset10"),
                               (TestCase,),
                               dict(TestUtilityFuns.__dict__, opset_version=10))


# opset 11 tests
TestUtilityFuns_opset11 = type(str("TestUtilityFuns_opset11"),
                               (TestCase,),
                               dict(TestUtilityFuns.__dict__, opset_version=11))


if __name__ == '__main__':
    run_tests()
