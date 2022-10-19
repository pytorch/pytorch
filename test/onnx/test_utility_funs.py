# Owner(s): ["module: onnx"]

import copy
import functools
import io
import warnings
from typing import Callable

import onnx
import parameterized

import torch
import torch.onnx
import torch.utils.cpp_extension
import torchvision
from autograd_helper import CustomFunction as CustomFunction2
from pytorch_test_common import (
    skipIfNoCuda,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
)
from torch.onnx import _constants, OperatorExportTypes, TrainingMode, utils
from torch.onnx._globals import GLOBALS
from torch.onnx.symbolic_helper import _unpack_list, parse_args
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoCaffe2, skipIfNoLapack
from verify import verify


class _BaseTestCase(common_utils.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def _model_to_graph(
        self,
        model,
        input,
        do_constant_folding=True,
        training=TrainingMode.EVAL,
        operator_export_type=OperatorExportTypes.ONNX,
        input_names=None,
        dynamic_axes=None,
    ):
        torch.onnx.utils._setup_trace_module_map(model, False)
        if training == torch.onnx.TrainingMode.TRAINING:
            model.train()
        elif training == torch.onnx.TrainingMode.EVAL:
            model.eval()
        utils._validate_dynamic_axes(dynamic_axes, model, None, None)
        graph, params_dict, torch_out = utils._model_to_graph(
            model,
            input,
            do_constant_folding=do_constant_folding,
            _disable_torch_constant_prop=True,
            operator_export_type=operator_export_type,
            training=training,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
        )
        return graph, params_dict, torch_out


@common_utils.instantiate_parametrized_tests
class TestUnconvertibleOps(common_utils.TestCase):
    """Unit tests for the `unconvertible_ops` function."""

    def setUp(self):
        class EinsumModule(torch.nn.Module):
            def forward(self, x):
                return torch.einsum("ii", x)

        self.einsum_module = EinsumModule()

    def test_it_returns_graph_and_unconvertible_ops_at_lower_opset_version(self):
        x = torch.randn(4, 4)

        # Einsum is supported since opset 12. It should be unconvertible at opset 9.
        graph, unconvertible_ops = utils.unconvertible_ops(
            self.einsum_module, (x,), opset_version=9
        )
        nodes = graph.nodes()
        self.assertEqual(next(nodes).kind(), "prim::Constant")
        self.assertEqual(next(nodes).kind(), "prim::ListConstruct")
        self.assertEqual(next(nodes).kind(), "prim::Constant")
        self.assertEqual(next(nodes).kind(), "aten::einsum")
        self.assertEqual(unconvertible_ops, ["aten::einsum"])

    @common_utils.parametrize(
        "jit_function",
        [
            common_utils.subtest(
                functools.partial(torch.jit.trace, example_inputs=torch.randn(4, 4)),
                name="traced",
            ),
            common_utils.subtest(torch.jit.script, name="scripted"),
        ],
    )
    def test_it_returns_unconvertible_ops_at_lower_opset_version_for_jit_module(
        self, jit_function: Callable
    ):
        module = jit_function(self.einsum_module)
        x = torch.randn(4, 4)

        # Einsum is supported since opset 12. It should be unconvertible at opset 9.
        _, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=9)
        self.assertEqual(unconvertible_ops, ["aten::einsum"])

    @common_utils.parametrize(
        "jit_function",
        [
            common_utils.subtest(lambda x: x, name="nn_module"),
            common_utils.subtest(
                functools.partial(torch.jit.trace, example_inputs=torch.randn(4, 4)),
                name="traced",
            ),
            common_utils.subtest(torch.jit.script, name="scripted"),
        ],
    )
    def test_it_returns_empty_list_when_all_ops_convertible(
        self, jit_function: Callable
    ):
        module = jit_function(self.einsum_module)
        x = torch.randn(4, 4)

        # Einsum is supported since opset 12
        _, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=12)
        self.assertEqual(unconvertible_ops, [])


@parameterized.parameterized_class(
    [
        {"opset_version": opset}
        for opset in range(_constants.ONNX_BASE_OPSET, _constants.ONNX_MAX_OPSET + 1)
    ],
    class_name_func=lambda cls, num, params_dict: f"{cls.__name__}_opset_{params_dict['opset_version']}",
)
class TestUtilityFuns(_BaseTestCase):
    opset_version = None

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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils._validate_dynamic_axes(
                {"input1": {}, "output": {}, "invalid_name1": {}, "invalid_name2": {}},
                None,
                ["input1", "input2"],
                ["output"],
            )
            messages = [str(warning.message) for warning in w]
        self.assertIn(
            "Provided key invalid_name1 for dynamic axes is not a valid input/output name",
            messages,
        )
        self.assertIn(
            "Provided key invalid_name2 for dynamic axes is not a valid input/output name",
            messages,
        )
        self.assertEqual(len(messages), 2)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_to_slice(self):
        class SplitModule(torch.nn.Module):
            def forward(self, x, y, t):
                splits = (x.size(1), y.size(1))
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        graph, _, _ = self._model_to_graph(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::SplitToSequence")

    def test_constant_fold_transpose(self):
        class TransposeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.transpose(a, 1, 0)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(3, 2)
        graph, _, __ = self._model_to_graph(
            TransposeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Transpose")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_reduceL2(self):
        class ReduceModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=2, dim=-2, keepdim=False)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            ReduceModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL2")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_reduceL1(self):
        class NormModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=1, dim=-2)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            NormModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL1")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice(self):
        class NarrowModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.narrow(a, 0, 0, 1)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            NarrowModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice_index_exceeds_dim(self):
        class SliceIndexExceedsDimModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[1:10]  # index exceeds dimension
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceIndexExceedsDimModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice_negative_index(self):
        class SliceNegativeIndexModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[0:-1]  # index relative to the end
                c = torch.select(a, dim=-1, index=-2)
                d = torch.select(a, dim=1, index=0)
                return b + x, c + d

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceNegativeIndexModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")

    def test_constant_fold_gather(self):
        class GatherModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.select(a, dim=1, index=-2)
                c = torch.index_select(a, dim=-2, index=torch.tensor([0, 1]))
                return b + 1, c + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        model = GatherModule()
        model(x)
        graph, _, __ = self._model_to_graph(
            GatherModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Gather")

    def test_constant_fold_unsqueeze(self):
        class UnsqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.unsqueeze(a, -2)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 2, 3)
        graph, _, __ = self._model_to_graph(
            UnsqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_unsqueeze_multi_axies(self):
        class PReluModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                a = torch.randn(2, 3, 4, 5, 8, 7)
                return self.prelu(x) + a

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(2, 3, 4, 5, 8, 7)
        graph, _, __ = self._model_to_graph(
            PReluModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3, 4, 5]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 5)

    def test_constant_fold_squeeze_without_axes(self):
        class SqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a) + x + torch.squeeze(a)

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            SqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 4)

    def test_constant_fold_squeeze_with_axes(self):
        class SqueezeAxesModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a, dim=-3) + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            SqueezeAxesModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

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
                a = torch.tensor([[1.0, 2.0, 3.0]]).to(torch.float)
                b = torch.tensor([[4.0, 5.0, 6.0]]).to(torch.float)
                c = torch.cat((a, b), 0)
                d = b + c
                return x + d

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            ConcatModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Concat")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_lstm(self):
        class GruNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            def forward(self, input, initial_state):
                return self.mygru(input, initial_state)

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)
        graph, _, __ = self._model_to_graph(
            GruNet(),
            (input, h0),
            input_names=["input", "h0"],
            dynamic_axes={"input": [0, 1, 2], "h0": [0, 1, 2]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Concat")
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")

        if self.opset_version <= 12:
            self.assertEqual(len(list(graph.nodes())), 3)
        else:
            # Unsqueeze op parameter "axes" as an input instead of as an attribute when opset version >= 13
            self.assertEqual(len(list(graph.nodes())), 4)

    def test_constant_fold_transpose_matmul(self):
        class MatMulNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            def forward(self, A):
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        A = torch.randn(2, 3)
        graph, _, __ = self._model_to_graph(
            MatMulNet(), (A,), input_names=["A"], dynamic_axes={"A": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Transpose")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_reshape(self):
        class ReshapeModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                b = self.weight.reshape(1, -1, 1, 1)
                return x * b

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(4, 5)
        graph, _, __ = self._model_to_graph(
            ReshapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Reshape")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_div(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                div = self.weight.div(torch.tensor([1, 2, 3, 4, 5]))
                return div * x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Div")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_mul(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                mul = self.weight.mul(torch.tensor([1, 2, 3, 4, 5]))
                return mul / x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Mul")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_add(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                add = self.weight + torch.tensor([1, 2, 3, 4, 5])
                return add - x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        for node in graph.nodes():
            self.assertTrue(node.kind() != "onnx::Add")
        self.assertEqual(len(list(graph.nodes())), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(weight, torch.tensor([2, 3, 4, 5, 6]))

    def test_constant_fold_sub(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                sub = self.weight - torch.tensor([1, 2, 3, 4, 5])
                return sub + x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sub")
        self.assertEqual(len(list(graph.nodes())), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(weight, torch.tensor([0, -1, -2, -3, -4]))

    def test_constant_fold_sqrt(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                sqrt = torch.sqrt(self.weight)
                return sqrt / x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sqrt")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_shape(self):
        class ShapeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                shape = self.weight.shape[0]
                return x + shape

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            ShapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Shape")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_upsample_scale_fold_as_constant(self):
        # upsample scale is a constant, not a model parameter,
        # therefore should not be added as initializer after constant folding.
        model = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.randn(1, 32, 224, 224)
        f = io.BytesIO()
        torch.onnx.export(model, x, f)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(len(onnx_model.graph.initializer), 0)

    def test_verbose(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                return torch.exp(input)

        x = torch.randn(3, 4)

        def is_model_stripped(f, verbose=None):
            if verbose is None:
                torch.onnx.export(MyModule(), x, f, opset_version=self.opset_version)
            else:
                torch.onnx.export(
                    MyModule(), x, f, verbose=verbose, opset_version=self.opset_version
                )
            model = onnx.load(io.BytesIO(f.getvalue()))
            model_strip = copy.copy(model)
            onnx.helper.strip_doc_string(model_strip)
            return model == model_strip

        # test verbose=False (default)
        self.assertTrue(is_model_stripped(io.BytesIO()))
        # test verbose=True
        self.assertFalse(is_model_stripped(io.BytesIO(), True))

    # NB: remove this test once DataParallel can be correctly handled
    def test_error_on_data_parallel(self):
        model = torch.nn.DataParallel(torch.nn.ReflectionPad2d((1, 2, 3, 4)))
        x = torch.randn(1, 2, 3, 4)
        f = io.BytesIO()
        with self.assertRaisesRegex(
            ValueError,
            "torch.nn.DataParallel is not supported by ONNX "
            "exporter, please use 'attribute' module to "
            "unwrap model from torch.nn.DataParallel. Try ",
        ):
            torch.onnx.export(model, x, f, opset_version=self.opset_version)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_sequence_dim(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return [x, y]

        model = Module()
        # Export with scripting to keep output as Sequence type.
        # Tracing unpacks the list.
        script_model = torch.jit.script(model)
        x = torch.randn(2, 3)

        # Case 1: dynamic axis
        f = io.BytesIO()
        y = torch.randn(2, 3)
        torch.onnx.export(
            script_model,
            (x, y),
            f,
            opset_version=self.opset_version,
            input_names=["x", "y"],
            dynamic_axes={"y": [1]},
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        loop_output_value_info_proto = onnx_model.graph.output[0]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, None]
        )
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

        # Case 2: no dynamic axes.
        f = io.BytesIO()
        y = torch.randn(2, 3)
        torch.onnx.export(script_model, (x, y), f, opset_version=self.opset_version)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        loop_output_value_info_proto = onnx_model.graph.output[0]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, 3]
        )
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

    def test_export_mode(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                return y

        model = MyModule()
        x = torch.randn(10, 3, 128, 128)
        f = io.BytesIO()

        # set mode to in inference mode and export in training mode
        model.eval()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.TRAINING,
        )
        # verify that the model state is preserved
        self.assertEqual(model.training, old_state)

        # set mode to training mode and export in inference mode
        model.train()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.EVAL,
        )
        # verify that the model state is preserved
        self.assertEqual(model.training, old_state)

    def test_export_does_not_fail_on_frozen_scripted_module(self):
        class Inner(torch.nn.Module):
            def forward(self, x):
                if x > 0:
                    return x
                else:
                    return x * x

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = torch.jit.script(Inner())

            def forward(self, x):
                return self.inner(x)

        x = torch.zeros(1)
        # Freezing is only implemented in eval mode. So we need to call eval()
        outer_module = Outer().eval()
        module = torch.jit.trace_module(outer_module, {"forward": (x)})
        # jit.freeze removes the training attribute in the module
        module = torch.jit.freeze(module)

        torch.onnx.export(module, (x,), io.BytesIO(), opset_version=self.opset_version)

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function(self):
        class N(torch.nn.Module):
            def __init__(self, prob):
                super().__init__()
                self.dropout = torch.nn.Dropout(prob)

            def forward(self, x):
                return self.dropout(x)

        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=i) for i in range(num_layers)]
                )
                self.celu1 = torch.nn.CELU(1.0)
                self.celu2 = torch.nn.CELU(2.0)
                self.dropout = N(0.5)

            def forward(self, x, y, z):
                res1 = self.celu1(x)
                res2 = self.celu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.dropout(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        # Export specified modules. Test against specifying modules that won't
        # exist in the exported model.
        # Model export in inference mode will remove dropout node,
        # thus the dropout module no longer exist in graph.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={
                torch.nn.CELU,
                torch.nn.Dropout,
                torch.nn.LayerNorm,
            },
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))

        # Check function definition
        funcs = onnx_model.functions
        celu_funcs = [f for f in funcs if f.name == "CELU"]
        self.assertEqual(len(celu_funcs), 1)
        self.assertEqual(celu_funcs[0].domain, "torch.nn.modules.activation")
        self.assertEqual(len(celu_funcs[0].attribute), 3)
        ln_funcs = [f for f in funcs if f.name == "LayerNorm"]
        self.assertEqual(len(ln_funcs), 1)
        self.assertEqual(ln_funcs[0].domain, "torch.nn.modules.normalization")
        self.assertEqual(len(ln_funcs[0].attribute), 3)

        # Check local function nodes
        nodes = onnx_model.graph.node
        celu_ns = [n for n in nodes if n.op_type == "CELU"]
        ln_ns = [n for n in nodes if n.op_type == "LayerNorm"]
        self.assertEqual(len(celu_ns), 2)
        self.assertEqual(celu_ns[0].domain, "torch.nn.modules.activation")
        self.assertEqual(len(celu_ns[0].attribute), 3)
        self.assertEqual(len(ln_ns), 3)
        self.assertEqual(ln_ns[0].domain, "torch.nn.modules.normalization")
        self.assertEqual(len(ln_ns[0].attribute), 3)

        # Export specified modules.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={torch.nn.CELU},
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "CELU")

        # Export with empty specified modules. Normal export.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions=set(),
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 0)

        # Export all modules. Should contain {M, CELU, LayerNorm}.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions=True,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 3)

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_overloads(self):
        class NWithOverloads(torch.nn.Module):
            def forward(self, x, y=None, z=None):
                if y is None:
                    return x + 1
                elif z is None:
                    return x + y
                else:
                    return x + y, x + z

        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.n = NWithOverloads()

            def forward(self, x, y, z):
                return self.n(x), self.n(x, y), self.n(x, y, z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={NWithOverloads},
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 3)
        func_names = [f.name for f in funcs]
        self.assertIn("NWithOverloads", func_names)
        self.assertIn("NWithOverloads.1", func_names)
        self.assertIn("NWithOverloads.2", func_names)

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_infer_scopes(self):
        class M(torch.nn.Module):
            def forward(self, x):
                # Concatenation of scalars inserts unscoped tensors in IR graph.
                new_tensor_shape = x.size()[:-1] + (1, 1, -1)
                tensor = x.view(*new_tensor_shape)
                return tensor

        x = torch.randn(4, 5)
        f = io.BytesIO()
        torch.onnx.export(
            M(),
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            do_constant_folding=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertIn("M", [f.name for f in funcs])

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_predefined_attributes(self):
        class M(torch.nn.Module):
            num_layers: int

            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
                )

            def forward(self, x):
                for ln in self.lns:
                    x = ln(x)
                return x

        x = torch.randn(2, 3)
        f = io.BytesIO()
        model = M(3)
        torch.onnx.export(
            model,
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        m_funcs = [fn for fn in funcs if fn.name == "M"]
        self.assertEqual(m_funcs[0].attribute, ["num_layers"])
        ln_funcs = [fn for fn in funcs if fn.name == "LayerNorm"]
        self.assertEqual(ln_funcs[0].attribute, ["eps", "elementwise_affine"])

        from onnx import helper

        m_node = [n for n in onnx_model.graph.node if n.op_type == "M"]
        self.assertEqual(
            m_node[0].attribute[0],
            helper.make_attribute("num_layers", model.num_layers),
        )

        ln_nodes = [n for n in m_funcs[0].node if n.op_type == "LayerNorm"]
        expected_ln_attrs = [
            helper.make_attribute(
                "elementwise_affine", model.lns[0].elementwise_affine
            ),
            helper.make_attribute("eps", model.lns[0].eps),
        ]
        for ln_node in ln_nodes:
            self.assertIn(ln_node.attribute[0], expected_ln_attrs)
            self.assertIn(ln_node.attribute[1], expected_ln_attrs)

    def test_node_scope(self):
        class N(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=float(i)) for i in range(num_layers)]
                )
                self.gelu1 = torch.nn.GELU()
                self.gelu2 = torch.nn.GELU()
                self.relu = N()

            def forward(self, x, y, z):
                res1 = self.gelu1(x)
                res2 = self.gelu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.relu(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        model = M(3)
        expected_scope_names = {
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.M::/"
            "torch.nn.modules.activation.GELU::gelu1",
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.M::/"
            "torch.nn.modules.activation.GELU::gelu2",
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.M::/"
            "torch.nn.modules.normalization.LayerNorm::lns.0",
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.M::/"
            "torch.nn.modules.normalization.LayerNorm::lns.1",
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.M::/"
            "torch.nn.modules.normalization.LayerNorm::lns.2",
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.M::/"
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.N::relu/"
            "torch.nn.modules.activation.ReLU::relu",
            "test_utility_funs.TestUtilityFuns.test_node_scope.<locals>.M::",
        }

        graph, _, _ = self._model_to_graph(
            model, (x, y, z), input_names=[], dynamic_axes={}
        )
        for node in graph.nodes():
            self.assertIn(node.scopeName(), expected_scope_names)

        expected_torch_script_scope_names = {
            "test_utility_funs.M::/torch.nn.modules.activation.GELU::gelu1",
            "test_utility_funs.M::/torch.nn.modules.activation.GELU::gelu2",
            "test_utility_funs.M::/torch.nn.modules.normalization.LayerNorm::lns.0",
            "test_utility_funs.M::/torch.nn.modules.normalization.LayerNorm::lns.1",
            "test_utility_funs.M::/torch.nn.modules.normalization.LayerNorm::lns.2",
            "test_utility_funs.M::/test_utility_funs.N::relu/torch.nn.modules.activation.ReLU::relu",
            "test_utility_funs.M::",
        }

        graph, _, _ = self._model_to_graph(
            torch.jit.script(model), (x, y, z), input_names=[], dynamic_axes={}
        )
        for node in graph.nodes():
            self.assertIn(node.scopeName(), expected_torch_script_scope_names)

    def test_scope_of_constants_when_combined_by_cse_pass(self):
        layer_num = 3

        class M(torch.nn.Module):
            def __init__(self, constant):
                super().__init__()
                self.constant = constant

            def forward(self, x):
                # 'self.constant' is designed to be the same for all layers,
                # hence it is common sub expression.
                return x + self.constant

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [M(constant=torch.tensor(1.0)) for i in range(layers)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )

        # NOTE: Duplicated constants are populated due to implicit casting in scalar_type_analysis,
        #       so we expect 3 constants with different scopes. The 3 constants are for the 3 layers.
        #       If CSE in exporter is improved later, this test needs to be updated.
        #       It should expect 1 constant, with same scope as root.
        scope_prefix = "test_utility_funs.TestUtilityFuns.test_scope_of_constants_when_combined_by_cse_pass.<locals>"
        expected_root_scope_name = f"{scope_prefix}.N::"
        expected_layer_scope_name = f"{scope_prefix}.M::layers"
        expected_constant_scope_name = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        constant_scope_names = []
        for node in graph.nodes():
            if node.kind() == "onnx::Constant":
                constant_scope_names.append(node.scopeName())
        self.assertEqual(constant_scope_names, expected_constant_scope_name)

    def test_scope_of_nodes_when_combined_by_cse_pass(self):
        layer_num = 3

        class M(torch.nn.Module):
            def __init__(self, constant, bias):
                super().__init__()
                self.constant = constant
                self.bias = bias

            def forward(self, x):
                # 'constant' and 'x' is designed to be the same for all layers,
                # hence `x + self.constant` is common sub expression.
                # 'bias' is designed to be different for all layers,
                # hence `* self.bias` is not common sub expression.
                return (x + self.constant) * self.bias

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()

                self.layers = torch.nn.ModuleList(
                    [
                        M(constant=torch.tensor([1.0]), bias=torch.randn(1))
                        for i in range(layers)
                    ]
                )

            def forward(self, x):
                y = []
                for layer in self.layers:
                    y.append(layer(x))
                return y[0], y[1], y[2]

        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )
        scope_prefix = "test_utility_funs.TestUtilityFuns.test_scope_of_nodes_when_combined_by_cse_pass.<locals>"
        expected_root_scope_name = f"{scope_prefix}.N::"
        expected_layer_scope_name = f"{scope_prefix}.M::layers"
        expected_add_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.0"
        ]
        expected_mul_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        add_scope_names = []
        mul_scope_names = []
        for node in graph.nodes():
            if node.kind() == "onnx::Add":
                add_scope_names.append(node.scopeName())
            elif node.kind() == "onnx::Mul":
                mul_scope_names.append(node.scopeName())
        self.assertEqual(add_scope_names, expected_add_scope_names)
        self.assertEqual(mul_scope_names, expected_mul_scope_names)

    def test_aten_fallthrough(self):
        # Test aten export of op with no symbolic
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.erfc(x)

        x = torch.randn(2, 3, 4)
        GLOBALS.export_onnx_opset_version = self.opset_version
        graph, _, __ = self._model_to_graph(
            Module(),
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::erfc")

    def test_custom_op_fallthrough(self):
        # Test custom op
        op_source = """
        #include <torch/script.h>

        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
          return self + other;
        }

        static auto registry =
          torch::RegisterOperators("custom_namespace::custom_op", &custom_add);
        """

        torch.utils.cpp_extension.load_inline(
            name="custom_add",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )

        class FooModel(torch.nn.Module):
            def forward(self, input, other):
                # Calling custom op
                return torch.ops.custom_namespace.custom_op(input, other)

        x = torch.randn(2, 3, 4, requires_grad=False)
        y = torch.randn(2, 3, 4, requires_grad=False)
        model = FooModel()
        graph, _, __ = self._model_to_graph(
            model,
            (x, y),
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "custom_namespace::custom_op")

    def test_custom_opsets_gelu(self):
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::gelu", 9)

        def gelu(g, self, approximate):
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("::gelu", gelu, 9)
        model = torch.nn.GELU(approximate="none")
        x = torch.randn(3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
        )

        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        self.assertEqual(graph.opset_import[0].version, self.opset_version)
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")
        self.assertEqual(graph.opset_import[1].version, 1)

    def test_register_aten_custom_op_symbolic(self):
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "aten::gelu", 9)

        def gelu(g, self, approximate):
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("aten::gelu", gelu, 9)
        model = torch.nn.GELU(approximate="none")
        x = torch.randn(3, 3)
        f = io.BytesIO()
        torch.onnx.export(model, (x,), f, opset_version=self.opset_version)
        graph = onnx.load(io.BytesIO(f.getvalue()))

        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")

    @skipIfNoLapack
    def test_custom_opsets_inverse(self):
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

        class CustomInverse(torch.nn.Module):
            def forward(self, x):
                return torch.inverse(x) + x

        def linalg_inv(g, self):
            return g.op("com.microsoft::Inverse", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
        )

        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.node[0].op_type, "Inverse")
        self.assertEqual(graph.opset_import[0].version, self.opset_version)
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")
        self.assertEqual(graph.opset_import[1].version, 1)

    def test_onnx_fallthrough(self):
        # Test aten export of op with symbolic for aten
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.digamma(x)

        x = torch.randn(100, 128)
        graph, _, __ = self._model_to_graph(
            Module(),
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::digamma")

    # prim::ListConstruct is exported as onnx::SequenceConstruct for opset >= 11
    @skipIfUnsupportedMaxOpsetVersion(10)
    def test_prim_fallthrough(self):
        # Test prim op
        class PrimModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if isinstance(x, list):
                    y = x
                else:
                    y = [x]
                return y

        x = torch.tensor([2])
        model = PrimModule()
        model.eval()
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::ListConstruct")

    def test_custom_layer_tuple(self):
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def symbolic(g, input):
                return g.op("CustomNamespace::Custom", input, outputs=2)

            @staticmethod
            def forward(ctx, input):
                return input, input

        class Custom(torch.nn.Module):
            def forward(self, input):
                return CustomFunction.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        graph, _, _ = self._model_to_graph(
            model, batch, input_names=["batch"], dynamic_axes={"batch": [0, 1]}
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "CustomNamespace::Custom")

    def test_autograd_onnx_fallthrough(self):
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                (input,) = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input

        class Custom(torch.nn.Module):
            def forward(self, input):
                return CustomFunction.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        graph, _, _ = self._model_to_graph(
            model,
            batch,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["batch"],
            dynamic_axes={"batch": [0, 1]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::PythonOp")

    def test_autograd_module_name(self):
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                (input,) = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input

        class Custom(torch.nn.Module):
            def forward(self, input):
                return CustomFunction.apply(input) + CustomFunction2.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        graph, _, _ = self._model_to_graph(
            model,
            batch,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["batch"],
            dynamic_axes={"batch": [0, 1]},
        )
        iter = graph.nodes()
        autograd1 = next(iter)
        autograd2 = next(iter)
        self.assertEqual(autograd1.kind(), "prim::PythonOp")
        self.assertEqual(autograd2.kind(), "prim::PythonOp")
        self.assertNotEqual(autograd1.s("module"), autograd2.s("module"))

    def test_unused_initializers(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(1, 1)
                )
                self.k_proj = torch.nn.Linear(5, 5, bias=True)

            def forward(self, x):
                x = self.conv2(x)
                return x

        x = torch.randn(20, 16, 50, 100)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        _, params_dict, __ = self._model_to_graph(
            Model(),
            (x,),
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        self.assertEqual(len(params_dict), 2)

    def test_scripting_param(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = torch.nn.BatchNorm2d(16, affine=True)

            def forward(self, x):
                x = self.conv(x)
                bn = self.bn(x)
                return bn

        model = torch.jit.script(MyModule())
        x = torch.randn(10, 3, 128, 128)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            training=torch.onnx.TrainingMode.TRAINING,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        graph_input_params = [param.debugName() for param in graph.inputs()]
        for item in dict(model.named_parameters()):
            self.assertIn(
                item,
                graph_input_params,
                "Graph parameter names does not match model parameters.",
            )

    @skipIfNoCaffe2
    def test_modifying_params(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.tensor([2.0]))

            def forward(self, x):
                y = x * x
                self.param.data.add_(1.0)
                return y

        x = torch.tensor([1, 2])
        # Move import to local as caffe2 backend requires additional build flag,
        # and is only used in this test case.
        import caffe2.python.onnx.backend as backend

        verify(MyModel(), x, backend, do_constant_folding=False)

    def test_fuse_conv_bn(self):
        class Fuse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        graph, _, __ = self._model_to_graph(
            Fuse(),
            (x,),
            training=TrainingMode.EVAL,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::BatchNormalization")
            self.assertEqual(node.kind(), "onnx::Conv")

        self.assertEqual(len(list(graph.nodes())), 1)

    def test_fuse_resnet18(self):
        model = torchvision.models.resnet18(pretrained=False)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            training=TrainingMode.EVAL,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::BatchNormalization")

    def test_onnx_function_substitution_pass(self):
        @torch.jit.script
        def f(x: torch.Tensor, y: torch.Tensor):
            z = x - y
            return x + z

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return f(x, y)

        input_1 = torch.tensor([11])
        input_2 = torch.tensor([12])
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            MyModule(),
            (input_1, input_2),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": [0], "input_2": [0]},
        )
        # Check that the prim::Constant node in the graph for representing the
        # scripted function `f` is removed and the following prim::CallFunction
        # is replced by inline graph, with onnx::Sub and onnx::Add nodes.
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "prim::Constant")
        self.assertEqual(
            len(list(graph.nodes())), 2
        )  # onnx::Sub and onnx::Add nodes only.

    def test_onnx_value_name(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_weight = torch.nn.Parameter(torch.Tensor(3, 3))
                self.in_bias = torch.nn.Parameter(torch.Tensor(3))

            def forward(self, x):
                start = 0
                end = None
                weight = self.in_weight
                bias = self.in_bias
                weight = weight[start:end, :]
                if bias is not None:
                    bias = bias[start:end]
                return torch.nn.functional.linear(x, weight, bias)

        model = MyModule()
        x = torch.randn(3, 3)
        f = io.BytesIO()

        model.eval()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            keep_initializers_as_inputs=True,
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.input[1].name, "in_weight")
        self.assertEqual(graph.graph.input[2].name, "in_bias")

    def test_onnx_node_naming(self):
        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._module_1 = torch.nn.Linear(10, 10)
                self._module_2 = torch.nn.Linear(10, 10)
                self._module_3 = torch.nn.Linear(10, 10)
                self._module_4 = torch.nn.Linear(10, 10)

            def forward(self, x):
                y = self._module_1(x)
                z = self._module_2(y)
                z = self._module_3(y * z)
                z = self._module_4(y * z)
                return z

        module = MainModule()
        ref_node_names = [
            "/_module_1/Gemm",
            "/_module_2/Gemm",
            "/_module_3/Gemm",
            "/_module_4/Gemm",
            "/Mul",
            "/Mul_1",
        ]
        f = io.BytesIO()

        torch.onnx.export(module, torch.ones(1, 10), f, output_names=["y"])
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        for n in onnx_model.graph.node:
            self.assertIn(n.name, ref_node_names)

        torch.onnx.export(
            torch.jit.script(module), torch.ones(1, 10), f, output_names=["y"]
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        for n in onnx_model.graph.node:
            self.assertIn(n.name, ref_node_names)

    def _test_deduplicate_initializers(self, torchscript=False):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(3, 3)
                self.layer2 = torch.nn.Linear(3, 3)

                # Reusing layers.
                self.layer3 = self.layer1

                # Reusing parameters.
                self.layer2.weight = self.layer1.weight
                self.layer1.bias = self.layer2.bias

                # Parameter with different tensors equal in value.
                self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
                self.param2 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

            def forward(self, x):
                return (
                    self.layer3(self.layer2(self.layer1(x))) + self.param1 + self.param2
                )

        model = torch.jit.script(MyModule()) if torchscript else MyModule()

        x = torch.randn(3, 3)
        param_name_set = {k for k, _ in model.named_parameters()}

        # Test training mode.
        model.train()
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            training=TrainingMode.TRAINING,
            opset_version=self.opset_version,
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

        model.train()
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            training=TrainingMode.PRESERVE,
            opset_version=self.opset_version,
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

        # Test eval mode.
        model.eval()
        f = io.BytesIO()
        torch.onnx.export(model, (x,), f, opset_version=self.opset_version)
        graph = onnx.load(io.BytesIO(f.getvalue()))
        param_name_set.remove("param2")
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

    def test_deduplicate_initializers(self):
        self._test_deduplicate_initializers(torchscript=False)

    def test_deduplicate_initializers_torchscript(self):
        self._test_deduplicate_initializers(torchscript=True)

    @skipIfNoCuda
    def test_deduplicate_initializers_diff_devices(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w_cpu = torch.nn.Parameter(
                    torch.ones(3, device=torch.device("cpu"))
                )
                self.w_cuda = torch.nn.Parameter(
                    torch.ones(3, device=torch.device("cuda"))
                )

            def forward(self, x, y):
                return x + self.w_cpu, y + self.w_cuda

        x = torch.randn(3, 3, device=torch.device("cpu"))
        y = torch.randn(3, 3, device=torch.device("cuda"))
        f = io.BytesIO()
        torch.onnx.export(Model(), (x, y), f, opset_version=self.opset_version)
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, {"w_cpu"})

    def test_duplicated_output_node(self):
        class DuplicatedOutputNet(torch.nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, num_classes)

            def forward(self, input0, input1):
                out1 = self.fc1(input0)
                out2 = self.fc1(input1)
                return out1, out1, out2, out1, out2

        N, D_in, H, D_out = 64, 784, 500, 10
        pt_model = DuplicatedOutputNet(D_in, D_out)

        f = io.BytesIO()
        x = torch.randn(N, D_in)
        dynamic_axes = {
            "input0": {0: "input0_dim0", 1: "input0_dim1"},
            "input1": {0: "input1_dim0", 1: "input1_dim1"},
            "output-0": {0: "output-0_dim0", 1: "output-0_dim1"},
            "output-1": {0: "output-1_dim0", 1: "output-1_dim1"},
            "output-2": {0: "output-2_dim0", 1: "output-2_dim1"},
            "output-3": {0: "output-3_dim0", 1: "output-3_dim1"},
            "output-4": {0: "output-4_dim0", 1: "output-4_dim1"},
        }

        torch.onnx.export(
            pt_model,
            (x, x),
            f,
            input_names=["input0", "input1"],
            output_names=["output-0", "output-1", "output-2", "output-3", "output-4"],
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamic_axes=dynamic_axes,
            verbose=True,
            keep_initializers_as_inputs=True,
        )

        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.input[0].name, "input0")
        self.assertEqual(graph.graph.input[1].name, "input1")
        for i in range(5):
            self.assertEqual(graph.graph.output[i].name, f"output-{i}")
        self.assertEqual(graph.graph.node[0].op_type, "Gemm")
        self.assertEqual(graph.graph.node[1].op_type, "Identity")
        self.assertEqual(graph.graph.node[2].op_type, "Identity")
        self.assertEqual(graph.graph.node[3].op_type, "Gemm")
        self.assertEqual(graph.graph.node[4].op_type, "Identity")

    def test_deduplicate_ignore_upsample_scale(self):
        # upsample scale is a constant, not a model parameter,
        # therefore should be ignored by shared weight deduplication.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample_1 = torch.nn.Upsample(scale_factor=2)
                self.upsample_2 = torch.nn.Upsample(scale_factor=2)

            def forward(self, x):
                return self.upsample_1(x), self.upsample_2(x)

        f = io.BytesIO()
        x = torch.randn(1, 32, 224, 224)
        torch.onnx.export(Model(), x, f)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # aten::upsample converts to onnx::resize
        resize_nodes = [n for n in onnx_model.graph.node if n.op_type == "Resize"]
        self.assertEqual(len(resize_nodes), 2)
        for resize_node in resize_nodes:
            scale_node = [
                n for n in onnx_model.graph.node if n.output[0] == resize_node.input[2]
            ]
            self.assertEqual(len(scale_node), 1)
            self.assertEqual(scale_node[0].op_type, "Constant")

    def test_bad_symbolic_registration(self):
        _onnx_opset_version = 9

        @parse_args("v")
        def cat(g, tensor_list, dim):
            tensors = _unpack_list(tensor_list)
            return g.op("Concat", *tensors, axis_i=dim)

        torch.onnx.register_custom_op_symbolic("::cat", cat, _onnx_opset_version)

        class CatModel(torch.nn.Module):
            def forward(self, x):
                return torch.cat((x, x, x), 0)

        model = CatModel()
        x = torch.randn(2, 3)
        f = io.BytesIO()
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: torch.onnx.export(
                model, (x,), f, opset_version=_onnx_opset_version
            ),
            (
                "A mismatch between the number of arguments (2) and their descriptors (1) was found at symbolic function "
                "'cat'. If you believe this is not due to custom symbolic implementation within your code or an external "
                "library, please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to "
                "report this bug."
            ),
        )
        torch.onnx.unregister_custom_op_symbolic("::cat", _onnx_opset_version)


if __name__ == "__main__":
    common_utils.run_tests()
