# Owner(s): ["module: dynamo"]
import copy
import pickle
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export
from torch._export.serialize import (
    convert_fake_tensor_to_tensor_meta,
    convert_tensor_meta_to_fake_tensor,
    serialize,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.testing._internal.common_utils import run_tests, TestCase
from functorch.experimental import control_flow


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    def test_pickle(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            def true_fn(x):
                def inner_true_fn(y):
                    return x + y

                return inner_true_fn(x)

            def false_fn(x):
                def inner_false_fn(y):
                    return x - y

                return inner_false_fn(x)

            return control_flow.cond(x.shape[0] < 10, true_fn, false_fn, [x])

        inputs = (torch.ones(3),)
        ep = export(f, inputs)

        # Pickle the ExportGraphModule
        pickled_ep = pickle.dumps(convert_fake_tensor_to_tensor_meta(copy.deepcopy(ep))[0])
        loaded_ep = convert_tensor_meta_to_fake_tensor(pickle.loads(pickled_ep))

        for node1, node2 in zip(loaded_ep.graph.nodes, ep.graph.nodes):
            val1 = node1.meta.get("val", None)
            val2 = node2.meta.get("val", None)

            if val1 is None or val2 is None:
                # Either both are None
                self.assertEqual(val1, val2)
            elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                # Or both are fake tensors with the same shape/dtype
                self.assertEqual(val1.shape, val2.shape)
                self.assertEqual(val1.dtype, val2.dtype)
            elif isinstance(val1, list) and isinstance(val2, list):
                # Or both are fake tensors lists with one element and with the
                # same shape/dtype
                self.assertTrue(len(val1) == len(val2) and len(val1) == 1)
                self.assertEqual(val1[0].shape, val2[0].shape)
                self.assertEqual(val1[0].dtype, val2[0].dtype)
            else:
                # For expressions like 's0 < 10' can only compare through string
                self.assertEqual(str(val1), str(val2))

        self.assertTrue(torch.allclose(loaded_ep(*inputs), ep(*inputs)))

        # Check metadata
        self.assertEqual(ep.call_spec.in_spec, loaded_ep.call_spec.in_spec)
        self.assertEqual(ep.call_spec.out_spec, loaded_ep.call_spec.out_spec)

    def test_serialize_multiple_returns_from_node(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w, b):
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        exported_module = export(
            MyModule(),
            (
                torch.ones([512, 512], requires_grad=True),
                torch.ones([512]),
                torch.ones([512]),
            ),
        )

        serialized, _ = serialize(exported_module)
        node = serialized.graph.nodes[0]
        self.assertEqual(node.target.name, "aten.var_mean.correction")
        # aten::native_layer_norm returns 3 tensnors
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_serialize_list_returns(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, 2)

        input = torch.arange(10.0).reshape(5, 2)
        input.requires_grad = True
        exported_module = export(MyModule(), (input,))

        serialized, _ = serialize(exported_module)
        node = serialized.graph.nodes[0]
        self.assertEqual(node.target.name, "aten.split.Tensor")
        self.assertEqual(len(node.outputs), 1)
        # Input looks like:
        # tensor([[0, 1],
        #         [2, 3],
        #         [4, 5],
        #         [6, 7],
        #         [8, 9]])
        # Output looks like:
        # (tensor([[0, 1],
        #          [2, 3]]),
        #  tensor([[4, 5],
        #          [6, 7]]),
        #  tensor([[8, 9]]))
        self.assertEqual(len(node.outputs[0].as_tensors), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs[0].as_tensors:
            name = output.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_multi_return_some_unused(self) -> None:
        """
        Make sure the serialized output matches the op schema, even if some of
        the arguments are never used in the graph.
        """

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.var_mean.correction(x, [1])[0]

        exported_module = export(
            MyModule(),
            (torch.ones([512, 512], requires_grad=True),),
        )

        serialized, _ = serialize(exported_module)
        node = serialized.graph.nodes[0]
        self.assertEqual(node.target.name, "aten.var_mean.correction")
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_kwargs_default(self) -> None:
        """
        Tests that the kwargs default values are serialized even if they are not
        specified
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            values = torch.randn(3, 2)
            return torch.searchsorted(x, values, side="right", right=True)

        x, _ = torch.sort(torch.randn(3, 4))
        exported_module = export(f, (x,))
        serialized, _ = serialize(exported_module)

        node = serialized.graph.nodes[1]
        self.assertEqual(node.target.name, "aten.searchsorted.Tensor")
        self.assertEqual(len(node.inputs), 6)
        self.assertEqual(node.inputs[2].arg.as_bool, False)
        self.assertEqual(node.inputs[3].arg.as_bool, True)
        self.assertEqual(node.inputs[4].arg.as_string, "right")
        self.assertEqual(node.inputs[5].arg.as_none, ())


if __name__ == '__main__':
    run_tests()
