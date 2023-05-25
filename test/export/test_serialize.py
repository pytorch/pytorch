# Owner(s): ["module: dynamo"]
import copy
import pickle
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export
from torch._export.serialize import convert_fake_tensor_to_tensor_meta, convert_tensor_meta_to_fake_tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch.testing._internal.common_utils import run_tests, TestCase
from functorch.experimental import control_flow


class TestSerialize(TestCase):
    @unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
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


if __name__ == '__main__':
    run_tests()
