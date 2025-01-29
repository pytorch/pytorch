# Owner(s): ["module: nestedtensor"]

import torch
from torch.nested._internal.dict_tensor import (
    DictTensor,
    register_dict_tensor_func,
    set_func_registry,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDictTensor(TestCase):
    def test_basic_eager(self):
        # Create some tensors
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([4, 5, 6], dtype=torch.float32)
        c = torch.tensor([7, 8, 9], dtype=torch.float32)
        metadata = {"a": a, "b": b, "c": None}
        dict_tensor = DictTensor(metadata)
        # Test that dict_tensor is created correctly
        self.assertIsInstance(dict_tensor, DictTensor)
        self.assertEqual(dict_tensor.shape, (0,))
        # Accessing a field that is listed in all_fields but not present in metadata returns
        # None instead of raising AttributeError.
        self.assertIsNone(dict_tensor.c, c)

        # Test that accessing a non-existent field raises AttributeError
        with self.assertRaises(AttributeError):
            _ = dict_tensor.d

    def test_open_registration(self):
        tmp_registry = {}

        with set_func_registry(tmp_registry):
            # Create some tensors
            a = torch.tensor([1, 2, 3], dtype=torch.float32)
            b = torch.tensor([4, 5, 6], dtype=torch.float32)
            c = torch.tensor([7, 8, 9], dtype=torch.float32)
            metadata = {"a": a, "b": b, "c": c}
            dict_tensor = DictTensor(metadata)

            # Before registration, clone errors
            with self.assertRaisesRegex(
                NotImplementedError,
                "DictTensor does not support for aten.clone.default",
            ):
                dict_tensor.clone()

            # Define a custom clone function that rewraps the output into
            # a new DictTensor.
            @register_dict_tensor_func(torch.ops.aten.clone.default)
            def dict_tensor_clone(op, inp, *args, **kwargs):
                cloned_metadata = {}
                for k, v in inp.metadata.items():
                    cloned_metadata[k] = v.clone()
                return DictTensor(cloned_metadata)

            cloned_dict_tensor = dict_tensor.clone()
            self.assertIsInstance(cloned_dict_tensor, DictTensor)

            for key in dict_tensor.metadata.keys():
                assert isinstance(cloned_dict_tensor, DictTensor)
                self.assertEqual(
                    cloned_dict_tensor.metadata[key], dict_tensor.metadata[key]
                )
                self.assertFalse(
                    cloned_dict_tensor.metadata[key] is dict_tensor.metadata[key]
                )

        # After leaving the context, clone behaves as it did before.
        with self.assertRaisesRegex(
            NotImplementedError, "DictTensor does not support for aten.clone.default"
        ):
            dict_tensor.clone()

    def test_basic_compile(self):
        tmp_registry = {}

        with set_func_registry(tmp_registry):

            @register_dict_tensor_func(torch.ops.aten.clone.default)
            def dict_tensor_clone(op, inp, *args, **kwargs):
                # Unwraps to the 'a' attr and clones
                out = inp.metadata["a"].clone()
                return out

            #
            # Construct DictTensor outside the graph
            #
            a = torch.tensor([1, 2, 3], dtype=torch.float32)
            b = torch.tensor([4, 5, 6], dtype=torch.float32)
            c = torch.tensor([7, 8, 9], dtype=torch.float32)
            metadata = {"a": a, "b": b, "c": c}

            dict_tensor = DictTensor(metadata)

            @torch.compile(fullgraph=True)
            def fn1(x):
                return x.clone().clone()

            out = fn1(dict_tensor)
            self.assertFalse(isinstance(out, DictTensor))

            #
            # Construct DictTensor inside the graph
            #
            a = torch.tensor([1, 2, 3], dtype=torch.float32)
            b = torch.tensor([4, 5, 6], dtype=torch.float32)
            c = torch.tensor([7, 8, 9], dtype=torch.float32)
            metadata = {"a": a, "b": b, "c": c}

            @torch.compile(fullgraph=True)
            def fn2(y):
                x = DictTensor(metadata)
                return x.clone() * y

            out = fn2(a.clone())
            self.assertFalse(isinstance(out, DictTensor))


if __name__ == "__main__":
    run_tests()
