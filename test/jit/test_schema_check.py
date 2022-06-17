# Owner(s): ["oncall: jit"]

import os
import sys
import torch
from torch.utils._pytree import tree_map


from torch.testing._internal.schema_check_mode import SchemaCheckMode
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch.testing._internal.jit_utils import JitTestCase

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# This TorchDispatchTensor Subclass is used to simulate an incorrect schema
# which is then used to test that SchemaCheckMode behaves as expected

class IncorrectAliasTensor(torch.Tensor):
    INCORRECT_OPS = {"aten::add", "aten::sub"}

    elem: torch.Tensor

    __slots__ = ['elem']

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (IncorrectAliasTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=kwargs.get("requires_grad", False)
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        return super().__repr__(tensor_contents=f"{self.elem}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e
        unwrapped_args = tree_map(unwrap, args)
        out = func(*unwrapped_args, **tree_map(unwrap, kwargs))
        if func._schema.name in IncorrectAliasTensor.INCORRECT_OPS:
            args[0].elem = out

        return tree_map(wrap, out)

# Tests various schema checking functionalities.
class TestSchemaCheck(JitTestCase):
    # Tests that SchemaCheckMode records operator order with grad
    def test_schema_check_mode_operator_order(self):
        schema_check = SchemaCheckMode()
        with enable_torch_dispatch_mode(schema_check):
            x = torch.rand((3, 3), requires_grad=True)
            x.relu().sin()
            self.assertEqual(["aten::rand", "aten::relu", "aten::sin"], schema_check.ops)

    # Tests that SchemaCheckMode records operator order without grad
    def test_schema_check_tensor_operator_order_without_grad(self):
        schema_check = SchemaCheckMode()
        with enable_torch_dispatch_mode(schema_check):
            x = torch.rand((3, 3), requires_grad=False)
            x.relu().sin()
            self.assertEqual(["aten::rand", "aten::relu", "aten::sin"], schema_check.ops)

    # Tests that SchemaCheckMode wraps torch.Tensor
    def test_schema_check_tensor_functionality(self):
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.relu().sin()
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = x.relu().sin()
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when an argument's default is overriden
    def test_schema_check_tensor_functionality_default_replaced(self):
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.add(x, alpha=2)
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = x.add(x, alpha=2)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when there is a Tensor[] argument
    def test_schema_check_tensor_functionality_list_input(self):
        a = torch.rand((3, 3))
        b = torch.rand((3, 3))
        c = torch.rand((3, 3))
        expected = torch.linalg.multi_dot([a, b, c])
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = torch.linalg.multi_dot([a, b, c])
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when there is a kwarg tensor input
    def test_schema_check_tensor_functionality_kwarg_tensor(self):
        x = torch.rand((3, 5))
        w = torch.rand((4))
        expected = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor with a mutable op
    def test_schema_check_tensor_functionality_mutable_inputs(self):
        expected = torch.rand((3, 3), requires_grad=False)
        actual = torch.clone(expected)
        expected.sinh_()
        with enable_torch_dispatch_mode(SchemaCheckMode()):
            actual.sinh_()
        self.assertEqual(expected, actual)

    # Tests that an exception is raised for a mismatching mutation
    def test_mutation_check_fail(self):
        with self.assertRaisesRegex(RuntimeError, "Argument running_mean is not defined as mutable but was mutated"):
            x = torch.rand((3, 3), requires_grad=True)
            batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
            with enable_torch_dispatch_mode(SchemaCheckMode()):
                batch(x)

    # Tests that an exception is raised for a mismatching mutation over multiple ops
    def test_mutation_check_fail_multiple_operators(self):
        with self.assertRaisesRegex(RuntimeError, "Argument running_mean is not defined as mutable but was mutated"):
            x = torch.rand((3, 3), requires_grad=True)
            batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
            with enable_torch_dispatch_mode(SchemaCheckMode()):
                x = x.sinh()
                x = x.tanh()
                x = x.relu()
                batch(x)

    # Tests that an exception is raised for a mismatching alias
    def test_alias_check_fail(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3))
            with enable_torch_dispatch_mode(SchemaCheckMode()):
                IncorrectAliasTensor(x).add(IncorrectAliasTensor(y), alpha=2)

    # Tests that an exception is raised for a mismatching alias over multiple ops
    def test_alias_check_fail_multiple_operators(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            with enable_torch_dispatch_mode(SchemaCheckMode()):
                IncorrectAliasTensor(x).sin().relu().add(IncorrectAliasTensor(y), alpha=2)

    # Tests that an exception is raised for a centered mismatching alias over multiple ops
    def test_alias_check_fail_multiple_operators_centered(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            with enable_torch_dispatch_mode(SchemaCheckMode()):
                IncorrectAliasTensor(x).sin().add(IncorrectAliasTensor(y), alpha=2).relu()

    # Tests that isAliasOf returns as expected
    def test_is_alias_of(self):
        x = torch.rand((3, 3), requires_grad=True)
        y = torch.rand((3, 3), requires_grad=True)
        y = x.add(x, alpha=2)
        self.assertTrue(torch._C._is_alias_of(x, x))
        self.assertFalse(torch._C._is_alias_of(x, y))
