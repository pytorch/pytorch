# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import os
import sys
import torch
from torch.utils._pytree import tree_map
import unittest

from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO
from torch.fx.operator_schemas import normalize_function
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_device_type import ops, OpDTypes, instantiate_device_type_tests
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

def secretly_aliasing(x):
    return x.view(-1)

def secretly_mutating(x):
    x.mul_(2)
    return x * 3

def output_is_input(x):
    return x

custom_lib = torch.library.Library("bad_schemas", "DEF")  # noqa: TOR901
custom_lib.define("secretly_aliasing(Tensor x) -> Tensor")
custom_lib.define("secretly_mutating(Tensor x) -> Tensor")
custom_lib.define("output_is_input(Tensor(a) x) -> Tensor(a)")

custom_lib_cpu = torch.library.Library("bad_schemas", "IMPL", "CPU")  # noqa: TOR901
custom_lib_cpu.impl("secretly_aliasing", secretly_aliasing)
custom_lib_cpu.impl("secretly_mutating", secretly_mutating)
custom_lib_cpu.impl("output_is_input", output_is_input)

custom_lib_meta = torch.library.Library("bad_schemas", "IMPL", "Meta")  # noqa: TOR901
custom_lib_meta.impl("secretly_aliasing", secretly_aliasing)
custom_lib_meta.impl("secretly_mutating", secretly_mutating)
custom_lib_meta.impl("output_is_input", output_is_input)

# This TorchDispatchTensor Subclass is used to simulate an incorrect schema
# which is then used to test that SchemaCheckMode behaves as expected

class IncorrectAliasTensor(torch.Tensor):
    ALIAS_ARG_OUT = {"aten::add"}
    ALIAS_OUT_OUT = {"aten::aminmax"}
    MUTATE_ARGS_OUT = {"aten::sub"}

    elem: torch.Tensor

    __slots__ = ['elem']

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
        if func._schema.name in IncorrectAliasTensor.ALIAS_ARG_OUT:
            args[0].elem = out
        if func._schema.name in IncorrectAliasTensor.MUTATE_ARGS_OUT:
            args[0].elem = torch.rand(args[0].elem.shape)
        if func._schema.name in IncorrectAliasTensor.ALIAS_OUT_OUT:
            incorrect_out = list(out)
            incorrect_out[0] = incorrect_out[1]
            return tree_map(wrap, tuple(incorrect_out))

        return tree_map(wrap, out)

# Tests various schema checking functionalities.
class TestSchemaCheck(JitTestCase):
    def setUp(self):
        if TEST_WITH_TORCHDYNAMO:
            self.skipTest("SchemaCheckMode is ignored by dynamo")
        super().setUp()

    # Tests that SchemaCheckMode records operator order with grad
    def test_schema_check_mode_operator_order(self):
        with SchemaCheckMode() as schema_check:
            x = torch.rand((3, 3), requires_grad=True)
            x.relu().sin()
        self.assertEqual(["aten::rand", "aten::relu", "aten::detach", "aten::sin"], schema_check.ops)

    # Tests that SchemaCheckMode records operator order without grad
    def test_schema_check_mode_operator_order_without_grad(self):
        with SchemaCheckMode() as schema_check:
            x = torch.rand((3, 3), requires_grad=False)
            x.relu().sin()
        self.assertEqual(["aten::rand", "aten::relu", "aten::sin"], schema_check.ops)

    # Tests that SchemaCheckMode records mutations and aliases with none expected
    def test_schema_check_mode_mutated_aliasing_none(self):
        # NB: previously requires_grad=True, but this induces a detach for
        # saved variable
        x = torch.rand((3, 3))
        with SchemaCheckMode() as schema_check:
            actual = x.relu().sin()
        self.assertEqual([], schema_check.mutated)
        self.assertEqual([], schema_check.aliasing)

    # Tests that SchemaCheckMode records mutations and aliases with mutation expected
    def test_schema_check_mode_mutated_aliasing_mutation(self):
        actual = torch.rand((3, 3), requires_grad=False)
        with SchemaCheckMode() as schema_check:
            actual.sinh_()
        self.assertEqual([('aten::sinh_', 'input')], schema_check.mutated)
        self.assertEqual([('aten::sinh_', 'input', 'output_0')], schema_check.aliasing)

    # Tests that SchemaCheckMode records mutations and aliases with resize_
    def test_schema_check_mode_mutated_aliasing_resize_(self):
        actual = torch.rand((3, 3), requires_grad=False)
        with SchemaCheckMode() as schema_check:
            actual.resize_(9)
        self.assertEqual([('aten::resize_', 'input')], schema_check.mutated)
        self.assertEqual([('aten::resize_', 'input', 'output_0')], schema_check.aliasing)

    # Tests that SchemaCheckMode records mutations and aliases with aliasing inputs
    def test_schema_check_mode_mutated_aliasing_aliasing_inputs(self):
        actual = torch.rand((3, 3))
        y = actual
        with SchemaCheckMode() as schema_check:
            actual.add_(y)
        self.assertEqual(
            [
                ('aten::add_', 'input'),
                ('aten::add_', 'other')
            ],
            schema_check.mutated
        )
        self.assertEqual(
            [
                ('aten::add_', 'input', 'output_0'),
                ('aten::add_', 'other', 'output_0')
            ],
            schema_check.aliasing
        )

    # Tests that SchemaCheckMode records mutations and alias with as_strided
    def test_schema_check_mode_mutated_aliasing_as_strided(self):
        x = torch.rand((3, 6, 4))
        with SchemaCheckMode() as schema_check:
            x.as_strided_([3, 6, 4], [9, 1, 1])
        self.assertEqual(
            [
                ('aten::as_strided_', 'input')
            ],
            schema_check.mutated
        )
        self.assertEqual(
            [
                ('aten::as_strided_', 'input', 'output_0')
            ],
            schema_check.aliasing
        )

    # Tests that SchemaCheckMode records mutations and aliases with multiple outputs
    def test_schema_check_mode_mutated_aliasing_multiple_outputs(self):
        x = torch.arange(9.)
        m_actual = torch.arange(9.)
        e_actual = torch.zeros([9], dtype=torch.int32)
        with SchemaCheckMode() as schema_check:
            torch.frexp(x, out=(m_actual, e_actual))
        self.assertEqual(
            [
                ('aten::frexp', 'mantissa'),
                ('aten::frexp', 'exponent')
            ],
            schema_check.mutated
        )
        self.assertEqual(
            [
                ('aten::frexp', 'mantissa', 'output_0'),
                ('aten::frexp', 'exponent', 'output_1')
            ],
            schema_check.aliasing
        )

    # Tests that SchemaCheckMode records mutations and aliases with aliasing outputs
    def test_schema_check_mode_mutated_aliasing_aliasing_outputs(self):
        x = torch.rand((3, 3))
        actual = torch.zeros(3)
        with SchemaCheckMode() as schema_check:
            torch.aminmax(x, dim=0, out=[actual, actual])
        self.assertEqual(
            [
                ('aten::aminmax', 'min'),
                ('aten::aminmax', 'max')
            ],
            schema_check.mutated
        )
        self.assertEqual(
            [
                ('aten::aminmax', 'min', 'output_0'),
                ('aten::aminmax', 'min', 'output_1'),
                ('aten::aminmax', 'max', 'output_0'),
                ('aten::aminmax', 'max', 'output_1')
            ],
            schema_check.aliasing
        )

    # Tests that SchemaCheckMode wraps torch.Tensor
    def test_schema_check_mode_functionality(self):
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.relu().sin()
        with SchemaCheckMode():
            actual = x.relu().sin()
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when an argument's default is overriden
    def test_schema_check_mode_functionality_default_replaced(self):
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.add(x, alpha=2)
        with SchemaCheckMode():
            actual = x.add(x, alpha=2)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when there is a Tensor[] argument
    def test_schema_check_mode_functionality_list_input(self):
        a = torch.rand((3, 3))
        b = torch.rand((3, 3))
        c = torch.rand((3, 3))
        expected = torch.linalg.multi_dot([a, b, c])
        with SchemaCheckMode():
            actual = torch.linalg.multi_dot([a, b, c])
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor with an op that has the (a -> *) notation
    def test_schema_check_mode_functionality_wildcard_after(self):
        x = torch.rand((3, 3))
        expected = x.chunk(6)
        with SchemaCheckMode():
            actual = x.chunk(6)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor when there is a kwarg tensor input
    @unittest.skipIf(not torch._C.has_spectral, "ATen not built with FFT.")
    def test_schema_check_mode_functionality_kwarg_tensor(self):
        x = torch.rand((3, 5))
        w = torch.rand(4)
        expected = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        with SchemaCheckMode():
            actual = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps torch.Tensor with a mutable op
    def test_schema_check_mode_functionality_mutable_inputs(self):
        expected = torch.rand((3, 3), requires_grad=False)
        actual = torch.clone(expected)
        expected.sinh_()
        with SchemaCheckMode():
            actual.sinh_()
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps Torch.tensor when inputs alias
    def test_schema_check_mode_functionality_aliasing_inputs(self):
        expected = torch.rand((3, 3))
        x = expected
        actual = torch.clone(expected)
        y = actual
        expected.add_(x)
        with SchemaCheckMode():
            actual.add_(y)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps Torch.tensor with multiple tensor outputs
    def test_schema_check_mode_functionality_with_multiple_outputs(self):
        x = torch.arange(9.)
        m_expected, e_expected = torch.frexp(x)
        m_actual = torch.arange(9.)
        e_actual = torch.zeros([9], dtype=torch.int32)
        with SchemaCheckMode():
            torch.frexp(x, out=(m_actual, e_actual))
        self.assertEqual(m_expected, m_actual)
        self.assertEqual(e_expected, e_actual)

    # Tests that SchemaCheckMode wraps Torch.tensor with aliasing outputs due to aliasing inputs
    def test_schema_check_mode_functionality_with_multiple_outputs_aliasing(self):
        x = torch.rand((3, 3))
        actual = torch.zeros(3)
        with SchemaCheckMode():
            torch.aminmax(x, dim=0, out=[actual, actual])
        self.assertEqual(torch.amax(x, dim=0), actual)

    # Tests that SchemaCheckMode wraps Torch.tensor in ops with real Device input
    def test_schema_check_mode_functionality_device_input(self):
        with SchemaCheckMode():
            x = torch.rand((3, 3), device="cpu", dtype=torch.double)
            y = x + x
        self.assertEqual(x + x, y)

    # Tests that SchemaCheckMode wraps Torch.tensor in special training op edge case
    def test_schema_check_mode_functionality_training_op(self):
        x = torch.rand((3, 3), requires_grad=True)
        batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
        expected = batch(x)
        with SchemaCheckMode():
            actual = batch(x)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps Torch.tensor with nested training op edge case
    def test_schema_check_mode_functionality_nested_training_op(self):
        actual = torch.rand((3, 3))
        batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
        expected = torch.clone(actual)
        expected.sinh_()
        expected.tanh_()
        expected.relu_()
        expected = batch(expected)

        with SchemaCheckMode():
            actual.sinh_()
            actual.tanh_()
            actual.relu_()
            actual = batch(actual)
        self.assertEqual(expected, actual)

    # Tests that SchemaCheckMode wraps Torch.tensor with empty list input
    def test_schema_check_mode_empty_list_input(self):
        expected = torch.atleast_1d([])
        with SchemaCheckMode():
            actual = torch.atleast_1d([])
        self.assertEqual(expected, actual)

    # Tests that an exception is raised for a mismatching mutation
    def test_mutation_check_fail(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined as mutable but was mutated"):
            x = torch.rand((3, 3))
            y = torch.rand((3, 3))
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sub(IncorrectAliasTensor(y))

    # # Tests that an exception is raised for a mismatching mutation over multiple ops
    def test_mutation_check_fail_multiple_operators(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined as mutable but was mutated"):
            x = torch.rand((3, 3))
            y = torch.rand((3, 3))
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sin().cos().sub(IncorrectAliasTensor(y))

    # Tests that an exception is raised for a mismatching alias
    def test_alias_check_fail_simple(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.rand((3, 3))
            with SchemaCheckMode():
                IncorrectAliasTensor(x).add(IncorrectAliasTensor(y), alpha=2)

    # Tests that an exception is raised for a mismatching alias over multiple ops
    def test_alias_check_fail_multiple_operators(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sin().relu().add(IncorrectAliasTensor(y), alpha=2)

    # Tests that an exception is raised for a centered mismatching alias over multiple ops
    def test_alias_check_fail_multiple_operators_centered(self):
        with self.assertRaisesRegex(RuntimeError, "Argument input is not defined to alias output but was aliasing"):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sin().add(IncorrectAliasTensor(y), alpha=2).relu()

    # Tests that an exception is raised for a centered mismatching alias over multiple ops
    def test_alias_check_fail_outputs_unexpectedly_aliasing(self):
        with self.assertRaisesRegex(RuntimeError, "Outputs 0 and 1 alias unexpectedly"):
            x = torch.rand((3, 3))
            with SchemaCheckMode() as s:
                IncorrectAliasTensor(x).aminmax(dim=0)

    # When this file was written, python op registration didn't exist.
    # It's probably worth re-writing the entire file to use it,
    # but instead I just added extra tests.
    def test_alias_check_fail_custom_ops_secretly_aliasing(self):
        def f(x):
            return torch.ops.bad_schemas.secretly_aliasing(x)

        x = torch.rand((3, 3))
        with self.assertRaisesRegex(RuntimeError, "not defined to alias output but was aliasing"):
            with SchemaCheckMode() as s:
                out = f(x)

    def test_alias_check_fail_custom_ops_secretly_mutating(self):
        def f(x):
            return torch.ops.bad_schemas.secretly_mutating(x)

        x = torch.rand((3, 3))
        with self.assertRaisesRegex(RuntimeError, "not defined as mutable but was mutated"):
            with SchemaCheckMode() as s:
                out = f(x)

    def test_alias_check_fail_custom_ops_output_is_input(self):
        def f(x):
            return torch.ops.bad_schemas.output_is_input(x)

        x = torch.rand((3, 3))
        with self.assertRaisesRegex(RuntimeError, "are not allowed to directly return inputs"):
            with SchemaCheckMode() as s:
                out = f(x)

    # Tests that is_alias_of returns as expected
    def test_is_alias_of_basic(self):
        x = torch.rand((3, 3), requires_grad=True)
        y = torch.rand((3, 3), requires_grad=True)
        y = x.add(x, alpha=2)
        self.assertTrue(torch._C._is_alias_of(x, x))
        self.assertFalse(torch._C._is_alias_of(x, y))

    # Tests that is_alias_of returns as expected with empty containers
    def test_is_alias_of_empty_container(self):
        x = []
        y = torch.rand((3, 3), requires_grad=True)
        self.assertFalse(torch._C._is_alias_of(x, x))
        self.assertFalse(torch._C._is_alias_of(x, y))

    # Tests that overlaps returns as expected
    def test_overlaps_basic(self):
        x = torch.rand((3, 3), requires_grad=True)
        y = torch.rand((3, 3), requires_grad=True)
        z = [x, y]
        self.assertTrue(torch._C._overlaps(x, x))
        self.assertFalse(torch._C._overlaps(x, y))
        self.assertTrue(torch._C._overlaps(z, x))
        self.assertTrue(torch._C._overlaps(z, y))

    # Tests that overlaps returns correctly with empty containers
    def test_overlaps_empty_container(self):
        x = []
        y = [torch.rand((3, 3), requires_grad=True)]
        # Empty containers return false
        self.assertFalse(torch._C._overlaps(y, x))
        self.assertTrue(torch._C._overlaps(y, y))

    # Tests that SchemaInfo Bindings work as expected
    def test_schema_info_bind_basic(self):
        class SchemaInfoBindTestMode(TorchDispatchMode):
            def __init__(self, test_self):
                self.test_self = test_self

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                named_arg_list = normalize_function(
                    func,
                    args,
                    kwargs,
                    normalize_to_only_use_kwargs=True
                ).kwargs
                schema_info_value_test = torch._C._SchemaInfo(func._schema)
                schema_info_values_test = torch._C._SchemaInfo(func._schema)
                self.test_self.assertFalse(schema_info_value_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                self.test_self.assertFalse(schema_info_values_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                for i in named_arg_list:
                    schema_info_value_test.add_argument_value(i, named_arg_list[i])
                schema_info_values_test.add_argument_values(named_arg_list)
                self.test_self.assertTrue(schema_info_value_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                self.test_self.assertTrue(schema_info_values_test.may_alias(
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0),
                    torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))

                return func(*args, **kwargs)
        x = torch.rand((3, 3))
        with SchemaInfoBindTestMode(self) as schemaInfoCheck:
            x.add(x)


class TestSchemaCheckModeOpInfo(JitTestCase):
    @ops(op_db, dtypes=OpDTypes.supported)
    def test_schema_correctness(self, device, dtype, op):
        # Currently torch.equal isn't supported with torch.complex32
        # There's also errors with complex64 and complex128
        if (dtype == torch.complex32):
            return
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            with SchemaCheckMode():
                op(sample.input, *sample.args, **sample.kwargs)

instantiate_device_type_tests(TestSchemaCheckModeOpInfo, globals(), only_for=("cpu", "cuda"))

if __name__ == '__main__':
    run_tests()
