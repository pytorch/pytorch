#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

from torch.testing._internal.common_utils import run_tests, TestCase


os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

import os
import sys


# Make test/comms importable so `helpers` / `integration` resolve when this
# file is run directly (run_test.py runs `python comms/unit/<file>.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.comm_test_helpers import skip_if_torch_compile_not_supported_or_enabled

import torch
from torch.comms.functional.param_parsing import (
    CollectiveParamSchema,
    ParamKind,
    ParamSpec,
    ParsedArgs,
)


@skip_if_torch_compile_not_supported_or_enabled()
class TestParamSpec(TestCase):
    def test_has_default_true(self):
        spec = ParamSpec("x", ParamKind.INPUT, "Tensor", default_value=None)
        self.assertTrue(spec.has_default())

    def test_has_default_false(self):
        spec = ParamSpec("x", ParamKind.INPUT, "Tensor")
        self.assertFalse(spec.has_default())

    def test_is_tensor(self):
        spec = ParamSpec("x", ParamKind.INPUT, "Tensor")
        self.assertTrue(spec.is_tensor())
        self.assertFalse(spec.is_tensor_list())
        self.assertTrue(spec.is_tensor_like())

    def test_is_tensor_list(self):
        spec = ParamSpec("x", ParamKind.INPUT, "Tensor[]")
        self.assertFalse(spec.is_tensor())
        self.assertTrue(spec.is_tensor_list())
        self.assertTrue(spec.is_tensor_like())

    def test_not_tensor_like(self):
        spec = ParamSpec("x", ParamKind.EXTRA, "int")
        self.assertFalse(spec.is_tensor())
        self.assertFalse(spec.is_tensor_list())
        self.assertFalse(spec.is_tensor_like())


@skip_if_torch_compile_not_supported_or_enabled()
class TestParsedArgs(TestCase):
    def setUp(self):
        super().setUp()
        self.object_param = ParamSpec("self", ParamKind.CLASS_OBJECT, "MyClass")
        self.tensor_param = ParamSpec("tensor", ParamKind.INPUT, "Tensor", mutable=True)
        self.tensor_list_param = ParamSpec(
            "tensors", ParamKind.INPUT, "Tensor[]", mutable=True
        )
        self.extra_param = ParamSpec(
            "async_op", ParamKind.EXTRA, "bool", default_value=False
        )
        self.immutable_tensor_param = ParamSpec(
            "src", ParamKind.INPUT, "Tensor", mutable=False
        )

    def test_from_lib_args_basic(self):
        all_params = [self.object_param, self.tensor_param, self.extra_param]
        obj = object()
        tensor = torch.randn(3)
        args = (obj, tensor, True)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        self.assertIs(parsed.values[0], obj)
        self.assertIs(parsed.values[1], tensor)
        self.assertEqual(parsed.values[2], True)

    def test_from_method_args_positional(self):
        all_params = [self.object_param, self.tensor_param, self.extra_param]
        obj = object()
        tensor = torch.randn(3)

        parsed = ParsedArgs.from_method_args(obj, (tensor, True), {}, all_params)

        self.assertIs(parsed.values[0], obj)
        self.assertIs(parsed.values[1], tensor)
        self.assertEqual(parsed.values[2], True)

    def test_from_method_args_kwargs(self):
        all_params = [self.object_param, self.tensor_param, self.extra_param]
        obj = object()
        tensor = torch.randn(3)

        parsed = ParsedArgs.from_method_args(
            obj, (tensor,), {"async_op": True}, all_params
        )

        self.assertIs(parsed.values[0], obj)
        self.assertIs(parsed.values[1], tensor)
        self.assertEqual(parsed.values[2], True)

    def test_from_method_args_default(self):
        all_params = [self.object_param, self.tensor_param, self.extra_param]
        obj = object()
        tensor = torch.randn(3)

        parsed = ParsedArgs.from_method_args(obj, (tensor,), {}, all_params)

        self.assertIs(parsed.values[0], obj)
        self.assertIs(parsed.values[1], tensor)
        self.assertEqual(parsed.values[2], False)  # default value

    def test_get_value(self):
        all_params = [self.object_param, self.tensor_param, self.extra_param]
        obj = object()
        tensor = torch.randn(3)
        args = (obj, tensor, True)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        self.assertIs(parsed.get_value("self"), obj)
        self.assertIs(parsed.get_value("tensor"), tensor)
        self.assertEqual(parsed.get_value("async_op"), True)
        self.assertIsNone(parsed.get_value("nonexistent"))

    def test_tensor_input_indices(self):
        all_params = [
            self.object_param,
            self.tensor_param,
            self.extra_param,
        ]
        obj = object()
        tensor = torch.randn(3)
        args = (obj, tensor, False)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        # tensor_param is at index 1
        self.assertEqual(parsed.get_tensor_input_indices(), [1])

    def test_mutable_tensor_indices(self):
        all_params = [
            self.object_param,
            self.tensor_param,  # mutable
            self.immutable_tensor_param,  # immutable
            self.extra_param,
        ]
        obj = object()
        tensor = torch.randn(3)
        src = torch.randn(3)
        args = (obj, tensor, src, False)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        # Only tensor_param (index 1) is mutable
        self.assertEqual(parsed.get_mutable_tensor_indices(), [1])

    def test_mutable_outputs(self):
        all_params = [
            self.object_param,
            self.tensor_param,  # mutable
            self.extra_param,
        ]
        obj = object()
        tensor = torch.randn(3)
        args = (obj, tensor, False)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        self.assertEqual(len(parsed.get_mutable_outputs()), 1)
        self.assertIs(parsed.get_mutable_outputs()[0], tensor)

    def test_mutable_outputs_flat_with_list(self):
        all_params = [
            self.object_param,
            self.tensor_list_param,  # mutable tensor list
            self.extra_param,
        ]
        obj = object()
        tensors = [torch.randn(3), torch.randn(3)]
        args = (obj, tensors, False)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        # Should flatten the list
        flat = parsed.get_mutable_outputs_flat()
        self.assertEqual(len(flat), 2)
        self.assertIs(flat[0], tensors[0])
        self.assertIs(flat[1], tensors[1])

    def test_has_requires_grad(self):
        all_params = [self.object_param, self.tensor_param]
        obj = object()
        tensor = torch.randn(3, requires_grad=True)
        args = (obj, tensor)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        self.assertTrue(parsed.has_requires_grad())

    def test_has_requires_grad_false(self):
        all_params = [self.object_param, self.tensor_param]
        obj = object()
        tensor = torch.randn(3, requires_grad=False)
        args = (obj, tensor)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        self.assertFalse(parsed.has_requires_grad())

    def test_tensor_inputs_flat_with_mutable_mask(self):
        all_params = [
            self.object_param,
            self.tensor_param,  # mutable
            self.immutable_tensor_param,  # immutable
        ]
        obj = object()
        tensor = torch.randn(3)
        src = torch.randn(3)
        args = (obj, tensor, src)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        flat_inputs, mask = parsed.get_tensor_inputs_flat_with_mutable_mask()
        self.assertEqual(len(flat_inputs), 2)
        self.assertIs(flat_inputs[0], tensor)
        self.assertIs(flat_inputs[1], src)
        self.assertEqual(mask, [True, False])

    def test_to_values(self):
        all_params = [self.object_param, self.tensor_param, self.extra_param]
        obj = object()
        tensor = torch.randn(3)
        args = (obj, tensor, True)

        parsed = ParsedArgs.from_lib_args(args, all_params)

        values = parsed.to_values()
        self.assertEqual(len(values), 3)
        self.assertIs(values[0], obj)
        self.assertIs(values[1], tensor)
        self.assertEqual(values[2], True)


try:
    from torch._opaque_base import OpaqueBaseMeta

    class DummyClass(metaclass=OpaqueBaseMeta):
        """Dummy class for testing opaque type registration."""

except ImportError:
    pass  # skip test down below will catch this


@skip_if_torch_compile_not_supported_or_enabled()
class TestCollectiveParamSchema(TestCase):
    def test_from_raw_specs_basic(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertIsNotNone(schema.object_param)
        self.assertEqual(len(schema.input_params), 1)
        self.assertEqual(len(schema.extra_params), 1)
        self.assertEqual(len(schema.output_params), 0)

    def test_from_raw_specs_tensor_list(self):
        param_specs = [
            ParamSpec("tensors", ParamKind.INPUT, list[torch.Tensor], mutable=True),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertEqual(len(schema.input_params), 1)
        self.assertEqual(schema.input_params[0].torch_type, "Tensor[]")

    def test_all_params(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        # all_params = object + inputs + extras
        self.assertEqual(len(schema.all_params), 3)
        self.assertEqual(schema.all_params[0].name, "self")
        self.assertEqual(schema.all_params[1].name, "tensor")
        self.assertEqual(schema.all_params[2].name, "async_op")

    def test_mutable_params(self):
        param_specs = [
            ParamSpec("dst", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("src", ParamKind.INPUT, torch.Tensor, mutable=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertEqual(len(schema.mutable_params), 1)
        self.assertEqual(schema.mutable_params[0].name, "dst")

    def test_mutable_indices(self):
        param_specs = [
            ParamSpec("dst", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("src", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        # dst is at index 1 in all_params (after self)
        self.assertEqual(schema.mutable_indices, [1])

    def test_signature(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)
        sig = schema.signature

        # Should contain the opaque type for self
        self.assertIn("self", sig)
        # Should have mutable tensor annotation
        self.assertIn("Tensor(a!)", sig)
        self.assertIn("async_op", sig)

    def test_functional_signature(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)
        sig = schema.functional_signature

        # Should NOT have mutation annotation
        self.assertNotIn("!", sig)
        self.assertIn("Tensor tensor", sig)

    def test_inplace_return_type_single_tensor(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        # Single mutable tensor returns aliased type
        self.assertIn("Tensor(a!)", schema.inplace_return_type)

    def test_inplace_return_type_multiple_tensors(self):
        param_specs = [
            ParamSpec("dst", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("src", ParamKind.INPUT, torch.Tensor, mutable=True),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        # Multiple mutable tensors return tuple of aliased types
        ret = schema.inplace_return_type
        self.assertIn("Tensor(a!)", ret)
        self.assertIn("Tensor(b!)", ret)

    def test_functional_return_type_single_tensor(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        # Single mutable tensor returns plain Tensor
        self.assertEqual(schema.functional_return_type, "Tensor")

    def test_needs_async_dummy_return(self):
        # Async op with no mutable inputs needs dummy return
        param_specs = [
            ParamSpec("src", ParamKind.INPUT, torch.Tensor, mutable=False),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertTrue(schema.needs_async_dummy_return)

    def test_no_async_dummy_return_with_mutable(self):
        # Async op with mutable inputs doesn't need dummy return
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertFalse(schema.needs_async_dummy_return)

    def test_parse_lib_args(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        obj = DummyClass()
        tensor = torch.randn(3)
        parsed = schema.parse_lib_args((obj, tensor))

        self.assertIs(parsed.get_value("self"), obj)
        self.assertIs(parsed.get_value("tensor"), tensor)

    def test_parse_method_args(self):
        param_specs = [
            ParamSpec("tensor", ParamKind.INPUT, torch.Tensor, mutable=True),
            ParamSpec("async_op", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        obj = DummyClass()
        tensor = torch.randn(3)
        parsed = schema.parse_method_args(obj, (tensor,), {"async_op": True})

        self.assertIs(parsed.get_value("self"), obj)
        self.assertIs(parsed.get_value("tensor"), tensor)
        self.assertEqual(parsed.get_value("async_op"), True)

    def test_type_conversion_int(self):
        param_specs = [
            ParamSpec("count", ParamKind.EXTRA, int, default_value=1),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertEqual(schema.extra_params[0].torch_type, "int")

    def test_type_conversion_bool(self):
        param_specs = [
            ParamSpec("flag", ParamKind.EXTRA, bool, default_value=False),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertEqual(schema.extra_params[0].torch_type, "bool")

    def test_type_conversion_optional(self):
        param_specs = [
            ParamSpec("dtype", ParamKind.EXTRA, torch.dtype | None, default_value=None),
        ]

        schema = CollectiveParamSchema.from_raw_specs(DummyClass, param_specs)

        self.assertEqual(schema.extra_params[0].torch_type, "ScalarType?")


if __name__ == "__main__":
    run_tests()
