# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._C import parse_schema


class TestFunctionSchema(TestCase):
    def test_serialize_and_deserialize(self):
        schemas = torch._C._jit_get_all_schemas()
        # so far we have around 1700 registered schemas
        self.assertGreater(len(schemas), 1000)
        for schema in schemas:
            parsed_schema = parse_schema(str(schema))
            self.assertEqual(parsed_schema, schema)
            self.assertTrue(parsed_schema.is_backward_compatible_with(schema))

    def test_out_schema(self):
        schema_with_out = parse_schema('any.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)')
        self.assertTrue(schema_with_out.arguments[-1].is_out)
        schema_without_out = parse_schema('any.not_out(Tensor self, Tensor b) -> Tensor')
        self.assertFalse(schema_without_out.arguments[-1].is_out)

    def test_backward_compatible_structure(self):
        old_schema = parse_schema('any.over(Tensor self, *, Tensor b) -> Tensor')
        # BC: A new schema without changes.
        new_schema = parse_schema('any.over(Tensor self, *, Tensor b) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema with different name.
        new_schema = parse_schema('any_.over(Tensor self, *, Tensor b) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema with different overload name.
        new_schema = parse_schema('any.other(Tensor self, *, Tensor b) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema that adds vararg.
        new_schema = parse_schema('any.over(Tensor self, *, Tensor b, ...) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema with different number of outputs.
        new_schema = parse_schema('any.over(Tensor self, *, Tensor b) -> (Tensor, Tensor)')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))

    def test_backward_compatible_outputs(self):
        old_schema = parse_schema('any.over(Tensor self, *, Tensor b) -> Tensor')
        # No-BC: A new schema with output becoming of optional type.
        new_schema = parse_schema('any.over(Tensor self, *, Tensor b) -> Tensor?')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # BC: (the opposite case) An schema where the output is not of optional type anymore.
        self.assertTrue(old_schema.is_backward_compatible_with(new_schema))
        # No-BC: A new schema with a different output type.
        new_schema = parse_schema('any.over(Tensor self, *, Tensor b) -> int')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema with a different output type.
        new_schema = parse_schema('any.over(Tensor self, *, Tensor b) -> Tensor out')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))

    def test_backward_compatible_arguments(self):
        old_schema = parse_schema('any(Tensor self, *, Tensor b, int c) -> Tensor')
        # No-BC: A new schema with less arguments.
        new_schema = parse_schema('any(Tensor self, *, Tensor b) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema with more arguments, appended, but no default value.
        new_schema = parse_schema('any(Tensor self, *, Tensor b, int c, int d) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # BC: A new schema with more arguments, appended, that have a default value.
        new_schema = parse_schema('any(Tensor self, *, Tensor b, int c, int d=1) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema with more arguments, not-appended, that have a default value.
        new_schema = parse_schema('any(Tensor self, int d=1, *, Tensor b, int c) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # BC: A new schema where old kwargs becomes positional.
        new_schema = parse_schema('any(Tensor self, Tensor b, *, int c) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        # BC: (the opposite case) A new schema where an old positional argument becomes kwarg.
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        # BC: A new schema where all old kwargs become positional.
        new_schema = parse_schema('any(Tensor self, Tensor b, int c) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        # BC: (the opposite case) A new schema where all old positional arguments become kwarg.
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        # No-BC: A new schema where old kwargs appear in different order.
        new_schema = parse_schema('any(Tensor self, *, int c, Tensor b) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # BC: A new schema where argument becomes of type optional.
        new_schema = parse_schema('any(Tensor self, *, Tensor b, int? c) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        # BC: A new schema where argument gains a default value.
        new_schema = parse_schema('any(Tensor self, *, Tensor b, int c=1) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema where argument is "renamed".
        new_schema = parse_schema('any(Tensor self, *, Tensor b, int renamed) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        # No-BC: A new schema where argument type changes to an incompatible type.
        new_schema = parse_schema('any(Tensor self, *, Tensor b, int[] c) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))

    def test_backward_compatible_with_smart_serialization(self):
        # cases where out arg is provided
        old_schema = parse_schema('foo(Tensor self, *, int a, Tensor(a!) out) -> Tensor(a!)')
        new_schema_same_out = parse_schema('foo(Tensor self, *, int a, int b=1, Tensor(a!) out) -> Tensor(a!)')
        new_schema_wrong_default = parse_schema('foo(Tensor self, *, int b=1, int a, Tensor(a!) out) -> Tensor(a!)')
        new_schema_more_out = parse_schema('foo(Tensor self, *, int a, int b=1, Tensor(a!) out, Tensor(b!) b) -> Tensor(a!)')
        new_schema_wrong_pos = parse_schema('foo(Tensor self, *, int a, int b=1, Tensor(b!) b, Tensor(a!) out) -> Tensor(a!)')
        self.assertTrue(new_schema_same_out.is_backward_compatible_with(old_schema))
        self.assertTrue(new_schema_more_out.is_backward_compatible_with(old_schema))
        self.assertFalse(new_schema_wrong_default.is_backward_compatible_with(old_schema))
        self.assertFalse(new_schema_wrong_pos.is_backward_compatible_with(old_schema))

        # cases where out arg is not provided
        old_schema_without_arg = parse_schema('foo(Tensor self, int a, int b=1) -> int')
        new_schema_without_arg = parse_schema('foo(Tensor self, int a, int b=1, int c=2) -> int')
        new_schema_without_arg_multiple_default = parse_schema('foo(Tensor self, int a, int b=1, int c=2, int d=3) -> int')
        new_schema_without_arg_wrong_pos = parse_schema('foo(Tensor self, int a, int c=2, int b=1) -> int')
        self.assertTrue(new_schema_without_arg.is_backward_compatible_with(old_schema_without_arg))
        self.assertTrue(new_schema_without_arg_multiple_default.is_backward_compatible_with(old_schema_without_arg))
        self.assertFalse(new_schema_without_arg_wrong_pos.is_backward_compatible_with(old_schema_without_arg))

    def test_string_optional_parameter_default_value(self):
        schema_a = parse_schema("example::op(str? order=\"NCHW\") -> (Tensor)")
        schema_b = parse_schema(str(schema_a))
        self.assertEqual(schema_a, schema_b)

    def test_forward_compatible_arguments_without_out(self):
        old_schema = parse_schema('any(Tensor self, int a, int b=1) -> Tensor')
        # deleting default arg is FC compatible
        new_schema = parse_schema('any(Tensor self, int a) -> Tensor')
        is_fc, _ = new_schema.check_forward_compatible_with(old_schema)
        self.assertTrue(is_fc)
        # adding default arg is FC compatible
        new_schema = parse_schema('any(Tensor self, int a, int b=1, int c=1) -> Tensor')
        is_fc, _ = new_schema.check_forward_compatible_with(old_schema)
        self.assertTrue(is_fc)
        # adding default arg with container type is NOT FC compatible
        new_schema = parse_schema('any(Tensor self, int a, int b=1, int[2] c=1) -> Tensor')
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "Function schema is not forward compatible since the new argument"
                                 " \'c\' of type int[] has a container type as its default value.")
        # updating the default value of a default arg is NOT FC compatible
        new_schema = parse_schema('any(Tensor self, int a, int b=4) -> Tensor')
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "\'b\' is not forward compatible with the older version of the schema")
        # updating the arg name of a default arg is NOT FC compatible
        new_schema = parse_schema('any(Tensor self, int a, int c=1) -> Tensor')
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "\'c\' is not forward compatible with the older version of the schema")
        # not adding default arg in the end is NOT FC compatible
        new_schema = parse_schema('any(Tensor self, int a, int c=1, int b=1) -> Tensor')
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "\'c\' is not forward compatible with the older version of the schema")
        # making default arg into positional arg is NOT FC compatible
        new_schema = parse_schema('any(Tensor self, int a, int b) -> Tensor')
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "\'b\' is not forward compatible with the older version of the schema")
        # making positional arg into default arg is NOT FC compatible
        new_schema = parse_schema('any(Tensor self, int a=1, int b=1) -> Tensor')
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "\'a\' is not forward compatible with the older version of the schema")

    def test_forward_compatible_arguments_real_use_case(self):
        # this change introduced forward incompatibility in the past
        old_slice_schema = parse_schema('slice(Tensor(a) self, int dim=0, int start=0, int end=0, int step=1) -> Tensor(a)')
        new_slice_schema = parse_schema('slice(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)')
        is_fc, reason = new_slice_schema.check_forward_compatible_with(old_slice_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "\'start\' is not forward compatible with the older version of the schema")

    def test_forward_compatible_arguments_with_out(self):
        old_schema = parse_schema('any(Tensor self, *, int a, int b=1, Tensor(a!) out) -> Tensor(a!)')
        new_schema = parse_schema('any(Tensor self, *, int a, Tensor(a!) out) -> Tensor(a!)')
        is_fc, _ = new_schema.check_forward_compatible_with(old_schema)
        self.assertTrue(is_fc)
        new_schema = parse_schema('any(Tensor self, *, int a, int b=1, int c=1, Tensor(a!) out) -> Tensor(a!)')
        is_fc, _ = new_schema.check_forward_compatible_with(old_schema)
        self.assertTrue(is_fc)
        new_schema = parse_schema('any(Tensor self, *, int a, Tensor(d!) d, int b=1, Tensor(a!) out) -> Tensor(a!)')
        is_fc, reason = new_schema.check_forward_compatible_with(old_schema)
        self.assertFalse(is_fc)
        self.assertEqual(reason, "Function schema should have the same number of out arguments")

    def test_schema_error(self):
        with self.assertRaisesRegex(RuntimeError, r"schemas with vararg \(...\) can't have default value args"):
            schema = parse_schema("any.foo(int arg1, int arg2=0, ...)")

    def test_tensor_list_alias_annotation_properly_parsed(self):
        schema_str = 'foo(Tensor self, *, Tensor(a!)[] out) -> ()'
        schema = parse_schema(schema_str)
        self.assertTrue(schema.arguments[-1].alias_info.is_write)
        self.assertEqual(str(schema), schema_str)

    def test_tensor_option_arguments_properly_parsed(self):
        schema_str = '_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, ' \
                     'bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor'
        schema = parse_schema(schema_str)
        # fake type of MemoryFormat? is int?
        self.assertEqual(schema.arguments[-1].type.str(), "int?")
        # fake type of Layout? is int?
        self.assertEqual(schema.arguments[2].type.str(), "int?")
        # fake type of Device? is Device?
        self.assertEqual(schema.arguments[3].type.str(), "Device?")
        # print real types in FunctionSchema
        self.assertEqual(str(schema), schema_str)


if __name__ == '__main__':
    run_tests()
