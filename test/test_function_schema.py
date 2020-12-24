
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

if __name__ == '__main__':
    run_tests()
