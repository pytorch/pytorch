from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from common_utils import TestCase
from torch._C import parse_schema


class TestFunctionSchema(TestCase):
    def test_serialize_and_deserialize(self):
        schemas = torch._C._jit_get_all_schemas()
        # so far we have around 1700 registered schemas
        self.assertGreater(len(schemas), 1000)
        for schema in schemas:
            parsed_schema = parse_schema(str(schema))
            self.assertEqual(parsed_schema, schema)
            self.assertTrue(parsed_schema.is_backcompat_with(schema))

    def test_backward_compatible_optional_arg(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int? dim, bool keepdim=False) -> Tensor')
        self.assertTrue(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))

    def test_backward_compatible_default_value(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int dim=5, bool keepdim=False) -> Tensor')
        self.assertTrue(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))

    def test_backward_compatible_add_args(self):
        old_schema = parse_schema('any(Tensor self, int dim) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        self.assertTrue(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool? keepdim) -> Tensor')
        self.assertTrue(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))

    def test_backward_incompatible_name(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any_(Tensor self, int dim, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))

    def test_backward_incompatible_vararg(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False, ...) -> Tensor')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))

    def test_backward_incompatible_varret(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> (Tensor, ...)')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))

    def test_backward_incompatible_returns(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> int')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor?')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor out')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))

    def test_backward_incompatible_args(self):
        old_schema = parse_schema('any(Tensor self, int[] dims, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor s, int[] dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int[] dims, *, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int[3] dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int[](a) dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backcompat_with(old_schema))
        self.assertFalse(old_schema.is_backcompat_with(new_schema))


if __name__ == '__main__':
    run_tests()
