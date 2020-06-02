from __future__ import absolute_import, division, print_function, unicode_literals

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

    def test_backward_compatible_args(self):
        old_schema = parse_schema('any(Tensor self, int dim) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int? dim) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim=5) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))

    def test_backward_compatible_kwargs(self):
        old_schema = parse_schema('any(Tensor self, *, Tensor out) -> Tensor')
        new_schema = parse_schema('any(Tensor self, *, bool extra1=True, Tensor out, bool extra2=False) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, Tensor out) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))

    def test_backward_compatible_ret(self):
        old_schema = parse_schema('any(Tensor self) -> Tensor?')
        new_schema = parse_schema('any(Tensor self) -> Tensor')
        self.assertTrue(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))

    def test_backward_incompatible_name(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any_(Tensor self, int dim, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))

    def test_backward_incompatible_vararg(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False, ...) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))

    def test_backward_incompatible_returns(self):
        old_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> (Tensor, ...)')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> int')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor?')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertTrue(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dim, bool keepdim=False) -> Tensor out')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))

    def test_backward_incompatible_args(self):
        old_schema = parse_schema('any(Tensor self, int[] dims, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor s, int[] dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int[3] dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int[](a) dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int dims, bool keepdim=False) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int[] dim, bool keepdim=False, bool? extra=None) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))

    def test_backward_incompatible_kwargs(self):
        old_schema = parse_schema('any(Tensor self, int[] dims, *, bool keepdim=False) -> Tensor')
        new_schema = parse_schema('any(Tensor self, int[] dims, *, bool keepdim) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertTrue(old_schema.is_backward_compatible_with(new_schema))
        new_schema = parse_schema('any(Tensor self, int[] dims, *, bool keepdim=False, bool extra) -> Tensor')
        self.assertFalse(new_schema.is_backward_compatible_with(old_schema))
        self.assertFalse(old_schema.is_backward_compatible_with(new_schema))


if __name__ == '__main__':
    run_tests()
