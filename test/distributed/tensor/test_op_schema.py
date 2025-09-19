# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOpSchema(TestCase):
    def test_equality_checks_lists_of_dtensor_spec(self):
        """If x == y, then we must have h(x) == h(y)."""
        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema1 = OpSchema(op=None, args_schema=[dts, [dts]], kwargs_schema={})
        schema2 = OpSchema(op=None, args_schema=[dts, [dts, dts]], kwargs_schema={})
        # This is a regression test; these schemas used to compare equal.
        self.assertNotEqual(schema1, schema2)
        self.assertNotEqual(hash(schema1), hash(schema2))


if __name__ == "__main__":
    run_tests()
