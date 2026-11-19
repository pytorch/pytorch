# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import random

from torch.distributed.tensor import Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    _rebuild_schema_from_cache,
    _split_schema_for_cache,
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    from torch.utils._cxx_pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_leaves


class TestOpSchema(TestCase):
    def test_equality_checks_lists_of_dtensor_spec(self):
        """If x == y, then we must have h(x) == h(y)."""
        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema1 = OpSchema(op=None, args_schema=(dts, [dts]), kwargs_schema={})
        schema2 = OpSchema(op=None, args_schema=(dts, [dts, dts]), kwargs_schema={})
        # This is a regression test; these schemas used to compare equal.
        self.assertNotEqual(schema1, schema2)
        self.assertNotEqual(hash(schema1), hash(schema2))

    def test_equality_respects_static_attributes(self):
        def _get_sample_op_schemas(static_arg_val, static_kwarg_val):
            dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
            static_argnum = 2
            static_kwargkey = ["statickwarg"]
            annotated_schemas = [
                (False, False, None),
                (True, False, RuntimeSchemaInfo(static_argnum=static_argnum)),
                (False, True, RuntimeSchemaInfo(static_kwargkey=static_kwargkey)),
                (
                    True,
                    True,
                    RuntimeSchemaInfo(
                        static_argnum=static_argnum, static_kwargkey=static_kwargkey
                    ),
                ),
            ]

            # non-tensor args show up in hash iff the argnum is static/
            # kwargs show up in hash iff their name is in static_kwargkey.
            # random elements are random because they are not supposed to matter for
            # equality at all.
            args_schema = (dts, random.randint(1, 1000000), static_arg_val)
            kwargs_schema = {
                "ignoredkwarg": random.randint(1, 1000000),
                "statickwarg": static_kwarg_val,
            }
            return [
                (
                    has_static_arg,
                    has_static_kwarg,
                    OpSchema(
                        op=None,
                        args_schema=args_schema,
                        kwargs_schema=kwargs_schema,
                        schema_info=si,
                    ),
                )
                for (has_static_arg, has_static_kwarg, si) in annotated_schemas
            ]

        for lhs_has_static_arg, lhs_has_static_kwarg, lhs in _get_sample_op_schemas(
            1, 2
        ):
            # Static arg/kwarg both match
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                1, 2
            ):
                if (
                    lhs_has_static_arg == rhs_has_static_arg
                    and lhs_has_static_kwarg == rhs_has_static_kwarg
                ):
                    self.assertEqual(lhs, rhs)
                else:
                    self.assertNotEqual(lhs, rhs)

            # Static arg mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                3, 2
            ):
                if (
                    lhs_has_static_arg
                    or rhs_has_static_arg
                    or lhs_has_static_kwarg != rhs_has_static_kwarg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

            # Static kwarg mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                1, 3
            ):
                if (
                    lhs_has_static_kwarg
                    or rhs_has_static_kwarg
                    or lhs_has_static_arg != rhs_has_static_arg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

            # Static arg/kwarg both mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                3, 4
            ):
                if (
                    lhs_has_static_arg
                    or rhs_has_static_arg
                    or lhs_has_static_kwarg
                    or rhs_has_static_kwarg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

    def test_no_dtensor_spec_leak_in_comparison_key(self):
        """Regression test for https://github.com/pytorch/pytorch/issues/181761.

        DTensor_OpSchema_recompute_comparison_key_impl used to release() the
        inner args_to_hash tuple before passing it to PyTuple_Pack on the
        no-static_kwargkey path, leaking one ref to every DTensorSpec in
        args_schema for the lifetime of the process.
        """
        import weakref

        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        ref = weakref.ref(dts)
        OpSchema(op=None, args_schema=(dts,), kwargs_schema={})
        del dts
        self.assertIsNone(ref(), "DTensorSpec leaked via OpSchema._comparison_key")

    def test_split_rebuild_round_trip(self):
        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema = OpSchema(op=None, args_schema=(dts, [dts]), kwargs_schema={})
        # schema with specs in args
        skeleton, specs = _split_schema_for_cache(schema)
        rebuilt = _rebuild_schema_from_cache(skeleton, specs)
        self.assertEqual(rebuilt, schema)
        self.assertEqual(hash(rebuilt), hash(schema))
        self.assertIsInstance(rebuilt.args_schema[1], list)

    def test_skeleton_holds_no_spec(self):
        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema = OpSchema(op=None, args_schema=(dts, [dts]), kwargs_schema={})
        skeleton, _ = _split_schema_for_cache(schema)
        _, args_sk, kwargs_sk, schema_info = skeleton
        for leaf in tree_leaves((args_sk, kwargs_sk)):
            self.assertNotIsInstance(leaf, DTensorSpec)

    def test_none_in_args_survives(self):
        """A literal None must not be confused with a removed spec.

        The placeholder is a dedicated type rather than None precisely because
        None is a legal args_schema value. With None as the placeholder the
        rebuild cannot tell an original None from a removed spec.
        """
        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema = OpSchema(
            op=None,
            args_schema=(None, dts, [None, dts]),
            kwargs_schema={"k": None},
        )
        skeleton, specs = _split_schema_for_cache(schema)

        # Only the two specs were taken. The three Nones stayed in the skeleton.
        self.assertEqual(len(specs), 2)

        rebuilt = _rebuild_schema_from_cache(skeleton, specs)
        self.assertIsNone(rebuilt.args_schema[0])
        self.assertIsNone(rebuilt.args_schema[2][0])
        self.assertIsNone(rebuilt.kwargs_schema["k"])
        self.assertEqual(rebuilt, schema)

    def test_rebuild_substitutes_new_specs(self):
        a = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema = OpSchema(op=None, args_schema=(a, 7), kwargs_schema={"out": a})
        skeleton, specs = _split_schema_for_cache(schema)
        fresh = [
            DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None) for _ in specs
        ]
        rebuilt = _rebuild_schema_from_cache(skeleton, fresh)
        self.assertIs(rebuilt.args_schema[0], fresh[0])
        self.assertIs(rebuilt.kwargs_schema["out"], fresh[1])
        self.assertEqual(rebuilt.args_schema[1], 7)

    def test_needs_pytree_round_trip(self):
        a, b, c, d = (
            DTensorSpec(mesh=None, placements=(Shard(0),) * i, tensor_meta=None)
            for i in range(4)
        )
        schema = OpSchema(
            op=None,
            args_schema=([a, b], [c, d], 1.0),
            kwargs_schema={},
            schema_info=RuntimeSchemaInfo(needs_pytree=True),
        )
        skeleton, specs = _split_schema_for_cache(schema)
        rebuilt = _rebuild_schema_from_cache(skeleton, specs)

        self.assertEqual(len(rebuilt.args_spec), 4)
        self.assertEqual(rebuilt.schema_info, schema.schema_info)
        self.assertIsInstance(rebuilt.args_schema[0], list)
        self.assertEqual(rebuilt.args_schema[2], 1.0)
        self.assertEqual(rebuilt, schema)

    def test_split_refuses_unreachable_specs(self):
        spec = DTensorSpec(mesh=None, placements=(Shard(0),), tensor_meta=None)
        strategy = OpStrategy([OpSpec(output_specs=spec)])
        self.assertIsNone(_split_schema_for_cache(OpSchema(None, (strategy,), {})))

        # TupleStrategy is a registered pytree node, so the walk descends into it
        # and its OpStrategy children surface as leaves.
        tup = TupleStrategy([OpStrategy([OpSpec(output_specs=spec)])])
        self.assertIsNone(_split_schema_for_cache(OpSchema(None, (tup,), {})))


if __name__ == "__main__":
    run_tests()
