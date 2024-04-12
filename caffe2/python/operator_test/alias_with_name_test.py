#!/usr/bin/env python3

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, utils
from hypothesis import given


class TestAliasWithNameOp(hu.HypothesisTestCase):
    @given(
        shape=st.lists(st.integers(0, 5), min_size=1, max_size=3),
        dtype=st.sampled_from([np.float32, np.int64]),
        **hu.gcs
    )
    def test_alias_with_name_op(self, shape, dtype, dc, gc):
        test_input = (100 * np.random.random(shape)).astype(dtype)
        test_inputs = [test_input]

        alias_op = core.CreateOperator(
            "AliasWithName",
            ["input"],
            ["output"],
            device_option=gc,
        )
        alias_op.arg.add().CopyFrom(utils.MakeArgument("name", "whatever_name"))

        def reference_func(x):
            return (x,)

        self.assertReferenceChecks(gc, alias_op, test_inputs, reference_func)
