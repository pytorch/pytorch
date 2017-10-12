# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestLengthsTileOp(hu.HypothesisTestCase):

    @given(
        inputs=st.integers(min_value=1, max_value=20).flatmap(
            lambda size: st.tuples(
                hu.arrays([size]),
                hu.arrays([size], dtype=np.int32,
                          elements=st.integers(min_value=0, max_value=20)),
            )
        ),
        **hu.gcs)
    def test_lengths_tile(self, inputs, gc, dc):
        data, lengths = inputs

        def lengths_tile_op(data, lengths):
            return [np.concatenate([
                [d] * l for d, l in zip(data, lengths)
            ])]

        op = core.CreateOperator(
            "LengthsTile",
            ["data", "lengths"],
            ["output"]
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[data, lengths],
            reference=lengths_tile_op,
        )

        self.assertGradientChecks(
            device_option=gc,
            op=op,
            inputs=[data, lengths],
            outputs_to_check=0,
            outputs_with_grads=[0]
        )
