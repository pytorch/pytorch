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

import numpy as np
import hypothesis.strategies as st
import unittest
import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core
from hypothesis import given


class TestUpSample(hu.HypothesisTestCase):
    @given(height_scale=st.floats(1.0, 4.0) | st.just(2.0),
           width_scale=st.floats(1.0, 4.0) | st.just(2.0),
           height=st.integers(4, 32),
           width=st.integers(4, 32),
           num_channels=st.integers(1, 4),
           batch_size=st.integers(1, 4),
           seed=st.integers(0, 65535),
           **hu.gcs_cpu_only)
    def test_upsample(self, height_scale, width_scale, height, width,
                     num_channels, batch_size, seed,
                     gc, dc):

        np.random.seed(seed)
        op = core.CreateOperator(
            "UpsampleBilinear",
            ["X"],
            ["Y"],
            width_scale=width_scale,
            height_scale=height_scale,
        )

        X = np.random.rand(
            batch_size, num_channels, height, width).astype(np.float32)

        def ref(X):
            output_height = np.int32(height * height_scale)
            output_width = np.int32(width * width_scale)

            Y = np.random.rand(
                batch_size, num_channels, output_height,
                output_width).astype(np.float32)

            rheight = ((height - 1) / (output_height - 1)
                    if output_height > 1
                    else float(0))
            rwidth = (width - 1) / (output_width - 1) if output_width > 1 else float(0)

            for i in range(output_height):
                h1r = rheight * i
                h1 = int(h1r)
                h1p = 1 if h1 < height - 1 else 0
                h1lambda = h1r - h1
                h0lambda = float(1) - h1lambda
                for j in range(output_width):
                    w1r = rwidth * j
                    w1 = int(w1r)
                    w1p = 1 if w1 < width - 1 else 0
                    w1lambda = w1r - w1
                    w0lambda = float(1) - w1lambda
                    Y[:, :, i, j] = (h0lambda * (w0lambda * X[:, :, h1, w1] +
                        w1lambda * X[:, :, h1, w1 + w1p]) +
                        h1lambda * (w0lambda * X[:, :, h1 + h1p, w1] +
                        w1lambda * X[:, :, h1 + h1p, w1 + w1p]))

            return Y,

        self.assertReferenceChecks(gc, op, [X], ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=0.1, threshold=1e-2)

    @given(height_scale=st.floats(1.0, 4.0) | st.just(2.0),
           width_scale=st.floats(1.0, 4.0) | st.just(2.0),
           height=st.integers(4, 32),
           width=st.integers(4, 32),
           num_channels=st.integers(1, 4),
           batch_size=st.integers(1, 4),
           seed=st.integers(0, 65535),
           **hu.gcs_cpu_only)
    def test_upsample_grad(self, height_scale, width_scale, height, width,
                          num_channels, batch_size, seed, gc, dc):

        np.random.seed(seed)

        output_height = np.int32(height * height_scale)
        output_width = np.int32(width * width_scale)
        X = np.random.rand(batch_size,
                           num_channels,
                           height,
                           width).astype(np.float32)
        dY = np.random.rand(batch_size,
                            num_channels,
                            output_height,
                            output_width).astype(np.float32)

        op = core.CreateOperator(
            "UpsampleBilinearGradient",
            ["dY", "X"],
            ["dX"],
            width_scale=width_scale,
            height_scale=height_scale,
        )

        def ref(dY, X):
            dX = np.zeros_like(X)

            rheight = ((height - 1) / (output_height - 1)
                    if output_height > 1
                    else float(0))
            rwidth = (width - 1) / (output_width - 1) if output_width > 1 else float(0)

            for i in range(output_height):
                h1r = rheight * i
                h1 = int(h1r)
                h1p = 1 if h1 < height - 1 else 0
                h1lambda = h1r - h1
                h0lambda = float(1) - h1lambda
                for j in range(output_width):
                    w1r = rwidth * j
                    w1 = int(w1r)
                    w1p = 1 if w1 < width - 1 else 0
                    w1lambda = w1r - w1
                    w0lambda = float(1) - w1lambda
                    dX[:, :, h1, w1] += h0lambda * w0lambda * dY[:, :, i, j]
                    dX[:, :, h1, w1 + w1p] += h0lambda * w1lambda * dY[:, :, i, j]
                    dX[:, :, h1 + h1p, w1] += h1lambda * w0lambda * dY[:, :, i, j]
                    dX[:, :, h1 + h1p, w1 + w1p] += h1lambda * w1lambda * dY[:, :, i, j]

            return dX,

        self.assertDeviceChecks(dc, op, [dY, X], [0])
        self.assertReferenceChecks(gc, op, [dY, X], ref)


if __name__ == "__main__":
    unittest.main()
