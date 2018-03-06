from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np


class TestClipTensorByScalingOp(hu.HypothesisTestCase):

    @given(n=st.integers(5, 8), d=st.integers(2, 4),
           threshold=st.floats(0.1, 10),
           inplace=st.booleans(),
           **hu.gcs_cpu_only)
    def test_clip_tensor_by_scaling(self, n, d, threshold, inplace, gc, dc):

        tensor = np.random.rand(n, d).astype(np.float32)
        val = np.array(np.linalg.norm(tensor))

        def clip_tensor_by_scaling_ref(tensor_data, val_data):
            if val_data > threshold:
                ratio = threshold / float(val_data)
                tensor_data = tensor_data * ratio

            return [tensor_data]

        op = core.CreateOperator(
            "ClipTensorByScaling",
            ["tensor", "val"],
            ['Y'] if not inplace else ["tensor"],
            threshold=threshold,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[tensor, val],
            reference=clip_tensor_by_scaling_ref,
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
