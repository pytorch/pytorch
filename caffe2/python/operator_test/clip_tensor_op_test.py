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
           additional_threshold=st.floats(0.1, 10),
           use_additional_threshold=st.booleans(),
           inplace=st.booleans(),
           **hu.gcs_cpu_only)
    def test_clip_tensor_by_scaling(self, n, d, threshold, additional_threshold,
                                    use_additional_threshold, inplace, gc, dc):

        tensor = np.random.rand(n, d).astype(np.float32)
        val = np.array(np.linalg.norm(tensor))
        additional_threshold = np.array([additional_threshold]).astype(np.float32)

        def clip_tensor_by_scaling_ref(tensor_data, val_data,
                                       additional_threshold=None):

            if additional_threshold is not None:
                final_threshold = threshold * additional_threshold
            else:
                final_threshold = threshold

            if val_data > final_threshold:
                ratio = final_threshold / float(val_data)
                tensor_data = tensor_data * ratio

            return [tensor_data]

        op = core.CreateOperator(
            "ClipTensorByScaling",
            ["tensor", "val"] if not use_additional_threshold else (
                ["tensor", "val", "additional_threshold"]),
            ['Y'] if not inplace else ["tensor"],
            threshold=threshold,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[tensor, val] if not use_additional_threshold else (
                [tensor, val, additional_threshold]),
            reference=clip_tensor_by_scaling_ref,
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
