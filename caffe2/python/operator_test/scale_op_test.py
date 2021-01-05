




from caffe2.python import core, workspace

import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np


class TestScaleOps(serial.SerializedTestCase):
    @serial.given(dim=st.sampled_from([[1, 386, 1], [386, 1, 1],
                                       [1, 256, 1], [256, 1, 1],
                                       [1024, 256, 1], [1, 1024, 1],
                                       [1, 1, 1]]),
                    scale=st.floats(0.0, 10.0),
                    num_tensors=st.integers(1, 10),
                    **hu.gcs)
    def test_scale_ops(self, dim, scale, num_tensors, gc, dc):
        in_tensors = []
        in_tensor_ps = []
        out_tensors = []
        out_ref_tensors = []
        # initialize tensors
        for i in range(num_tensors):
            tensor = "X_{}".format(i)
            X = np.random.rand(*dim).astype(np.float32) - 0.5
            in_tensors.append(tensor)
            in_tensor_ps.append(X)
            out_tensor = "O_{}".format(i)
            out_tensors.append(out_tensor)
            workspace.FeedBlob(tensor, X, device_option=gc)

        # run ScaleBlobs operator
        scale_blobs_op = core.CreateOperator(
            "ScaleBlobs",
            in_tensors,
            out_tensors,
            scale=scale,
        )
        scale_blobs_op.device_option.CopyFrom(gc)
        workspace.RunOperatorOnce(scale_blobs_op)

        # run Scale op for each tensor and compare with ScaleBlobs
        for i in range(num_tensors):
            tensor = "X_{}".format(i)
            out_ref_tensor = "O_ref_{}".format(i)
            scale_op = core.CreateOperator(
                "Scale",
                [tensor],
                [out_ref_tensor],
                scale=scale,
            )
            scale_op.device_option.CopyFrom(gc)
            workspace.RunOperatorOnce(scale_op)
            o_ref = workspace.FetchBlob(out_ref_tensor)
            o = workspace.FetchBlob(out_tensors[i])
            np.testing.assert_allclose(o, o_ref)

if __name__ == '__main__':
    unittest.main()
