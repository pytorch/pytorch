




import unittest
import numpy as np
from random import randint
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace


@unittest.skipIf(not workspace.C.use_mkldnn, "No MKLDNN support.")
class CopyTest(unittest.TestCase):
    def _get_deep_device(self):
        return caffe2_pb2.DeviceOption(device_type=caffe2_pb2.IDEEP)

    def test_copy_to_ideep(self):
        op = core.CreateOperator(
            "CopyCPUToIDEEP",
            ["X"],
            ["X_ideep"],
        )
        op.device_option.CopyFrom(self._get_deep_device())
        n = randint(1, 128)
        c = randint(1, 64)
        h = randint(1, 128)
        w = randint(1, 128)
        X = np.random.rand(n, c, h, w).astype(np.float32)
        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        X_ideep = workspace.FetchBlob("X_ideep")
        np.testing.assert_allclose(X, X_ideep)

    def test_copy_to_ideep_zero_dim(self):
        op = core.CreateOperator(
            "CopyCPUToIDEEP",
            ["X"],
            ["X_ideep"],
        )
        op.device_option.CopyFrom(self._get_deep_device())
        n = 0
        c = randint(1, 128)
        X = np.random.rand(n, c).astype(np.float32)
        workspace.FeedBlob("X", X)
        workspace.RunOperatorOnce(op)
        X_ideep = workspace.FetchBlob("X_ideep")
        np.testing.assert_allclose(X, X_ideep)

    def test_copy_from_ideep(self):
        op = core.CreateOperator(
            "CopyIDEEPToCPU",
            ["X_ideep"],
            ["X"],
        )
        op.device_option.CopyFrom(self._get_deep_device())
        n = randint(1, 128)
        c = randint(1, 64)
        h = randint(1, 128)
        w = randint(1, 128)
        X = np.random.rand(n, c, h, w).astype(np.float32)
        workspace.FeedBlob("X_ideep", X, self._get_deep_device())
        workspace.RunOperatorOnce(op)
        X_ideep = workspace.FetchBlob("X")
        np.testing.assert_allclose(X, X_ideep)

    def test_copy_from_ideep_zero_dim(self):
        op = core.CreateOperator(
            "CopyIDEEPToCPU",
            ["X_ideep"],
            ["X"],
        )
        op.device_option.CopyFrom(self._get_deep_device())
        n = 0
        c = randint(1, 64)
        X = np.random.rand(n, c).astype(np.float32)
        workspace.FeedBlob("X_ideep", X, self._get_deep_device())
        workspace.RunOperatorOnce(op)
        X_ideep = workspace.FetchBlob("X")
        np.testing.assert_allclose(X, X_ideep)

    def test_copy_from_ideep_fallthrough(self):
        op = core.CreateOperator(
            "CopyIDEEPToCPU",
            ["X_ideep"],
            ["X"],)
        op.device_option.CopyFrom(self._get_deep_device())
        n = randint(1, 128)
        c = randint(1, 64)
        h = randint(1, 128)
        w = randint(1, 128)
        X = np.random.rand(n, c, h, w).astype(np.float32)
        workspace.FeedBlob("X_ideep", X)
        workspace.RunOperatorOnce(op)
        X_ideep = workspace.FetchBlob("X")
        np.testing.assert_allclose(X, X_ideep)

if __name__ == "__main__":
    unittest.main()
