



from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np
import unittest


class DoOpTest(TestCase):
    def test_operator(self):
        def make_net():
            subnet = core.Net('subnet')
            subnet.Add(["X", "Y"], "Z")

            net = core.Net("net")
            net.CreateScope([], "W")

            net.Do(
                ["outer_X", "outer_Y", "W"],
                ["outer_Z", "W"],
                net=subnet.Proto(),
                inner_blobs=["X", "Y", "Z"],
                outer_blobs_idx=[0, 1, 2],
            )

            return net

        net = make_net()

        workspace.ResetWorkspace()
        workspace.FeedBlob("outer_X", np.asarray([1, 2]))
        workspace.FeedBlob("outer_Y", np.asarray([3, 4]))

        workspace.RunNetOnce(net)
        outer_Z_val = workspace.FetchBlob("outer_Z")
        self.assertTrue(np.all(outer_Z_val == np.asarray([4, 6])))

    def test_reuse_workspace(self):
        def make_net():
            param_init_subnet = core.Net('param_init_subnet')
            param_init_subnet.ConstantFill([], "X", shape=[1], value=1)
            param_init_subnet.ConstantFill([], "Y", shape=[1], value=2)

            subnet = core.Net("subnet")
            subnet.Add(["X", "Y"], "Z")

            net = core.Net("net")
            net.CreateScope([], "W")
            net.Do(
                "W", "W",
                net=param_init_subnet.Proto(),
                inner_blobs=[],
                outer_blobs_idx=[],
            )

            net.Do(
                "W", ["outer_Z", "W"],
                net=subnet.Proto(),
                inner_blobs=["Z"],
                outer_blobs_idx=[0],
                reuse_workspace=True,
            )

            return net

        net = make_net()

        workspace.ResetWorkspace()
        workspace.RunNetOnce(net)
        outer_Z_val = workspace.FetchBlob("outer_Z")
        self.assertTrue(np.all(outer_Z_val == np.asarray([3])))


if __name__ == '__main__':
    unittest.main()
