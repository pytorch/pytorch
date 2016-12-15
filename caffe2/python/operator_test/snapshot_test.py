from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
import os
import shutil
import tempfile
import unittest


class SnapshotTest(unittest.TestCase):
    """A simple test case to make sure that the snapshot behavior is correct.
    """

    def testSnapshot(self):
        temp_root = tempfile.mkdtemp()
        net = core.Net("test_snapshot")
        # Note(jiayq): I am being a bit lazy here and am using the old iter
        # convention that does not have an input. Optionally change it to the
        # new style if needed.
        net.Iter([], "iter")
        net.ConstantFill([], "value", shape=[1, 2, 3])
        net.Snapshot(["iter", "value"], [],
                     db=os.path.join(temp_root, "test_snapshot_at_%05d"),
                     db_type="leveldb", every=10, absolute_path=True)
        self.assertTrue(workspace.CreateNet(net))
        for i in range(100):
            self.assertTrue(workspace.RunNet("test_snapshot"))
        for i in range(1, 10):
            # Print statements are only for debugging purposes.
            # print("Asserting %d" % i)
            # print(os.path.join(temp_root, "test_snapshot_at_%05d" % (i * 10)))
            self.assertTrue(os.path.exists(
                os.path.join(temp_root, "test_snapshot_at_%05d" % (i * 10))))

        # Finally, clean up.
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    import unittest
    unittest.main()
