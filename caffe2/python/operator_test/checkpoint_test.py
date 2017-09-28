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

from caffe2.python import core, workspace
import os
import shutil
import tempfile
import unittest


class CheckpointTest(unittest.TestCase):
    """A simple test case to make sure that the checkpoint behavior is correct.
    """

    def testCheckpoint(self):
        temp_root = tempfile.mkdtemp()
        net = core.Net("test_checkpoint")
        # Note(jiayq): I am being a bit lazy here and am using the old iter
        # convention that does not have an input. Optionally change it to the
        # new style if needed.
        net.Iter([], "iter")
        net.ConstantFill([], "value", shape=[1, 2, 3])
        net.Checkpoint(["iter", "value"], [],
                     db=os.path.join(temp_root, "test_checkpoint_at_%05d"),
                     db_type="leveldb", every=10, absolute_path=True)
        self.assertTrue(workspace.CreateNet(net))
        for i in range(100):
            self.assertTrue(workspace.RunNet("test_checkpoint"))
        for i in range(1, 10):
            # Print statements are only for debugging purposes.
            # print("Asserting %d" % i)
            # print(os.path.join(temp_root, "test_checkpoint_at_%05d" % (i * 10)))
            self.assertTrue(os.path.exists(
                os.path.join(temp_root, "test_checkpoint_at_%05d" % (i * 10))))

        # Finally, clean up.
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    import unittest
    unittest.main()
