from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import os
import tempfile
import shutil

from caffe2.distributed.python import StoreHandlerTimeoutError
from caffe2.distributed.store_ops_test_util import StoreOpsTests
from caffe2.python import core, workspace, dyndep
from caffe2.python.test_util import TestCase

dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:file_store_handler_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:store_ops")


class TestFileStoreHandlerOp(TestCase):
    testCounter = 0

    def setUp(self):
        super(TestFileStoreHandlerOp, self).setUp()
        self.tmpdir = tempfile.mkdtemp()

        # Use counter to tell test cases apart
        TestFileStoreHandlerOp.testCounter += 1

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        super(TestFileStoreHandlerOp, self).tearDown()

    def create_store_handler(self):
        # Use new path for every test so they are isolated
        path = self.tmpdir + "/" + str(TestFileStoreHandlerOp.testCounter)

        # Ensure path exists (including counter)
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

        store_handler = "store_handler"
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "FileStoreHandlerCreate",
                [],
                [store_handler],
                path=path))

        return store_handler

    def test_set_get(self):
        StoreOpsTests.test_set_get(self.create_store_handler)

    def test_get_timeout(self):
        with self.assertRaises(StoreHandlerTimeoutError):
            StoreOpsTests.test_get_timeout(self.create_store_handler)
