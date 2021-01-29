




import os
import uuid

from caffe2.distributed.python import StoreHandlerTimeoutError  # type: ignore[import]
from caffe2.distributed.store_ops_test_util import StoreOpsTests
from caffe2.python import core, workspace, dyndep
from caffe2.python.test_util import TestCase

dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:redis_store_handler_ops")
dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:store_ops")


class TestRedisStoreHandlerOp(TestCase):
    def setUp(self):
        super(TestRedisStoreHandlerOp, self).setUp()
        self.uuid = str(uuid.uuid4()) + "/"

    def tearDown(self):
        super(TestRedisStoreHandlerOp, self).tearDown()

    def create_store_handler(self):
        store_handler = "store_handler"
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "RedisStoreHandlerCreate",
                [],
                [store_handler],
                prefix=self.uuid,
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379))))
        return store_handler

    def test_set_get(self):
        StoreOpsTests.test_set_get(self.create_store_handler)

    def test_get_timeout(self):
        with self.assertRaises(StoreHandlerTimeoutError):
            StoreOpsTests.test_get_timeout(self.create_store_handler)
