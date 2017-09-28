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

import os
import uuid

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
