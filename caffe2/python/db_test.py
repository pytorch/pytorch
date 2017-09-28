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

from caffe2.python import workspace

import os
import tempfile
import unittest


class TestDB(unittest.TestCase):
    def setUp(self):
        handle, self.file_name = tempfile.mkstemp()
        os.close(handle)
        self.data = [
            (
                "key{}".format(i).encode("ascii"),
                "value{}".format(i).encode("ascii")
            )
            for i in range(1, 10)
        ]

    def testSimple(self):
        db = workspace.C.create_db(
            "minidb", self.file_name, workspace.C.Mode.write)

        for key, value in self.data:
            transaction = db.new_transaction()
            transaction.put(key, value)
            del transaction

        del db  # should close DB

        db = workspace.C.create_db(
            "minidb", self.file_name, workspace.C.Mode.read)
        cursor = db.new_cursor()
        data = []
        while cursor.valid():
            data.append((cursor.key(), cursor.value()))
            cursor.next()  # noqa: B305
        del cursor

        db.close()  # test explicit db closer
        self.assertEqual(data, self.data)
