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
from caffe2.python.text_file_reader import TextFileReader
from caffe2.python.test_util import TestCase
from caffe2.python.schema import Struct, Scalar, FetchRecord
import tempfile
import numpy as np


class TestTextFileReader(TestCase):
    def test_text_file_reader(self):
        schema = Struct(
            ('field1', Scalar(dtype=str)),
            ('field2', Scalar(dtype=str)),
            ('field3', Scalar(dtype=np.float32)))
        num_fields = 3
        col_data = [
            ['l1f1', 'l2f1', 'l3f1', 'l4f1'],
            ['l1f2', 'l2f2', 'l3f2', 'l4f2'],
            [0.456, 0.789, 0.10101, -24342.64],
        ]
        row_data = list(zip(*col_data))
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as txt_file:
            txt_file.write(
                '\n'.join(
                    '\t'.join(str(x) for x in f)
                    for f in row_data
                ) + '\n'
            )
            txt_file.flush()

            for num_passes in range(1, 3):
                for batch_size in range(1, len(row_data) + 2):
                    init_net = core.Net('init_net')
                    reader = TextFileReader(
                        init_net,
                        filename=txt_file.name,
                        schema=schema,
                        batch_size=batch_size,
                        num_passes=num_passes)
                    workspace.RunNetOnce(init_net)

                    net = core.Net('read_net')
                    should_stop, record = reader.read_record(net)

                    results = [np.array([])] * num_fields
                    while True:
                        workspace.RunNetOnce(net)
                        arrays = FetchRecord(record).field_blobs()
                        for i in range(num_fields):
                            results[i] = np.append(results[i], arrays[i])
                        if workspace.FetchBlob(should_stop):
                            break
                    for i in range(num_fields):
                        col_batch = np.tile(col_data[i], num_passes)
                        if col_batch.dtype in (np.float32, np.float64):
                            np.testing.assert_array_almost_equal(
                                col_batch, results[i], decimal=3)
                        else:
                            np.testing.assert_array_equal(col_batch, results[i])

if __name__ == "__main__":
    import unittest
    unittest.main()
