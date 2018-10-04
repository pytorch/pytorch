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

import unittest
from caffe2.python import workspace, brew, model_helper
from caffe2.python.modeling.get_entry_from_blobs import GetEntryFromBlobs

import numpy as np


class GetEntryFromBlobsTest(unittest.TestCase):
    def test_get_entry_from_blobs(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=10, dim_out=8)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=8, dim_out=4)
        i1, i2 = np.random.randint(4, size=2)
        net_modifier = GetEntryFromBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=10,
            i1=i1,
            i2=i2,
        )
        net_modifier(model.net)

        workspace.FeedBlob('data', np.random.rand(10, 10).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        fc1_w_entry = workspace.FetchBlob('fc1_w_{0}_{1}'.format(i1, i2))

        self.assertEqual(fc1_w_entry.size, 1)
        self.assertEqual(fc1_w_entry[0], fc1_w[i1][i2])
        assert model.net.output_record() is None

    def test_get_entry_from_blobs_modify_output_record(self):
        model = model_helper.ModelHelper(name="test")
        data = model.net.AddExternalInput("data")
        fc1 = brew.fc(model, data, "fc1", dim_in=4, dim_out=4)

        # no operator name set, will use default
        brew.fc(model, fc1, "fc2", dim_in=4, dim_out=4)
        i1, i2 = np.random.randint(4), np.random.randint(5) - 1
        net_modifier = GetEntryFromBlobs(
            blobs=['fc1_w', 'fc2_w'],
            logging_frequency=10,
            i1=i1,
            i2=i2,
        )
        net_modifier(model.net, modify_output_record=True)

        workspace.FeedBlob('data', np.random.rand(10, 4).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        fc1_w = workspace.FetchBlob('fc1_w')
        if i2 < 0:
            fc1_w_entry = workspace.FetchBlob('fc1_w_{0}_all'.format(i1))
        else:
            fc1_w_entry = workspace.FetchBlob('fc1_w_{0}_{1}'.format(i1, i2))

        if i2 < 0:
            self.assertEqual(fc1_w_entry.size, 4)
            for j in range(4):
                self.assertEqual(fc1_w_entry[0][j], fc1_w[i1][j])
        else:
            self.assertEqual(fc1_w_entry.size, 1)
            self.assertEqual(fc1_w_entry[0], fc1_w[i1][i2])

        assert 'fc1_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
        assert 'fc2_w' + net_modifier.field_name_suffix() in\
            model.net.output_record().field_blobs(),\
            model.net.output_record().field_blobs()
