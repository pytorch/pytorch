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

from caffe2.python import core, schema
from caffe2.python.modeling.net_modifier import NetModifier

import numpy as np


class ComputeStatisticsForBlobs(NetModifier):
    """
    This class modifies the net passed in by adding ops to compute statistics
    for certain blobs. For each blob in the list, its min, max, mean and standard
    deviation will be computed.

    Args:
        blobs: list of blobs to compute norm for
        logging_frequency: frequency for printing norms to logs
    """

    def __init__(self, blobs, logging_frequency):
        self._blobs = blobs
        self._logging_frequency = logging_frequency

    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None):

        for blob_name in self._blobs:
            blob = core.BlobReference(blob_name)
            if not net.BlobIsDefined(blob):
                raise Exception('blob {0} is not defined in net {1}'.format(
                    blob, net.Name()))

            cast_blob = net.Cast(blob, to=core.DataType.FLOAT)
            stats_name = net.NextScopedBlob(prefix=blob + '_summary')
            stats = net.Summarize(cast_blob, stats_name, to_file=0)
            net.Print(stats, [], every_n=self._logging_frequency)

            output_field_name = str(blob) + '_summary'
            output_scalar = schema.Scalar((np.float, (1,)), stats)

            if net.output_record() is None:
                net.set_output_record(
                    schema.Struct((output_field_name, output_scalar))
                )
            else:
                net.AppendOutputRecordField(
                    output_field_name,
                    output_scalar)
