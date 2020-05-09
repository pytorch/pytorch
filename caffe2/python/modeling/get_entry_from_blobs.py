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


class GetEntryFromBlobs(NetModifier):
    """
    This class modifies the net passed in by adding ops to get a certain entry
    from certain blobs.

    Args:
        blobs: list of blobs to get entry from
        logging_frequency: frequency for printing entry values to logs
        i1, i2: the first, second dimension of the blob. (currently, we assume
        the blobs to be 2-dimensional blobs). When i2 = -1, print all entries
        in blob[i1]
    """

    def __init__(self, blobs, logging_frequency, i1=0, i2=0):
        self._blobs = blobs
        self._logging_frequency = logging_frequency
        self._i1 = i1
        self._i2 = i2
        self._field_name_suffix = '_{0}_{1}'.format(i1, i2) if i2 >= 0 \
            else '_{0}_all'.format(i1)

    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None,
                    modify_output_record=False):

        i1, i2 = [self._i1, self._i2]
        if i1 < 0:
            raise ValueError('index is out of range')

        for blob_name in self._blobs:
            blob = core.BlobReference(blob_name)
            assert net.BlobIsDefined(blob), 'blob {} is not defined in net {} whose proto is {}'.format(blob, net.Name(), net.Proto())

            blob_i1 = net.Slice([blob], starts=[i1, 0], ends=[i1 + 1, -1])
            if self._i2 == -1:
                blob_i1_i2 = net.Copy([blob_i1],
                    [net.NextScopedBlob(prefix=blob + '_{0}_all'.format(i1))])
            else:
                blob_i1_i2 = net.Slice([blob_i1],
                    net.NextScopedBlob(prefix=blob + '_{0}_{1}'.format(i1, i2)),
                    starts=[0, i2], ends=[-1, i2 + 1])

            if self._logging_frequency >= 1:
                net.Print(blob_i1_i2, [], every_n=self._logging_frequency)

            if modify_output_record:
                output_field_name = str(blob) + self._field_name_suffix
                output_scalar = schema.Scalar((np.float), blob_i1_i2)

                if net.output_record() is None:
                    net.set_output_record(
                        schema.Struct((output_field_name, output_scalar))
                    )
                else:
                    net.AppendOutputRecordField(output_field_name, output_scalar)

    def field_name_suffix(self):
        return self._field_name_suffix
