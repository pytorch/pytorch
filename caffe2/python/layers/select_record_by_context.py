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

import logging

from caffe2.python import schema
from caffe2.python.layers.layers import (
    InstantiationContext,
    ModelLayer,
)


logging.basicConfig()
logger = logging.getLogger(__name__)


class SelectRecordByContext(ModelLayer):
    """
    Allowing model to follow different paths for each instatiation context and
    join later at some point. The implementation use `Alias` because schema
    sometimes clone fields internally so we need static blob name for output
    """

    def __init__(self, model, input_record, name='select_record_by_context',
                 check_field_metas=True, **kwargs):
        super(SelectRecordByContext, self).__init__(model, name, input_record,
                                                    **kwargs)

        assert isinstance(input_record, schema.Struct)
        assert len(input_record) > 1

        ref_record = input_record[0]
        for record in input_record:
            assert schema.equal_schemas(record, ref_record,
                                        check_field_metas=check_field_metas)

        self.output_schema = schema.NewRecord(model.net, ref_record)

    def _set_output_blobs(self, net, context):
        assert context in self.input_record, (
            "{} context is not in input record".format(context)
        )
        record = self.input_record[context]

        for in_blob, out_blob in zip(
                record.field_blobs(), self.output_schema.field_blobs()
        ):
            net.Alias(in_blob, out_blob)

    def add_ops(self, net):
        self._set_output_blobs(net, InstantiationContext.PREDICTION)

    def add_eval_ops(self, net):
        self._set_output_blobs(net, InstantiationContext.EVAL)

    def add_train_ops(self, net):
        self._set_output_blobs(net, InstantiationContext.TRAINING)

    def add_ops_to_accumulate_pred(self, net):
        self._set_output_blobs(net, InstantiationContext.ACCUMULATE_PRED)
