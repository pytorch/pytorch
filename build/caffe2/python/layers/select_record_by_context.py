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


logger = logging.getLogger(__name__)


class SelectRecordByContext(ModelLayer):
    """
    Allowing model to follow different paths for each instatiation context and
    join later at some point. The implementation use `Alias` because schema
    sometimes clone fields internally so we need static blob name for output
    """

    def __init__(
        self,
        model,
        input_record,
        name='select_record_by_context',
        check_field_metas=True,
        use_copy=False,
        default_output_record_field=None,
        **kwargs
    ):
        super(SelectRecordByContext, self).__init__(model, name, input_record,
                                                    **kwargs)

        assert isinstance(input_record, schema.Struct)
        assert len(input_record) > 1

        self.use_copy = use_copy
        self.default_output_record = (
            input_record[default_output_record_field]
            if (default_output_record_field is not None) else None
        )
        ref_record = input_record[0]
        for record in input_record:
            assert schema.equal_schemas(record, ref_record,
                                        check_field_metas=check_field_metas)

        self.output_schema = schema.NewRecord(model.net, ref_record)

    def _set_output_blobs(self, net, context):
        record = self.input_record.get(context, self.default_output_record)
        assert record is not None, (
            "{} context is not in input record without providing default"
            " output".format(context)
        )
        for in_blob, out_blob in zip(
                record.field_blobs(), self.output_schema.field_blobs()
        ):
            if self.use_copy:
                net.Copy(in_blob, out_blob)
            else:
                net.Alias(in_blob, out_blob)

    def add_ops(self, net):
        self._set_output_blobs(net, InstantiationContext.PREDICTION)

    def add_eval_ops(self, net):
        self._set_output_blobs(net, InstantiationContext.EVAL)

    def add_train_ops(self, net):
        self._set_output_blobs(net, InstantiationContext.TRAINING)

    def add_ops_to_accumulate_pred(self, net):
        self._set_output_blobs(net, InstantiationContext.ACCUMULATE_PRED)
