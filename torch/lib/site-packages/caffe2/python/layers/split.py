## @package split
# Module caffe2.python.layers.split
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)


class Split(ModelLayer):

    def __init__(self, model, input_record, num_splits, axis=1,
                 name='split', **kwargs):
        super(Split, self).__init__(model, name, input_record, **kwargs)
        self.axis = axis
        # Assume that first dimension is batch, so actual axis in shape is
        # axis - 1
        axis -= 1
        assert axis >= 0

        assert isinstance(input_record, schema.Scalar),\
            "Incorrect input type. Excpected Scalar, but received: {0}".\
            format(input_record)

        input_shape = input_record.field_type().shape
        assert len(input_shape) >= axis
        assert input_shape[axis] % num_splits == 0

        output_shape = list(input_shape)
        output_shape[axis] = int(output_shape[axis] / num_splits)

        data_type = input_record.field_type().base

        output_scalars = [
            schema.Scalar(
                (data_type, output_shape),
                self.get_next_blob_reference('output_{}'.format(i)),
            )
            for i in range(num_splits)
        ]
        self.output_schema = schema.Tuple(*output_scalars)

    def add_ops(self, net):
        net.Split(
            self.input_record.field_blobs(),
            self.output_schema.field_blobs(),
            axis=self.axis,
        )
