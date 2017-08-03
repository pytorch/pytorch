## @package concat
# Module caffe2.python.layers.concat
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
from future.utils import viewitems
import numpy as np


class Concat(ModelLayer):

    def __init__(self, model, input_record, axis=1, add_axis=0,
                 name='concat', **kwargs):
        super(Concat, self).__init__(model, name, input_record, **kwargs)
        self.axis = axis
        self.add_axis = add_axis
        assert not (axis == 0 and add_axis == 1), \
            "It's not allowed to add axis=0"
        assert isinstance(input_record, schema.Struct),\
            "Incorrect input type. Excpected Struct, but received: {0}".\
            format(input_record)

        shapes = []
        for field_name, field_type in viewitems(input_record.fields):
            assert isinstance(field_type, schema.Scalar),\
                "Incorrect input type for {}. Excpected Scalar, but got: {}".\
                format(field_name, field_type)
            # Assume that first dimension is batch, so actual axis in shape is
            # axis - 1
            assert len(field_type.field_type().shape) >= axis,\
                "Concat expects that limited dimensions of the input tensor"
            shapes.append(list(field_type.field_type().shape))

        if add_axis:
            for i in range(len(shapes)):
                shapes[i].insert(axis, 1)

        if axis == 0:
            self.output_schema = schema.from_blob_list(
                input_record[0],
                [self.get_next_blob_reference('output')]
            )
            return

        concat_dim = 0
        for shape in shapes:
            concat_dim += shape[axis - 1]
            shape[axis - 1] = 0
            assert shape == shapes[0],\
                "Shapes {0} and {1} are not compatible for Concat".\
                format(shape, shapes[0])
        output_dims = shapes[0]
        output_dims[axis - 1] = concat_dim

        self.output_schema = schema.Scalar(
            (np.float32, output_dims),
            self.get_next_blob_reference('output'))

    def add_ops(self, net):
        net.Concat(
            self.input_record.field_blobs(),
            [
                self.output_schema.field_blobs()[0],
                self.output_schema.field_blobs()[0] + "_concat_dims"
            ],
            axis=self.axis,
            add_axis=self.add_axis,
        )
