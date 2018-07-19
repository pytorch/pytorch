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

import logging
logger = logging.getLogger(__name__)

class Concat(ModelLayer):
    """
    Construct Concat layer
    Assume that first dimension is batch,

    Example:

        embedding_dim = 64
        input_record = self.new_record(schema.Struct(
            ('input1', schema.Scalar((np.float32, (embedding_dim, )))),
            ('input2', schema.Scalar((np.float32, (embedding_dim, )))),
            ('input3', schema.Scalar((np.float32, (embedding_dim, )))),
        ))

        output = self.model.Concat(input_record)
        self.assertEqual(
            schema.Scalar((np.float32, ((len(input_record.fields) * embedding_dim, )))),
            output
        )

        # Note that in Concat layer we assume first dimension is batch.
        # so input is B * embedding_dim
        # add_axis=1 make it B * 1 * embedding_dim
        # Concat on axis=1 make it B * N * embedding_dim

        output = self.model.Concat(input_record, axis=1, add_axis=1)
        self.assertEqual(
            schema.Scalar((np.float32, ((len(input_record.fields), embedding_dim)))),
            output
        )
    """

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
            shape = list(field_type.field_type().shape)
            if add_axis:
                shape.insert(axis - 1, 1)
            assert len(shape) >= axis,\
                "Concat expects that limited dimensions of the input tensor"
            shapes.append(shape)
        logger.info('Concat Layer input shapes: ' + str(shapes))

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

        logger.info('Concat Layer output_dims: ' + str(output_dims))
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
