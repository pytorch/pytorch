## @package expand_dims
# Module caffe2.python.layers.expand_dims
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
)


class ExpandDims(ModelLayer):

    def __init__(self, model, input_record, dims,
                 name='expand_dims', **kwargs):
        super(ExpandDims, self).__init__(model, name, input_record, **kwargs)
        self.dims = dims
        # Assume that first dimension is batch, so actual dims[i] in shape is
        # dims[i] - 1
        dims = [d - 1 for d in dims]
        assert all([d >= 0 for d in dims])
        assert isinstance(input_record, schema.Scalar),\
            "Incorrect input type. Excpected Scalar, but received: {0}".\
            format(input_record)

        input_dims = list(input_record.field_type().shape)
        dims = sorted(set(dims))
        assert len(input_dims) + len(dims) >= dims[-1] + 1

        output_dims = input_dims[:]
        for dim in dims:
            output_dims.insert(dim, 1)

        self.output_schema = schema.Scalar(
            (input_record.field_type().base, output_dims),
            model.net.NextScopedBlob(name + '_output'))

    def add_ops(self, net):
        net.ExpandDims(
            self.input_record(),
            self.output_schema(),
            dims=self.dims,
        )
