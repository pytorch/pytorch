## @package dot_product
# Module caffe2.python.layers.dot_product
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
from future.utils import viewitems


class DotProduct(ModelLayer):

    def __init__(self, model, input_record, name='dot_product', **kwargs):
        super(DotProduct, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Struct),\
            "Incorrect input type. Excpected Struct, but received: {0}".\
            format(input_record)
        assert len(input_record.get_children()) == 2, (
            "DotProduct accept 2 inputs")
        assert len(set(input_record.field_types())) == 1, (
            "Inputs should be of the same field type")

        for field_name, field_type in viewitems(input_record.fields):
            assert isinstance(field_type, schema.Scalar),\
                "Incorrect input type for {}. Excpected Scalar, but got: {}".\
                format(field_name, field_type)

        self.output_schema = schema.Scalar(
            (input_record.field_types()[0].base, ()),
            self.get_next_blob_reference('output'))

    def add_ops(self, net):
        net.DotProduct(
            self.input_record.field_blobs(),
            self.output_schema(),
        )
