## @package simple_operator_layers
# Module caffe2.python.layers.simple_operator_layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)


def simple_init(self, model, input_record, *args, **kwargs):
    ModelLayer.__init__(self, model, self.operator, input_record, **kwargs)
    assert self.operator is not None, "Try to create invalid operator layer"
    self.args = args
    self.output_schema = schema.NewRecord(self.model.net, input_record)


def first_field_schema_init(self, model, input_record, *args, **kwargs):
    ModelLayer.__init__(self, model, self.operator, input_record, **kwargs)
    assert self.operator is not None, "Try to create invalid operator layer"
    assert isinstance(input_record, schema.Struct),\
        "Operator {0} expects schema.Struct as input, received {1} instead".\
        format(self.operator, input_record)
    self.args = args
    self.output_schema = schema.NewRecord(self.model.net, input_record[0])


def simple_add_ops(self, net):
    getattr(
        net,
        self.operator)(
        self.input_record.field_blobs(),
        self.output_schema.field_blobs(),
        *self.args,
        **self.kwargs
    )

_simple_operators = ['Softmax', 'Relu', 'Sigmoid', 'Tanh']
_first_field_schema_operators = ['Sum']

# We need to store refs for all created types, to make sure that they won't be
# GCed before we actually register them.
_known_layers = []

for operator in _simple_operators:
    # Generate class instance with name 'operator', that is doing going to use
    # simple_init and simple_add_ops implementations for __init__ and add_ops
    # calls. It'll also get automatically registered in the registry.
    _known_layers.append(
        type(
            str(operator),
            (ModelLayer,),
            {'__init__': simple_init,
             'add_ops': simple_add_ops,
             'operator': operator
             }
        )
    )

for operator in _first_field_schema_operators:
    # Generate class instance with name 'operator', that is doing going to use
    # first_field_schema_init and simple_add_ops implementations for __init__
    # and add_ops calls. It'll also get automatically registered in the
    # registry.
    _known_layers.append(
        type(
            str(operator),
            (ModelLayer,),
            {'__init__': first_field_schema_init,
             'add_ops': simple_add_ops,
             'operator': operator
             }
        )
    )
