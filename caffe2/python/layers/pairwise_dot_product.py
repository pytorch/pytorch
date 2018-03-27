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


class PairwiseDotProduct(ModelLayer):

    def __init__(self, model, input_record, output_dim,
                 name='pairwise_dot_product', **kwargs):
        super(PairwiseDotProduct, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Struct), (
            "Incorrect input type. Excpected Struct, but received: {0}".
            format(input_record))
        assert (
            ('all_embeddings' in input_record) ^
            ('x_embeddings' in input_record and 'y_embeddings' in input_record)
        ), (
            "either (all_embeddings) xor (x_embeddings and y_embeddings) " +
            "should be given."
        )
        if 'all_embeddings' in input_record:
            x_embeddings = input_record['all_embeddings']
            y_embeddings = input_record['all_embeddings']
        else:
            x_embeddings = input_record['x_embeddings']
            y_embeddings = input_record['y_embeddings']

        assert isinstance(x_embeddings, schema.Scalar), (
            "Incorrect input type for x. Expected Scalar, " +
            "but received: {0}".format(x_embeddings))
        assert isinstance(y_embeddings, schema.Scalar), (
            "Incorrect input type for y. Expected Scalar, " +
            "but received: {0}".format(y_embeddings)
        )

        if 'indices_to_gather' in input_record:
            indices_to_gather = input_record['indices_to_gather']
            assert isinstance(indices_to_gather, schema.Scalar), (
                "Incorrect type of indices_to_gather. "
                "Expected Scalar, but received: {0}".format(indices_to_gather)
            )
            self.indices_to_gather = indices_to_gather
        else:
            self.indices_to_gather = None

        self.x_embeddings = x_embeddings
        self.y_embeddings = y_embeddings

        dtype = x_embeddings.field_types()[0].base

        self.output_schema = schema.Scalar(
            (dtype, (output_dim,)),
            self.get_next_blob_reference('output')
        )

    def add_ops(self, net):
        Y = net.BatchMatMul(
            [self.x_embeddings(), self.y_embeddings()],
            [self.x_embeddings() + '_matmul'],
            trans_b=1,
        )

        if self.indices_to_gather:
            flattened = net.Flatten(
                Y, Y + '_flatten',
            )
            net.BatchGather(
                [flattened, self.indices_to_gather()],
                self.output_schema(),
            )
        else:
            net.Flatten(Y, self.output_schema())
