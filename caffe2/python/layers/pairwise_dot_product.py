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
        assert 'all_embeddings' in input_record, "all_embeddings is not given."
        all_embeddings = input_record['all_embeddings']
        assert isinstance(all_embeddings, schema.Scalar), (
            "Incorrect input type. Excpected Scalar, but received: {0}".
            format(all_embeddings))
        if 'indices_to_gather' in input_record:
            indices_to_gather = input_record['indices_to_gather']
            assert isinstance(indices_to_gather, schema.Scalar), (
                "Incorrect type of indices_to_gather. "
                "Expected Scalar, but received: {0}".format(indices_to_gather)
            )

        self.all_embeddings = all_embeddings
        self.indices_to_gather = indices_to_gather

        dtype = all_embeddings.field_types()[0].base
        self.output_schema = schema.Scalar(
            (dtype, (output_dim)),
            self.get_next_blob_reference('output')
        )

    def add_ops(self, net):
        Y = net.BatchMatMul(
            [self.all_embeddings(), self.all_embeddings()],
            [self.all_embeddings() + '_matmul'],
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
