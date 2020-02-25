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


class PairwiseSimilarity(ModelLayer):

    def __init__(self, model, input_record, output_dim, pairwise_similarity_func='dot',
                 name='pairwise_similarity', **kwargs):
        super(PairwiseSimilarity, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Struct), (
            "Incorrect input type. Expected Struct, but received: {0}".
            format(input_record))
        assert (
            ('all_embeddings' in input_record) ^
            ('x_embeddings' in input_record and 'y_embeddings' in input_record)
        ), (
            "either (all_embeddings) xor (x_embeddings and y_embeddings) " +
            "should be given."
        )
        self.pairwise_similarity_func = pairwise_similarity_func
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
        n = x_embeddings.field_type().shape[0]
        self.output_schema = schema.Scalar(
            (dtype, (output_dim,)),
            self.get_next_blob_reference('output')
        )

        if self.pairwise_similarity_func == "mahalanobis":
            assert 'inv_cov' in input_record, "inverse covariance expected for mahalanobis"
            inv_cov = input_record['inv_cov']
            assert 'x_diag' in input_record, "x embeddings diag indexes expected for mahalanobis"
            x_diag = input_record['x_diag']
            assert 'y_diag' in input_record, "y embeddings diag indexes expected for mahalanobis"
            y_diag = input_record['y_diag']
        else:
            inv_cov = None
            x_diag = None
            y_diag = None

        self.inv_cov = inv_cov
        self.x_diag = x_diag
        self.y_diag = y_diag
        self.output_dim = output_dim

        self.my_const = self.model.maybe_add_global_constant('MAHALANOBIS_CONST_TERM', -2.0)

    def _diag(self, net, X, diag_indexes):
        x2 = net.FlattenToVec(X)
        x3 = net.ExpandDims(x2, dims=[1])
        x4 = net.EnsureDense(x3)
        x5 = net.Gather([x4, diag_indexes])  # (n, 1)
        return x5

    def add_ops(self, net):
        if self.pairwise_similarity_func == "cosine_similarity":
            x_embeddings_norm = net.Normalize(self.x_embeddings(), axis=1)
            y_embeddings_norm = net.Normalize(self.y_embeddings(), axis=1)
            Y = net.BatchMatMul(
                [x_embeddings_norm, y_embeddings_norm],
                [self.get_next_blob_reference(x_embeddings_norm + '_matmul')],
                trans_b=1,
            )
        elif self.pairwise_similarity_func == "dot":
            Y = net.BatchMatMul(
                [self.x_embeddings(), self.y_embeddings()],
                [self.get_next_blob_reference(self.x_embeddings() + '_matmul')],
                trans_b=1,
            )
        elif self.pairwise_similarity_func == "mahalanobis":
            x_inv_cov = net.BatchMatMul(
                [self.x_embeddings(), self.inv_cov()]
            )
            y_inv_cov = net.BatchMatMul(
                [self.y_embeddings(), self.inv_cov()]
            )
            x_inv_cov_xT = net.BatchMatMul([x_inv_cov, self.x_embeddings()], trans_b=1)
            y_inv_cov_yT = net.BatchMatMul([y_inv_cov, self.y_embeddings()], trans_b=1)
            x_inv_cov_yT = net.BatchMatMul([x_inv_cov, self.y_embeddings()], trans_b=1)
            diag_xx = self._diag(net, x_inv_cov_xT, self.x_diag())
            diag_yy = self._diag(net, y_inv_cov_yT, self.y_diag())
            diag_yyT = net.Transpose(diag_yy, axes=(1, 0))
            xy = net.Mul([self.my_const, x_inv_cov_yT])
            d_xy = net.Add([xy, diag_xx], broadcast=1, axes=1)
            d_xy_d = net.Add([d_xy, diag_yyT], broadcast=1, axes=0)
            Y = net.Sqrt(d_xy_d)
        else:
            raise NotImplementedError(
                "pairwise_similarity_func={} is not valid".format(
                    self.pairwise_similarity_func
                )
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
