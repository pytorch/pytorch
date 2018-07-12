## @package BlobWeightedSum
# Module caffe2.python.layers.blob_weighted_sum
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer


class BlobWeightedSum(ModelLayer):
    """
    This layer implements the weighted sum:
    weighted element-wise sum of input blobs.
    """
    def __init__(
        self,
        model,
        input_record,
        init_weights=None,
        weight_optim=None,
        name='blob_weighted_sum',
        **kwargs
    ):
        super(BlobWeightedSum, self).__init__(model, name, input_record, **kwargs)

        self.blobs = self.input_record.field_blobs()

        self.num_weights = len(self.blobs)
        assert self.num_weights > 1, (
            "BlobWeightedSum expects more than one input blobs"
        )

        assert len(input_record.field_types()[0].shape) > 0, (
            "BlobWeightedSum expects limited dimensions of the input tensor"
        )

        assert all(
            input_record.field_types()[0].shape == input_record.field_types()[i].shape
            for i in range(1, self.num_weights)
        ), "Shape of input blobs should be the same shape {}".format(
            input_record.field_types()[0].shape
        )

        if init_weights:
            assert self.num_weights == len(init_weights), (
                "the size of init_weights should be the same as input blobs, "
                "expects {}, got {}".format(self.num_weights, len(init_weights))
            )
        else:
            init_weights = [1.0] * self.num_weights

        self.weights = [
            self.create_param(
                param_name="w_{}".format(idx),
                shape=[1],
                initializer=('ConstantFill', {'value': float(init_weights[idx])}),
                optimizer=weight_optim
            ) for idx in range(self.num_weights)
        ]

        self.output_schema = schema.Scalar(
            input_record.field_types()[0],
            self.get_next_blob_reference('blob_weighted_sum_out')
        )

    def add_ops(self, net):
        net.WeightedSum(
            [x for pair in zip(self.blobs, self.weights) for x in pair],
            self.output_schema(),
            grad_on_w=True,
        )
