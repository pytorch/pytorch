## @package bucket_weighted
# Module caffe2.python.layers.bucket_weighted
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    get_categorical_limit,
    ModelLayer,
)

from caffe2.python.layers.tags import Tags

logger = logging.getLogger(__name__)


class BucketWeighted(ModelLayer):
    def __init__(self, model, input_record, max_score=0, bucket_boundaries=None,
                 hash_buckets=True, weight_optim=None, name="bucket_weighted"):
        super(BucketWeighted, self).__init__(model, name, input_record)

        assert isinstance(input_record, schema.List), "Incorrect input type"
        self.bucket_boundaries = bucket_boundaries
        self.hash_buckets = hash_buckets
        if bucket_boundaries is not None:
            self.shape = len(bucket_boundaries) + 1
        elif max_score > 0:
            self.shape = max_score
        else:
            self.shape = get_categorical_limit(input_record)

        self.bucket_w = self.create_param(param_name='bucket_w',
                                       shape=[self.shape, ],
                                       initializer=('ConstantFill', {'value': 1.0}),
                                       optimizer=weight_optim)

        self.output_schema = schema.Struct(
            ('bucket_weights',
                schema.Scalar((np.float32, self.shape),
                              self.get_next_blob_reference("bucket_w_gather")))
        )

        self.tags.update({Tags.HANDLE_AS_SPARSE_LAYER})

    def get_memory_usage(self):
        return self.shape

    def add_ops(self, net):
        if self.bucket_boundaries is not None:
            buckets_int = net.Bucketize(
                self.input_record.values(),
                "buckets_int",
                boundaries=self.bucket_boundaries
            )
        else:
            buckets = self.input_record.values()
            buckets_int = net.Cast(
                buckets,
                "buckets_int",
                to=core.DataType.INT32
            )
        if self.hash_buckets:
            buckets_int = net.IndexHash(
                buckets_int, "hashed_buckets_int", seed=0, modulo=self.shape
            )
        net.Gather(
            [self.bucket_w, buckets_int],
            self.output_schema.bucket_weights.field_blobs())
