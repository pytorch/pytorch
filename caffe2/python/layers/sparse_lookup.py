## @package sparse_lookup
# Module caffe2.python.layers.sparse_lookup
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    get_categorical_limit,
    IdList,
    IdScoreList,
    LayerPsParam,
    ModelLayer,
)
import functools
import math
import numpy as np
import operator


class SparseLookup(ModelLayer):
    _supported_reducers = ['PositionWeighted', 'LogMeanExp', 'LogSumExp', 'Max',
                           'Mean', 'Sum', 'Sqrt', 'None']

    def __init__(self, model, input_record, inner_shape, reducer,
                 weight_init=None, weight_optim=None,
                 name='sparse_lookup', **kwargs):

        super(SparseLookup, self).__init__(model, name, input_record, **kwargs)

        if reducer == "PositionWeighted":
            self.external_weights = input_record.values()

        if isinstance(inner_shape, int):
            inner_shape = [inner_shape]
        assert isinstance(inner_shape, list) or isinstance(inner_shape, tuple),\
            "Unexpected type for inner_shape, expected list or tuple, got {0}".\
            format(type(inner_shape))

        # TODO Add some asserts about input type
        assert reducer in self._supported_reducers, "Unsupported reducer: {}".\
            format(reducer)
        self.reducer = reducer

        input_dim = get_categorical_limit(input_record)

        assert input_dim is not None, "Unbounded features are not supported"

        scale = math.sqrt(1.0 / input_dim)
        self.shape = [input_dim] + inner_shape
        self.weight_init = weight_init if weight_init else (
            'UniformFill', {'min': -scale, 'max': scale})

        if schema.equal_schemas(self.input_record, IdList):
            sparse_key = self.input_record.items()
        elif schema.equal_schemas(
                self.input_record,
                IdScoreList,
                check_field_types=False):
            sparse_key = self.input_record.keys()
        else:
            raise NotImplementedError()

        if self.input_record.lengths.metadata:
            avg_length = self.input_record.lengths.metadata.expected_value
        else:
            avg_length = None

        self.w = self.create_param(param_name='w',
                                   shape=self.shape,
                                   initializer=self.weight_init,
                                   optimizer=weight_optim,
                                   ps_param=LayerPsParam(
                                       sparse_key=sparse_key,
                                       average_length=avg_length
                                   ))

        self.output_schema = schema.Scalar(
            (np.float32, inner_shape),
            self.get_next_blob_reference('output'),
        )

    def get_memory_usage(self):
        return functools.reduce(operator.mul, self.shape) * 4

    def get_fp16_compatible_parameters(self):
        return [self.w]

    def add_ops(self, net):
        if schema.equal_schemas(self.input_record, IdList):
            if self.reducer in ['Sum', 'Mean']:
                net.__getattr__('SparseLengths' + self.reducer)(
                    [
                        self.w,
                        self.input_record.items(),
                        self.input_record.lengths(),
                    ],
                    self.output_schema.field_blobs(),
                    engine='fp16',
                )
            elif self.reducer == 'Sqrt':
                sqrt_weight = net.LengthsToWeights(
                    [self.input_record.lengths()],
                    [self.input_record.lengths() + '_sqrt'],
                    power=0.5,
                )
                net.SparseLengthsWeightedSum(
                    [
                        self.w,
                        sqrt_weight,
                        self.input_record.items(),
                        self.input_record.lengths(),
                    ],
                    self.output_schema.field_blobs(),
                    engine='fp16',
                )
            elif self.reducer == 'None':
                # Gather operator will gather the embedding for each id of
                # each IdScoreList.
                net.Gather(
                    [
                        self.w,
                        self.input_record.items(),
                    ],
                    self.output_schema.field_blobs(),
                    engine='fp16',
                )
            else:
                table_rows = net.Gather([self.w, self.input_record.items()])
                segment_ids = net.LengthsToSegmentIds(
                    self.input_record.lengths(),
                    self.input_record.lengths() + '_sid'),
                net.__getattr__('SortedSegmentRange' + self.reducer)(
                    [table_rows, segment_ids],
                    self.output_schema.field_blobs(),
                    engine='fp16',
                )
        elif schema.equal_schemas(
                self.input_record,
                IdScoreList,
                check_field_types=False):
            if self.reducer in ['Sum', 'Mean']:
                net.__getattr__('SparseLengthsWeighted' + self.reducer)(
                    [
                        self.w,
                        self.input_record.values(),
                        self.input_record.keys(),
                        self.input_record.lengths(),
                    ],
                    self.output_schema.field_blobs(),
                    engine='fp16',
                )
            elif self.reducer == 'PositionWeighted':
                net.SparseLengthsWeightedSum(
                    [
                        self.w,
                        self.external_weights,
                        self.input_record.keys(),
                        self.input_record.lengths(),
                    ],
                    self.output_schema.field_blobs(),
                    grad_on_weights=1,
                    engine='fp16',
                )
            elif self.reducer == 'None':
                # Gather operator will gather the embedding for each id of
                # each IdList.
                net.Gather(
                    [
                        self.w,
                        self.input_record.keys(),
                    ],
                    self.output_schema.field_blobs(),
                    engine='fp16',
                )
            else:
                raise "Only Sum, Mean, None are supported for IdScoreList input." +\
                    "Trying to create with {}".format(self.reducer)
        else:
            raise "Unsupported input type {0}".format(self.input_record)
