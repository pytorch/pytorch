## @package sparse_lookup
# Module caffe2.python.layers.sparse_lookup
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.helpers.arg_scope import get_current_scope
from caffe2.python import schema
from caffe2.python.layers.layers import (
    get_categorical_limit,
    get_key,
    IdList,
    IdScoreList,
    LayerPsParam,
    ModelLayer,
)
import collections
import functools
import math
import numpy as np
import operator


def get_sparse_lookup_predictor_version(version):
    assert version in {'fp32', 'fp16', 'uint8rowwise', 'fused_uint8rowwise'},\
        "Unexpected version of sparse_lookup layer {0}".format(version)
    return version


def get_sparse_lookup_trainer_version(version):
    assert version in {'fp32', 'fp16'},\
        "Unexpected version of sparse_lookup layer {0}".format(version)
    return version


def _is_id_list(input_record):
    return schema.equal_schemas(input_record, IdList)


def _is_id_score_list(input_record):
    return schema.equal_schemas(input_record,
                                IdScoreList,
                                check_field_types=False)


class SparseLookup(ModelLayer):
    _id_list_supported_reducers = [
        'LogMeanExp', 'LogSumExp', 'Max', 'Mean', 'Sum',
        'WeightedSum', 'WeightedMean', 'Sqrt', 'None']

    _id_score_list_supported_reducers = [
        'PositionWeighted', 'RecencyWeighted', 'Mean', 'Sum', 'WeightedSum',
        'WeightedMean', 'None'
    ]

    _fp16_compatible_init_op_types = [
        'Float16UniformFill'
    ]

    def __init__(self, model, input_record, inner_shape, reducer,
                 weight_init=None, weight_optim=None,
                 name='sparse_lookup', regularizer=None, **kwargs):

        super(SparseLookup, self).__init__(model, name, input_record, **kwargs)

        # TODO Add some asserts about input type
        if isinstance(inner_shape, int):
            inner_shape = [inner_shape]
        assert isinstance(inner_shape, list) or isinstance(inner_shape, tuple),\
            "Unexpected type for inner_shape, expected list or tuple, got {0}".\
            format(type(inner_shape))

        if reducer == "PositionWeighted":
            assert _is_id_score_list(self.input_record), (
                "PositionWeighted only support IdScoreList, but got {} " +
                "please use PositionWeighted layer to convert IdList " +
                "to IdScoreList").format(repr(self.input_record))
            self.external_weights = input_record.values()

        elif reducer == "RecencyWeighted":
            assert _is_id_score_list(self.input_record), (
                "RecencyWeighted only supports IdScoreList.")
            self.external_weights = input_record.values()
        self.reducer = reducer

        input_dim = get_categorical_limit(input_record)
        assert input_dim > 0, (
            "{} should have categorical limit > 0, but got {}".format(
                get_key(input_record)(), input_dim))

        self.input_dim = input_dim
        self.shape = [input_dim] + inner_shape

        cur_scope = get_current_scope()
        trainer_version = get_sparse_lookup_trainer_version(
            **cur_scope.get(get_sparse_lookup_trainer_version.__name__,
                            {'version': 'fp32'}))

        self.trainer_version = trainer_version

        default_init_op = self._get_default_init_op()

        self.weight_init = weight_init or default_init_op

        # If fp16 is used, make sure fp16 init op is used
        if self.trainer_version == "fp16":
            # if init op is UniformFill, we replace it directly
            if self.weight_init[0] == "UniformFill":
                self.weight_init = ("Float16UniformFill", self.weight_init[1])
            assert self.weight_init[0] in self._fp16_compatible_init_op_types, (
                "Fp16 training is enabled. Init op for weight parameter must be fp16 "
                "compatibale. Got {}. Supported ops: {}".format(
                    self.weight_init[0],
                    self._fp16_compatible_init_op_types)
            )

        if _is_id_list(self.input_record):
            sparse_key = self.input_record.items()
        elif _is_id_score_list(self.input_record):
            sparse_key = self.input_record.keys()
        else:
            raise NotImplementedError()

        if self.input_record.lengths.metadata:
            avg_length = self.input_record.lengths.metadata.expected_value
        else:
            avg_length = None

        self.w = self.create_param(
            param_name='w',
            shape=self.shape,
            initializer=self.weight_init,
            optimizer=weight_optim,
            ps_param=LayerPsParam(
                sparse_key=sparse_key,
                average_length=avg_length),
            regularizer=regularizer
        )

        self.scale_bias_init = ('ConstantFill', {'value': 0.0})

        self.scale_bias = self.create_param(
            param_name='scale_bias',
            shape=[],
            initializer=self.scale_bias_init,
            optimizer=model.NoOptim,
        )

        self.output_schema = schema.Scalar(
            (np.float32, inner_shape),
            self.get_next_blob_reference('output'),
        )

    def get_memory_usage(self):
        return functools.reduce(operator.mul, self.shape) * 4

    def get_fp16_compatible_parameters(self):
        return [self.w]

    def support_8bit(self):
        # Rowwise quantization makes sense only if shape it's 2D matrix with
        # second dimension >= 8
        if len(self.shape) != 2 or self.shape[1] < 8:
            return False
        return True

    def get_8bits_compatible_parameters(self, fused=True):
        if not self.support_8bit():
            return []
        if fused:
            RowwiseQuantized8BitsWeight = collections.namedtuple(
                'RowwiseQuantized8BitsWeight', 'w'
            )
            return [RowwiseQuantized8BitsWeight(self.w)]
        else:
            RowwiseQuantized8BitsWeight = collections.namedtuple(
                'RowwiseQuantized8BitsWeight', 'w, scale_bias'
            )
            return [RowwiseQuantized8BitsWeight(self.w, self.scale_bias)]

    def _get_default_init_op(self):
        scale = math.sqrt(1.0 / self.input_dim)

        if self.trainer_version == 'fp32':
            default_weight_init = ('UniformFill', {'min': -scale, 'max': scale})
        elif self.trainer_version == 'fp16':
            default_weight_init = ("Float16UniformFill", {'min': -scale, 'max': scale})
        else:
            raise NotImplementedError(
                "Train version {} is not currently supported".format(trainer_version)
            )

        return default_weight_init

    def _gather_wrapper(self, net, version, in_indices, out):
        # Gather can work on all kinds of input data types, and output
        # data with the same type. Convert the output of Gather to float,
        # because the follow-up Ops expect fp32.
        if version == 'fp32':
            return net.Gather([self.w, in_indices], out)
        elif version == 'fp16':
            gathered_w = net.Gather([self.w, in_indices], 'gathered_w')

            return net.HalfToFloat(gathered_w, out)
        elif version == 'uint8rowwise':
            gathered_w = net.Gather([self.w, in_indices], 'gathered_w')
            gathered_scale_bias = net.Gather(
                [self.scale_bias, in_indices],
                'gathered_scale_bias'
            )

            return net.Rowwise8BitQuantizedToFloat(
                [gathered_w, gathered_scale_bias], out)
        elif version == 'fused_uint8rowwise':
            gathered_w = net.Gather([self.w, in_indices], 'gathered_w')
            return net.Fused8BitRowwiseQuantizedToFloat(gathered_w, out)
        else:
            raise "Unsupported version of operators in SparseLookup " +\
                "layer: {0}".format(version)

    def _sparse_lengths_weighted_reducer(
            self, in_indices, weights, reducer,
            net, version, grad_on_weights=0):
        op_input = [
            self.w,
            weights,
            in_indices,
            self.input_record.lengths()
        ]
        layer_name = 'SparseLengths' + reducer

        if version in ['fp32', 'fp16']:
            # SparseLengths* Ops will accept either fp16 or fp32 embedding
            # matrix and output fp32 pooled embedding
            # A special case here is that we need FP16 engine for
            # SparseLengthsWeightedSum when FP16 embeedings are used for
            # correct backward updates
            if reducer == "WeightedSum" and version == "fp16":
                net.SparseLengthsWeightedSum(
                    op_input,
                    self.output_schema.field_blobs(),
                    grad_on_weights=grad_on_weights,
                    engine='FP16',
                )
            else:
                net.__getattr__(layer_name)(
                    op_input,
                    self.output_schema.field_blobs(),
                    grad_on_weights=grad_on_weights,
                )
        elif version == 'uint8rowwise':
            op_input.insert(len(op_input), self.scale_bias)
            net.__getattr__(layer_name + '8BitsRowwise')(
                op_input, self.output_schema.field_blobs())
        elif version == 'fused_uint8rowwise':
            net.__getattr__(layer_name + 'Fused8BitRowwise')(
                op_input, self.output_schema.field_blobs())
        else:
            raise "Unsupported version of operator in SparseLookUp " +\
                "layer: {0}".format(version)

    # deal with sparse features of id_list type
    def _add_ops_id_list(self, net, version):
        assert self.reducer in self._id_list_supported_reducers, (
            "Unsupported reducer: {} for ID_LIST".format(self.reducer)
        )
        if self.reducer in ['Sum', 'Mean', 'WeightedSum', 'WeightedMean']:
            op_input = [self.w,
                        self.input_record.items(),
                        self.input_record.lengths()]

            # For id list features, the behaviors of 'Sum' and
            # 'WeightedSum' are identical, since we can regard the weight on each
            # id as 1. Similarly, for 'Mean' and 'WeightedMean'.
            if self.reducer == 'WeightedSum':
                self.reducer = 'Sum'
            elif self.reducer == 'WeightedMean':
                self.reducer = 'Mean'

            layer_name = 'SparseLengths' + self.reducer
            if version in ['fp32', 'fp16']:
                # SparseLengths* Ops will accept either fp16 or fp32 embedding
                # matrix and output fp32 pooled embedding
                net.__getattr__(layer_name)(
                    op_input,
                    self.output_schema.field_blobs(),
                )
            elif version == 'uint8rowwise':
                op_input.insert(len(op_input), self.scale_bias)
                net.__getattr__(layer_name + '8BitsRowwise')(
                    op_input, self.output_schema.field_blobs())
            elif version == 'fused_uint8rowwise':
                net.__getattr__(layer_name + 'Fused8BitRowwise')(
                    op_input, self.output_schema.field_blobs())
            else:
                raise "Unsupported version of operator in SparseLookUp " +\
                    "layer: {0}".format(version)

        elif self.reducer == 'Sqrt':
            sqrt_weight = net.LengthsToWeights(
                [self.input_record.lengths()],
                [net.NextScopedBlob('lengths_sqrt')],
                power=0.5,
            )
            self._sparse_lengths_weighted_reducer(
                self.input_record.items(),
                sqrt_weight,
                'WeightedSum', net, version)

        elif self.reducer == 'None':
            # Gather operator will gather the embedding for each id of
            # each IdList.
            self._gather_wrapper(net, version, self.input_record.items(),
                                 self.output_schema.field_blobs())

        else:
            table_rows = self._gather_wrapper(
                net, version, self.input_record.items(), 'table_rows')

            segment_ids = net.LengthsToSegmentIds(
                self.input_record.lengths(),
                net.NextScopedBlob(self.input_record.lengths() + '_sid'))
            net.__getattr__('SortedSegmentRange' + self.reducer)(
                [table_rows, segment_ids],
                self.output_schema.field_blobs(),
            )

    # deal with sparse features of id_score_list type
    def _add_ops_id_score_list(self, net, version):
        assert self.reducer in self._id_score_list_supported_reducers, (
            "Unsupported reducer: {} for ID_SCORE_LIST".format(self.reducer)
        )
        if self.reducer in ['WeightedSum', 'WeightedMean']:
            self._sparse_lengths_weighted_reducer(
                self.input_record.keys(),
                self.input_record.values(),
                self.reducer, net, version)

        elif self.reducer in ['Sum', 'Mean']:
            op_input = [self.w,
                        self.input_record.keys(),
                        self.input_record.lengths()]

            layer_name = 'SparseLengths' + self.reducer

            if version in ['fp32', 'fp16']:
                net.__getattr__(layer_name)(
                    op_input,
                    self.output_schema.field_blobs(),
                )
            elif version == 'uint8rowwise':
                net.__getattr__(layer_name + '8BitsRowwise')(
                    op_input, self.output_schema.field_blobs())
            elif version == 'fused_uint8rowwise':
                net.__getattr__(layer_name + 'Fused8BitRowwise')(
                    op_input, self.output_schema.field_blobs())
            else:
                raise "Unsupported version of operator in SparseLookUp " +\
                    "layer: {0}".format(version)

        elif self.reducer in ['PositionWeighted', 'RecencyWeighted']:
            self._sparse_lengths_weighted_reducer(
                self.input_record.keys(),
                self.external_weights,
                'WeightedSum', net, version, grad_on_weights=1)

        elif self.reducer == 'None':
            # Gather operator will gather the embedding for each id of
            # each IdList.
            self._gather_wrapper(net, version, self.input_record.keys(),
                                 self.output_schema.field_blobs())
        else:
            raise "Only Sum, Mean, None are supported for IdScoreList input." +\
                "Trying to create with {}".format(self.reducer)

    def _add_ops(self, net, version='fp32'):
        if _is_id_list(self.input_record):
            self._add_ops_id_list(net, version=version)
        elif _is_id_score_list(self.input_record):
            self._add_ops_id_score_list(net, version=version)
        else:
            raise "Unsupported input type {0}".format(self.input_record)

    def add_train_ops(self, net):
        self._add_ops(net, self.trainer_version)

    def add_ops(self, net):
        cur_scope = get_current_scope()
        version = get_sparse_lookup_predictor_version(
            **cur_scope.get(get_sparse_lookup_predictor_version.__name__,
                            {'version': 'fp32'}))

        # TODO(amalevich): Layer should not be responsible for decision about
        # quantization.
        if not self.support_8bit() and version in {'uint8rowwise',
                                                   'fused_uint8rowwise'}:
            version = 'fp32'

        self._add_ops(net, version)
