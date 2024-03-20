## @package fc
# Module caffe2.python.layers.fc





from caffe2.python.helpers.arg_scope import get_current_scope
from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin
import math
import numpy as np


def get_fc_predictor_version(fc_version):
    assert fc_version in ["fp32", "fp16"], (
        "Only support fp32 and fp16 for the fully connected layer "
        "in the predictor net, the provided FC precision is {}".format(fc_version)
    )
    return fc_version


class FC(SamplingTrainableMixin, ModelLayer):

    def __init__(self, model, input_record, output_dims, weight_init=None,
                 bias_init=None, weight_optim=None, bias_optim=None, name='fc',
                 weight_reg=None, bias_reg=None, clip_param=None,
                 max_fc_size=None, axis=1, transposed=False,
                 uniform_weight_init_scale_numerator=1.0,
                 **kwargs):
        super().__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), (
            "Incorrect input type {}".format(input_record))
        assert len(input_record.field_types()[0].shape) > 0, (
            "FC expects limited dimensions of the input tensor")
        assert axis >= 1, "axis {} should >= 1.".format(axis)
        self.axis = axis
        input_dims = np.prod(input_record.field_types()[0].shape[axis - 1:])

        assert input_dims > 0, (
            "FC expects input dimensions > 0, got {}".format(input_dims))

        self.clip_args = None
        if (clip_param is not None):
            assert len(clip_param) == 2, (
                'clip_param must be a tuple / list '
                'of length 2 and in the form of (clip_min, clip max)'
            )
            clip_min, clip_max = clip_param
            assert clip_min is not None or clip_max is not None, (
                'clip_min, and clip_max in clip_param cannot both be None'
            )
            assert (
                (clip_min is None or clip_max is None) or clip_min < clip_max
            ), (
                'clip_param = [clip_min, clip_max] must have clip_min < clip_max'
            )
            self.clip_args = {}
            if clip_min is not None:
                self.clip_args['min'] = clip_min
            if clip_max is not None:
                self.clip_args['max'] = clip_max

        if uniform_weight_init_scale_numerator is None:
            uniform_weight_init_scale_numerator = 1.0

        scale = math.sqrt(uniform_weight_init_scale_numerator / input_dims)
        weight_init = weight_init if weight_init else (
            'UniformFill', {'min': -scale, 'max': scale})
        bias_init = bias_init if bias_init else (
            'UniformFill', {'min': -scale, 'max': scale})

        self.output_dim_vec = FC.calculate_fc_output_dims(
            max_fc_size, input_dims, output_dims)

        self.transposed = transposed
        if self.output_dim_vec is None or len(self.output_dim_vec) == 1:
            weight_shape = [input_dims, output_dims] if transposed else [output_dims, input_dims]
            self.w = self.create_param(param_name='w',
                                       shape=weight_shape,
                                       initializer=weight_init,
                                       optimizer=weight_optim,
                                       regularizer=weight_reg)

            self.b = self.create_param(param_name='b',
                                       shape=[output_dims, ],
                                       initializer=bias_init,
                                       optimizer=bias_optim,
                                       regularizer=bias_reg)
        else:
            self.w_vec = []
            self.b_vec = []

            for idx, output_dim in enumerate(self.output_dim_vec):
                weight_shape = [input_dims, output_dim] if transposed else [output_dim, input_dims]
                self.w_vec.append(self.create_param(param_name='w_sub_{}'.format(idx),
                                             shape=weight_shape,
                                             initializer=weight_init,
                                             optimizer=weight_optim,
                                             regularizer=weight_reg))

                self.b_vec.append(self.create_param(param_name='b_sub_{}'.format(idx),
                                             shape=[output_dim, ],
                                             initializer=weight_init,
                                             optimizer=weight_optim,
                                             regularizer=weight_reg))
        if axis == 1:
            output_shape = (output_dims, )
        else:
            output_shape = list(input_record.field_types()[0].shape)[0: axis - 1]
            output_shape = tuple(output_shape + [output_dims])

        self.output_schema = schema.Scalar(
            (np.float32, output_shape),
            self.get_next_blob_reference('output')
        )

    @staticmethod
    def calculate_fc_output_dims(max_fc_size, input_dim, output_dim):

        if not max_fc_size or max_fc_size < 0:
            return None

        assert max_fc_size >= input_dim, "Currently we split along the output " \
            "dimension. So we need max_fc_size >= input_dim. But, max_fc_size: " \
            "{}, input_dim: {}".format(max_fc_size, input_dim)

        output_dim_allowed = int(np.floor(max_fc_size / input_dim))
        num_fc = int(np.floor((output_dim - 1) / output_dim_allowed) + 1)

        output_dim_vec = [output_dim_allowed] * (num_fc - 1)

        output_dim_vec.append(output_dim - sum(output_dim_vec))

        return output_dim_vec

    def _insert_fc_ops(self, net, params, outputs, version):
        """
        Args:
            net: the caffe2 net to insert operator
            params: weight and bias for FC
            outputs: the output blobs
            version: support fp32 and fp16 for now.
        """
        if version == "fp32":
            if self.transposed:
                return net.FCTransposed(
                    self.input_record.field_blobs() + params,
                    outputs,
                    axis=self.axis,
                    **self.kwargs
                )
            else:
                return net.FC(
                    self.input_record.field_blobs() + params,
                    outputs,
                    axis=self.axis,
                    **self.kwargs
                )
        elif version == "fp16":
            return net.FbFCPacked(
                self.input_record.field_blobs() + params,
                outputs,
                axis=self.axis,
                **self.kwargs
            )
        else:
            raise Exception("unsupported FC type version {}".format(version))

    def _add_ops(self, net, params, version):
        """
        Args:
            params : the weight and bias,
                passed by either add_ops or add_train_ops function
            version : fp16 or fp32, might support in8 in the future.
        """
        if self.clip_args is not None:
            clipped_params = [net.NextScopedBlob(
                'clipped_%s' % str(p)) for p in params]
            for p, cp in zip(params, clipped_params):
                net.Clip([p], [cp], **self.clip_args)
            params = clipped_params

        if self.output_dim_vec is None or len(self.output_dim_vec) == 1:
            self._insert_fc_ops(net, params, self.output_schema.field_blobs(), version)
        else:
            w_vec = params[:int(len(params) / 2)]
            b_vec = params[int(len(params) / 2):]

            assert len(w_vec) == len(b_vec)

            output_blob_vec = []

            for i in range(len(self.output_dim_vec)):
                output_blob = net.NextScopedBlob(
                    'output_sub_{}'.format(i))
                insert_ret = self._insert_fc_ops(
                    net, [w_vec[i], b_vec[i]], [output_blob], version
                )
                output_blob_vec.append(insert_ret)
            net.Concat(output_blob_vec,
                       self.output_schema.field_blobs() +
                       [self.output_schema.field_blobs()[0] + "_concat_dims"])

    def add_ops(self, net):
        """Both the predict net and the eval net will call this function
        """
        version_info = get_current_scope().get(
            get_fc_predictor_version.__name__, {'fc_version': 'fp32'}
        )
        predictor_fc_fp_version = version_info['fc_version']
        self._add_ops(net, self.param_blobs, predictor_fc_fp_version)

    def add_train_ops(self, net):
        # use the train_param_blobs to be consistent with the SamplingTrain unittest
        self._add_ops(net, self.train_param_blobs, "fp32")

    def get_fp16_compatible_parameters(self):
        if self.output_dim_vec is None or len(self.output_dim_vec) == 1:
            return [self.w]
        else:
            return self.w_vec

    @property
    def param_blobs(self):
        if self.output_dim_vec is None or len(self.output_dim_vec) == 1:
            return [self.w, self.b]
        else:
            return self.w_vec + self.b_vec
