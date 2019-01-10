## @package fc
# Module caffe2.python.layers.fc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin
import math
import numpy as np


class FC(SamplingTrainableMixin, ModelLayer):

    def __init__(self, model, input_record, output_dims, weight_init=None,
                 bias_init=None, weight_optim=None, bias_optim=None, name='fc',
                 weight_reg=None, bias_reg=None, clip_param=None, **kwargs):
        super(FC, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), (
            "Incorrect input type {}".format(input_record))
        assert len(input_record.field_types()[0].shape) > 0, (
            "FC expects limited dimensions of the input tensor")

        input_dims = input_record.field_types()[0].shape[0]
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

        scale = math.sqrt(1.0 / input_dims)
        weight_init = weight_init if weight_init else (
            'UniformFill', {'min': -scale, 'max': scale})
        bias_init = bias_init if bias_init else (
            'UniformFill', {'min': -scale, 'max': scale})

        self.w = self.create_param(param_name='w',
                                   shape=[output_dims, input_dims],
                                   initializer=weight_init,
                                   optimizer=weight_optim,
                                   regularizer=weight_reg)

        self.b = self.create_param(param_name='b',
                                   shape=[output_dims, ],
                                   initializer=bias_init,
                                   optimizer=bias_optim,
                                   regularizer=bias_reg)

        self.output_schema = schema.Scalar(
            (np.float32, (output_dims, )),
            self.get_next_blob_reference('output')
        )

    def _add_ops(self, net, params):
        if self.clip_args is not None:
            clipped_params = [net.NextScopedBlob(
                'clipped_%s' % str(p)) for p in params]
            for p, cp in zip(params, clipped_params):
                net.Clip([p], [cp], **self.clip_args)
            net.FC(self.input_record.field_blobs() + clipped_params,
                   self.output_schema.field_blobs(), **self.kwargs)
        else:
            net.FC(self.input_record.field_blobs() + params,
                   self.output_schema.field_blobs(), **self.kwargs)

    @property
    def param_blobs(self):
        return [self.w, self.b]
