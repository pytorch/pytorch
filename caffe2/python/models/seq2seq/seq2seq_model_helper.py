## @package seq2seq_model_helper
# Module caffe2.python.models.seq2seq.seq2seq_model_helper
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import scope
from caffe2.python.model_helper import ModelHelper


class Seq2SeqModelHelper(ModelHelper):

    def __init__(self, init_params=True, **kwargs):
        arg_scope = {
            'use_cudnn': kwargs.pop('use_cudnn', True),
            'cudnn_exhaustive_search': kwargs.pop('cudnn_exhaustive_search', False),
            'order': 'NHWC',
        }
        if kwargs.get('ws_nbytes_limit', None):
            arg_scope['ws_nbytes_limit'] = kwargs.pop('ws_nbytes_limit')

        super(Seq2SeqModelHelper, self).__init__(
            init_params=init_params,
            arg_scope=arg_scope,
            **kwargs
        )
        self.non_trainable_params = []

    def AddParam(self, name, init=None, init_value=None, trainable=True):
        """Adds a parameter to the model's net and it's initializer if needed

        Args:
            init: a tuple (<initialization_op_name>, <initialization_op_kwargs>)
            init_value: int, float or str. Can be used instead of `init` as a
                simple constant initializer
            trainable: bool, whether to compute gradient for this param or not
        """
        if init_value is not None:
            assert init is None
            assert type(init_value) in [int, float, str]
            init = ('ConstantFill', dict(
                shape=[1],
                value=init_value,
            ))

        if self.init_params:
            param = self.param_init_net.__getattr__(init[0])(
                [],
                name,
                **init[1]
            )
        else:
            param = self.net.AddExternalInput(name)

        if trainable:
            self.params.append(param)
        else:
            self.non_trainable_params.append(param)

        return param

    def GetNonTrainableParams(self, namescope=None):
        '''
        Returns the params in current namescope
        '''
        if namescope is None:
            namescope = scope.CurrentNameScope()
        else:
            if not namescope.endswith(scope._NAMESCOPE_SEPARATOR):
                namescope += scope._NAMESCOPE_SEPARATOR

        if namescope == '':
            return self.non_trainable_params[:]
        else:
            return [
                p for p in self.non_trainable_params
                if p.GetNameScope() == namescope
            ]

    def GetAllParams(self, namescope=None):
        return (
            self.GetParams(namescope) +
            self.GetComputedParams(namescope) +
            self.GetNonTrainableParams(namescope)
        )
