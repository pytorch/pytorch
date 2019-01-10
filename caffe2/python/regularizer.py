# @package optimizer
# Module caffe2.python.optimizer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.python import core


class Regularizer(object):
    def __init__(self):
        self.apply_after_optimizer = False

    '''
    Adds regularization to train_net for given parameter. Its factor ahead of
    regularization is given when initialization.
    The param should be a BlobReference.
    '''

    def __call__(self, net, param_init_net, param, grad=None):
        assert isinstance(param, core.BlobReference)
        return self._run(net, param_init_net, param, grad)

    def _run(self, net, param_init_net, param, grad):
        raise Exception("Not Impelemented")


class L1Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L1Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

    def _run(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + '_l1_regularization')
        net.LpNorm([param], [output_blob], p=1)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class L2Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L2Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

    def _run(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + '_l2_regularization')
        net.LpNorm([param], [output_blob], p=2)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class MaxNorm(Regularizer):
    def __init__(self, norm=1.0):
        super(MaxNorm, self).__init__()
        self.norm = norm
        self.apply_after_optimizer = True

    def _run(self, net, param_init_net, param, grad):
        assert self.norm > 0, 'norm should be bigger than 0.'
        if isinstance(grad, core.GradientSlice):
            net.SparseNormalize(
                [param, grad.indices, grad.values],
                [param],
                use_max_norm=True,
                norm=self.norm,
            )
        else:
            raise NotImplementedError(
                "MaxNorm is not supported for dense parameters"
            )


class ConstantNorm(Regularizer):
    def __init__(self, norm=1.0):
        super(ConstantNorm, self).__init__()
        self.norm = norm
        self.apply_after_optimizer = True

    def _run(self, net, param_init_net, param, grad):
        assert self.norm > 0, 'norm should be bigger than 0.'
        if isinstance(grad, core.GradientSlice):
            net.SparseNormalize(
                [param, grad.indices, grad.values],
                [param],
                use_max_norm=False,
                norm=self.norm,
            )
        else:
            raise NotImplementedError(
                "ConstantNorm is not supported for dense parameters"
            )
