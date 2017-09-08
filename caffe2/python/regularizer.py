# @package optimizer
# Module caffe2.python.optimizer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.python import core


class Regularizer(object):
    def __init__(self):
        pass

    '''
    Adds regularization to train_net for given parameter. Its factor ahead of
    regularization is given when initialization.
    The param should be a BlobReference.
    '''

    def __call__(self, train_net, param):
        assert isinstance(param, core.BlobReference)
        return self._run(train_net, param)

    def _run(self, train_net, param):
        raise Exception("Not Impelemented")


class L1Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L1Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

    def _run(self, train_net, param):
        output_blob = train_net.NextScopedBlob(param + '_l1_regularization')
        train_net.LpNorm([param], [output_blob], p=1)
        train_net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class L2Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L2Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

    def _run(self, train_net, param):
        output_blob = train_net.NextScopedBlob(param + '_l2_regularization')
        train_net.LpNorm([param], [output_blob], p=2)
        train_net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob
