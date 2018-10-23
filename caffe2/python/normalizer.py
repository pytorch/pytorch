# @package optimizer
# Module caffe2.python.normalizer
from __future__ import absolute_import, division, print_function, unicode_literals


class Normalizer(object):
    def __init__(self):
        pass
    """
    Adds normalization to train_net for given parameter. Its factor ahead of
    regularization is given when initialization.
    The param should be a BlobReference.
    """

    def __call__(self, net, param):
        return self._run(net, param)

    def _run(self, net, param):
        raise Exception("Not Impelemented")


class BatchNormalizer(Normalizer):
    def __init__(self, momentum):
        super(BatchNormalizer, self).__init__()
        self._momentum = float(momentum)

    def _run(self, layer_model, param):
        return layer_model.BatchNormalization(
            param, momentum=self._momentum
        )


class LayerNormalizer(Normalizer):
    def __init__(self, epsilon, use_layer_norm_op=True):
        super(LayerNormalizer, self).__init__()
        self._epsilon = float(epsilon)
        self._use_layer_norm_op = use_layer_norm_op

    def _run(self, layer_model, param):
        # LayerNorm with dim = 1 is trivial and output constant 0
        if param.dtype.shape[0] > 1:
            return layer_model.LayerNormalization(
                param, epsilon=self._epsilon, use_layer_norm_op=self._use_layer_norm_op
            )
        else:
            return param
