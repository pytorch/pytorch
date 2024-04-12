# @package optimizer
# Module caffe2.python.normalizer



class Normalizer:
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
    def __init__(self, momentum, scale_init_value=1.0):
        super().__init__()
        self._momentum = float(momentum)
        self._scale_init_value = float(scale_init_value)

    def _run(self, layer_model, param):
        return layer_model.BatchNormalization(
            param, momentum=self._momentum, scale_init_value=self._scale_init_value
        )


class LayerNormalizer(Normalizer):
    def __init__(self, epsilon, use_layer_norm_op=True, scale_init_value=1.0):
        super().__init__()
        self._epsilon = float(epsilon)
        self._use_layer_norm_op = use_layer_norm_op
        self._scale_init_value = float(scale_init_value)

    def _run(self, layer_model, param):
        return layer_model.LayerNormalization(
            param, epsilon=self._epsilon, use_layer_norm_op=self._use_layer_norm_op, scale_init_value=self._scale_init_value
        )
