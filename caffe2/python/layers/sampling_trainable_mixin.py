## @package sampling_trainable_mixin
# Module caffe2.python.layers.sampling_trainable_mixin





import abc


class SamplingTrainableMixin(metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_param_blobs = None
        self._train_param_blobs_frozen = False

    @property
    @abc.abstractmethod
    def param_blobs(self):
        """
        List of parameter blobs for prediction net
        """
        pass

    @property
    def train_param_blobs(self):
        """
        If train_param_blobs is not set before used, default to param_blobs
        """
        if self._train_param_blobs is None:
            self.train_param_blobs = self.param_blobs
        return self._train_param_blobs

    @train_param_blobs.setter
    def train_param_blobs(self, blobs):
        assert not self._train_param_blobs_frozen
        assert blobs is not None
        self._train_param_blobs_frozen = True
        self._train_param_blobs = blobs

    @abc.abstractmethod
    def _add_ops(self, net, param_blobs):
        """
        Add ops to the given net, using the given param_blobs
        """
        pass

    def add_ops(self, net):
        self._add_ops(net, self.param_blobs)

    def add_train_ops(self, net):
        self._add_ops(net, self.train_param_blobs)
