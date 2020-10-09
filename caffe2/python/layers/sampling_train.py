## @package sampling_train
# Module caffe2.python.layers.sampling_train





from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer, get_layer_class
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin


class SamplingTrain(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        prediction_layer,
        output_dims,
        subtract_log_odd=True,
        name='sampling_train',
        **kwargs
    ):
        super(SamplingTrain, self).__init__(
            model, name, input_record, **kwargs
        )

        layer_class = get_layer_class(prediction_layer)
        assert issubclass(layer_class, SamplingTrainableMixin)

        assert 'indices' in input_record
        assert isinstance(input_record.indices, schema.Scalar),\
            "input_record.indices is expected to be a schema.Scalar"
        assert 'input' in input_record

        self.subtract_log_odd = subtract_log_odd
        if self.subtract_log_odd:
            assert 'sampling_prob' in input_record

        self._prediction_layer = layer_class(
            model,
            input_record.input,
            output_dims=output_dims,
            **kwargs
        )

        self._prediction_layer.train_param_blobs = [
            model.net.NextBlob(str(blob) + '_sampled')
            for blob in self._prediction_layer.param_blobs
        ]

        self.params = self._prediction_layer.params

        self.output_schema = self._prediction_layer.output_schema

    def add_ops(self, net):
        self._prediction_layer.add_ops(net)

    def add_train_ops(self, net):
        for full_blob, sampled_blob in zip(
            self._prediction_layer.param_blobs,
            self._prediction_layer.train_param_blobs
        ):
            net.Gather([full_blob, self.input_record.indices()], sampled_blob)
        self._prediction_layer.add_train_ops(net)
        if not self.subtract_log_odd:
            return
        log_q = net.Log(self.input_record.sampling_prob(),
                        net.NextScopedBlob("log_q"))
        net.Sub([self.output_schema(), log_q], self.output_schema(),
                broadcast=1, use_grad_hack=1)
