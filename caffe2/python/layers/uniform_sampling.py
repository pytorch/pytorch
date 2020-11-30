## @package uniform_sampling
# Module caffe2.python.layers.uniform_sampling





import numpy as np

from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer


class UniformSampling(ModelLayer):
    """
    Uniform sampling `num_samples - len(input_record)` unique elements from the
    range [0, num_elements). `samples` is the concatenation of input_record and
    the samples. input_record is expected to be unique.
    """

    def __init__(
        self,
        model,
        input_record,
        num_samples,
        num_elements,
        name='uniform_sampling',
        **kwargs
    ):
        super(UniformSampling, self).__init__(
            model, name, input_record, **kwargs
        )

        assert num_elements > num_samples > 0
        assert isinstance(input_record, schema.Scalar)

        self.num_elements = num_elements

        num_examples_init = ('GivenTensorInt64Fill',
                             {'values': [num_samples]})
        self.num_samples = self.create_param(param_name='num_examples',
                                              shape=(1,),
                                              initializer=num_examples_init,
                                              optimizer=model.NoOptim)

        sampling_blob_init = ('ConstantFill',
                              {'value': float(num_samples) / num_elements,
                               'dtype': core.DataType.FLOAT})
        self.sampling_prob = self.create_param(param_name='prob',
                                               shape=(num_samples,),
                                               initializer=sampling_blob_init,
                                               optimizer=model.NoOptim)

        self.output_schema = schema.Struct(
            (
                'samples', schema.Scalar(
                    np.int32, self.get_next_blob_reference("samples")
                )
            ),
            ('sampling_prob', schema.Scalar(np.float32, self.sampling_prob)),
        )

    def add_ops(self, net):
        net.StopGradient(self.sampling_prob, self.sampling_prob)

        shape = net.Shape([self.input_record()], net.NextScopedBlob("shape"))
        shape = net.Sub([self.num_samples, shape], shape)
        samples = net.UniqueUniformFill(
            [shape, self.input_record()],
            net.NextScopedBlob("samples_before_concat"),
            min=0,
            max=self.num_elements - 1,
            input_as_shape=True
        )

        net.Concat(
            [self.input_record(), samples],
            [self.output_schema.samples(), net.NextScopedBlob("split_info")],
            axis=0
        )
        net.StopGradient(
            self.output_schema.samples(), self.output_schema.samples()
        )
