## @package reservoir_sampling
# Module caffe2.python.layers.reservoir_sampling





from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer


class ReservoirSampling(ModelLayer):
    """
    Collect samples from input record w/ reservoir sampling. If you have complex
    data, use PackRecords to pack it before using this layer.

    This layer is not thread safe.
    """

    def __init__(self, model, input_record, num_to_collect,
                 name='reservoir_sampling', **kwargs):
        super().__init__(model, name, input_record, **kwargs)
        assert num_to_collect > 0
        self.num_to_collect = num_to_collect

        self.reservoir = self.create_param(
            param_name='reservoir',
            shape=[0],
            initializer=('ConstantFill',),
            optimizer=model.NoOptim,
        )
        self.num_visited_blob = self.create_param(
            param_name='num_visited',
            shape=[],
            initializer=('ConstantFill', {
                'value': 0,
                'dtype': core.DataType.INT64,
            }),
            optimizer=model.NoOptim,
        )
        self.mutex = self.create_param(
            param_name='mutex',
            shape=[],
            initializer=('CreateMutex',),
            optimizer=model.NoOptim,
        )

        self.extra_input_blobs = []
        self.extra_output_blobs = []
        if 'object_id' in input_record:
            object_to_pos = self.create_param(
                param_name='object_to_pos',
                shape=None,
                initializer=('CreateMap', {
                    'key_dtype': core.DataType.INT64,
                    'valued_dtype': core.DataType.INT32,
                }),
                optimizer=model.NoOptim,
            )
            pos_to_object = self.create_param(
                param_name='pos_to_object',
                shape=[0],
                initializer=('ConstantFill', {
                    'value': 0,
                    'dtype': core.DataType.INT64,
                }),
                optimizer=model.NoOptim,
            )
            self.extra_input_blobs.append(input_record.object_id())
            self.extra_input_blobs.extend([object_to_pos, pos_to_object])
            self.extra_output_blobs.extend([object_to_pos, pos_to_object])

        self.output_schema = schema.Struct(
            (
                'reservoir',
                schema.from_blob_list(input_record.data, [self.reservoir])
            ),
            ('num_visited', schema.Scalar(blob=self.num_visited_blob)),
            ('mutex', schema.Scalar(blob=self.mutex)),
        )

    def add_ops(self, net):
        net.ReservoirSampling(
            [self.reservoir, self.num_visited_blob, self.input_record.data(),
             self.mutex] + self.extra_input_blobs,
            [self.reservoir, self.num_visited_blob] + self.extra_output_blobs,
            num_to_collect=self.num_to_collect,
        )
