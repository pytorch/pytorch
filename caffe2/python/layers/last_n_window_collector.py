## @package last_n_window_collector
# Module caffe2.python.layers.last_n_window_collector





from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer


class LastNWindowCollector(ModelLayer):
    """
    Collect last-N samples from input record. If you have complex data,
    use PackRecords to pack it before using this layer.

    This layer is not thread safe.
    """

    def __init__(self, model, input_record, num_to_collect,
                 name='last_n_window_collector', **kwargs):
        super(LastNWindowCollector, self).__init__(
            model, name, input_record, **kwargs)
        assert num_to_collect > 0
        self.num_to_collect = num_to_collect
        assert isinstance(input_record, schema.Scalar), \
            "Got {!r}".format(input_record)

        self.last_n = self.create_param(param_name='last_n',
                                        shape=[0],
                                        initializer=('ConstantFill', {}),
                                        optimizer=model.NoOptim)

        self.next_blob = self.create_param(
            param_name='next',
            shape=[],
            initializer=('ConstantFill',
                         {'value': 0, 'dtype': core.DataType.INT32}),
            optimizer=model.NoOptim
        )

        self.mutex = self.create_param(
            param_name='mutex',
            shape=[],
            initializer=('CreateMutex',),
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

        self.output_schema = schema.Struct(
            (
                'last_n',
                schema.from_blob_list(input_record, [self.last_n])
            ),
            ('num_visited', schema.Scalar(blob=self.num_visited_blob)),
            ('mutex', schema.Scalar(blob=self.mutex)),
        )

    def add_ops(self, net):
        net.LastNWindowCollector(
            [self.last_n, self.next_blob, self.input_record(), self.mutex,
             self.num_visited_blob],
            [self.last_n, self.next_blob, self.num_visited_blob],
            num_to_collect=self.num_to_collect,
        )
