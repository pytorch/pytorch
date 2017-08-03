## @package last_n_window_collector
# Module caffe2.python.layers.last_n_window_collector
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

        self.output_schema = schema.from_blob_list(
            input_record, [self.get_next_blob_reference("output")])

    def add_ops(self, net):
        net.LastNWindowCollector(
            [self.last_n, self.next_blob, self.input_record()],
            [self.last_n, self.next_blob],
            num_to_collect=self.num_to_collect,
        )
        # Copy to make sure DAG of record is not broken.
        # Also, the output of this is likely going through a pipeline, which
        # will move data and require us to copy anyway.
        net.Copy(self.last_n, self.output_schema())
