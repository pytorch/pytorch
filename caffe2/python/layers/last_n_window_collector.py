## @package last_n_window_collector
# Module caffe2.python.layers.last_n_window_collector
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    LayerParameter,
    ModelLayer,
)


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

        self.last_n = model.net.NextScopedBlob(self.name + "_last_n")
        self.next_blob = model.net.NextScopedBlob(self.name + "_next")

        self.params.append(LayerParameter(
            parameter=self.last_n,
            initializer=core.CreateOperator(
                'ConstantFill', [], self.last_n, shape=[0]
            ),
            optimizer=model.NoOptim,
        ))
        self.params.append(LayerParameter(
            parameter=self.next_blob,
            initializer=core.CreateOperator(
                'ConstantFill',
                [],
                self.next_blob,
                shape=[],
                value=0,
                dtype=core.DataType.INT32,
            ),
            optimizer=model.NoOptim,
        ))

        self.output_schema = schema.from_blob_list(
            input_record, [model.net.NextScopedBlob(name + "_output")])

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
