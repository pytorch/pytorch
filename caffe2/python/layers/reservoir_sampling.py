## @package reservoir_sampling
# Module caffe2.python.layers.reservoir_sampling
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    LayerParameter,
    ModelLayer,
)


class ReservoirSampling(ModelLayer):
    """
    Collect samples from input record w/ reservoir sampling. If you have complex
    data, use PackRecords to pack it before using this layer.

    This layer is not thread safe.
    """

    def __init__(self, model, input_record, num_to_collect,
                 name='reservoir_sampling', **kwargs):
        super(ReservoirSampling, self).__init__(
            model, name, input_record, **kwargs)
        assert num_to_collect > 0
        self.num_to_collect = num_to_collect
        assert isinstance(input_record, schema.Scalar), \
            "Got {!r}".format(input_record)

        self.reservoir = model.net.NextScopedBlob(name + "_reservoir")
        self.num_visited_blob = model.net.NextScopedBlob(
            name + "_num_visited")

        self.params.append(LayerParameter(
            parameter=self.reservoir,
            initializer=core.CreateOperator(
                'ConstantFill', [], self.reservoir, shape=[0]
            ),
            optimizer=model.NoOptim,
        ))
        self.params.append(LayerParameter(
            parameter=self.num_visited_blob,
            initializer=core.CreateOperator(
                'ConstantFill',
                [],
                self.num_visited_blob,
                shape=[],
                value=0,
                dtype=core.DataType.INT64,
            ),
            optimizer=model.NoOptim,
        ))

        self.output_schema = schema.from_blob_list(
            input_record, [model.net.NextScopedBlob(name + "_output")])

    def add_ops(self, net):
        net.ReservoirSampling(
            [self.reservoir, self.num_visited_blob, self.input_record()],
            [self.reservoir, self.num_visited_blob],
            num_to_collect=self.num_to_collect,
        )
        # Copy to make sure DAG of record is not broken.
        # Also, the output of this is likely going through a pipeline, which
        # will move data and require us to copy anyway.
        net.Copy(self.reservoir, self.output_schema())
