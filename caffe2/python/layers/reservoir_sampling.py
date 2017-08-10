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

        self.reservoir = model.net.NextScopedBlob(name + "_reservoir")
        self.num_visited_blob = model.net.NextScopedBlob(
            name + "_num_visited")
        self.mutex = model.net.NextScopedBlob(name + "_mutex")

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
        self.params.append(
            LayerParameter(
                parameter=self.mutex,
                initializer=core.CreateOperator("CreateMutex", [], self.mutex),
                optimizer=model.NoOptim,
            ),
        )

        self.extra_input_blobs = []
        self.extra_output_blobs = []
        if 'object_id' in input_record:
            self.extra_input_blobs.append(input_record.object_id())
            object_to_pos = model.net.NextScopedBlob(name + "_object_to_pos")
            pos_to_object = model.net.NextScopedBlob(name + "_pos_to_object")
            self.extra_input_blobs.extend([object_to_pos, pos_to_object])
            self.extra_output_blobs.extend([object_to_pos, pos_to_object])
            self.params.append(LayerParameter(
                parameter=object_to_pos,
                initializer=core.CreateOperator(
                    'CreateMap', [], object_to_pos,
                    key_dtype=core.DataType.INT64,
                    valued_dtype=core.DataType.INT32,
                ),
                optimizer=model.NoOptim,
            ))
            self.params.append(LayerParameter(
                parameter=pos_to_object,
                initializer=core.CreateOperator(
                    'ConstantFill',
                    [],
                    pos_to_object,
                    shape=[0],
                    value=0,
                    dtype=core.DataType.INT64,
                ),
                optimizer=model.NoOptim,
            ))

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
