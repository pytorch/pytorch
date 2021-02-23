




import numpy as np

from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer


class MapToRange(ModelLayer):
    """
    This layer aims to build a mapping from raw keys to indices within [0, max_index).
    The mapping is continuously built during training. The mapping will be frozen during
    evaluation and prediction. Unseen keys will be assigned to index 0.
    """

    def __init__(
        self, model,
        input_record,
        max_index,
        name='map_to_range',
        **kwargs
    ):
        super(MapToRange, self).__init__(model, name, input_record, **kwargs)

        assert max_index > 0
        assert isinstance(input_record, schema.Scalar)

        self.max_index = max_index

        self.handler = self.create_param(
            param_name='handler',
            shape=[],
            initializer=('LongIndexCreate', {'max_elements': self.max_index}),
            optimizer=model.NoOptim
        )

        self.output_schema = schema.Struct(
            ('indices', schema.Scalar(
                np.int64, self.get_next_blob_reference("indices")
            )),
            ('handler', schema.Scalar(
                np.void, self.handler
            )),
        )

    def add_train_ops(self, net):
        if self.input_record.field_type().base != np.int64:
            keys = net.Cast(
                self.input_record(),
                net.NextScopedBlob("indices_before_mapping"),
                to=core.DataType.INT64
            )
        else:
            keys = self.input_record()

        # Load keys into indices
        indices = net.IndexGet([self.handler, keys],
                                self.output_schema.indices())

        net.StopGradient(indices, indices)

    def add_eval_ops(self, net):
        net.IndexFreeze(self.handler, self.handler)
        self.add_train_ops(net)

    def add_ops(self, net):
        self.add_eval_ops(net)
