## @package position_weighted
# Module caffe2.python.layers.position_weighted





import logging
import numpy as np

from caffe2.python import schema
from caffe2.python.layers.layers import (
    get_categorical_limit,
    ModelLayer,
)

from caffe2.python.layers.tags import Tags

logger = logging.getLogger(__name__)


class PositionWeighted(ModelLayer):
    def __init__(self, model, input_record, weight_optim=None,
                 name="position_weights"):
        super().__init__(model, name, input_record)

        assert isinstance(input_record, schema.List), "Incorrect input type"
        length_metadata = input_record.lengths.metadata
        max_length = (length_metadata.categorical_limit if length_metadata is
                      not None else None)
        if max_length is not None:
            self.shape = max_length
        else:
            self.shape = get_categorical_limit(input_record)
            logger.warning(
                '{}: categorical_limit of lengths is not available, using '
                'categorical_limit of the keys: {}'.format(
                    str(input_record.lengths()), self.shape))

        self.pos_w = self.create_param(param_name='pos_w',
                                       shape=[self.shape, ],
                                       initializer=('ConstantFill', {'value': 1.0}),
                                       optimizer=weight_optim)

        self.output_schema = schema.Struct(
            ('position_weights',
                schema.Scalar((np.float32, self.shape),
                              self.get_next_blob_reference("pos_w_gather")))
        )

        self.tags.update({Tags.HANDLE_AS_SPARSE_LAYER})

    def get_memory_usage(self):
        return self.shape

    def add_ops(self, net):
        inc_seq = net.LengthsRangeFill(
            [self.input_record.lengths()],
            self.input_record.lengths() + '_pos_w_seq'
        )

        net.Gather(
            [self.pos_w, inc_seq],
            self.output_schema.position_weights.field_blobs())
