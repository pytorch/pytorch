## @package split
# Module caffe2.python.layers.split
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)


class Split(ModelLayer):

    def __init__(self, model, input_record, num_splits, axis=1,
                 name='split', split_lengths=None, **kwargs):
        super(Split, self).__init__(model, name, input_record, **kwargs)
        self.axis = axis
        self.split_lengths = split_lengths
        # Assume that first dimension is batch, so actual axis in shape is
        # axis - 1
        axis -= 1
        assert axis >= 0

        assert isinstance(input_record, schema.Scalar),\
            "Incorrect input type. Excpected Scalar, but received: {0}".\
            format(input_record)

        input_shape = input_record.field_type().shape
        assert len(input_shape) >= axis

        data_type = input_record.field_type().base
        if split_lengths is not None:
            split_lengths_sum = sum(split_lengths)
            assert split_lengths_sum <= input_shape[axis],\
                'Sum of split lengths must be less than or equal to the'\
                'dimension of the input.'
            output_scalars = []
            if split_lengths_sum == input_shape[axis]:
                for i, split_length in enumerate(split_lengths):
                    output_shape = list(input_shape)
                    output_shape[axis] = split_length
                    output_scalars.append(
                        schema.Scalar(
                            (data_type, output_shape),
                            self.get_next_blob_reference('output_{}'.format(i)),
                        )
                    )
                self.split_lengths = split_lengths
            else:
                # Since the sum of the split_lengths is smaller than the
                # dimension of the input, we create len(split_lengths) + 1
                # splits, using the difference for the last split.
                for i, split_length in enumerate(split_lengths):
                    output_shape = list(input_shape)
                    output_shape[axis] = split_length
                    output_scalars.append(
                        schema.Scalar(
                            (data_type, output_shape),
                            self.get_next_blob_reference('output_{}'.format(i)),
                        )
                    )
                output_shape = list(input_shape)
                output_shape[axis] = input_shape[axis] - split_lengths_sum
                output_scalars.append(
                    schema.Scalar(
                        (data_type, output_shape),
                        self.get_next_blob_reference('output_{}'.format(
                            len(split_lengths)
                        )),
                    )
                )
                self.split_lengths = split_lengths
                self.split_lengths.append(input_shape[axis] - split_lengths_sum)
        else:
            assert input_shape[axis] % num_splits == 0
            output_shape = list(input_shape)
            output_shape[axis] = int(output_shape[axis] / num_splits)

            output_scalars = [
                schema.Scalar(
                    (data_type, output_shape),
                    self.get_next_blob_reference('output_{}'.format(i)),
                )
                for i in range(num_splits)
            ]

        self.output_schema = schema.Tuple(*output_scalars)

    def add_ops(self, net):
        if self.split_lengths:
            net.Split(
                self.input_record.field_blobs(),
                self.output_schema.field_blobs(),
                axis=self.axis,
                split=self.split_lengths,
            )
        else:
            net.Split(
                self.input_record.field_blobs(),
                self.output_schema.field_blobs(),
                axis=self.axis,
            )
