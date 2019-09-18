# @package batch_unjoined_mse_loss
# Module caffe2.python.layers.batch_unjoined_mse_loss
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.tags import Tags


class BatchUnjoinedMSELoss(ModelLayer):
    def __init__(self, model, input_record, name="batch_unjoined_mse_loss", **kwargs):
        super(BatchUnjoinedMSELoss, self).__init__(model, name, input_record, **kwargs)

        assert schema.is_schema_subset(
            schema.Struct(("label", schema.Scalar()), ("prediction", schema.Scalar())),
            input_record,
        )
        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            np.float32, self.get_next_blob_reference("output")
        )

    def add_ops(self, net):
        prediction = net.Squeeze(
            self.input_record.prediction(),
            net.NextScopedBlob("squeezed_prediction"),
            dims=[1],
        )

        label = self.input_record.label.field_blobs()
        if self.input_record.label.field_type().base != (
            self.input_record.prediction.field_type().base
        ):

            label = net.Cast(
                label,
                net.NextScopedBlob("cast_label"),
                to=schema.data_type_for_dtype(
                    self.input_record.prediction.field_type()
                ),
            )

        label = net.StopGradient(label, net.NextScopedBlob("stopped_label"))

        l2dist = net.SquaredL2Distance([label, prediction], net.NextScopedBlob("l2"))

        neg_sample_correction = net.Scale(net.Sqr(prediction), scale=0.5)

        pos_sample_error = net.Sub(
            [l2dist, neg_sample_correction], net.NextScopedBlob("pos_sample_error")
        )

        label_0 = net.ConstantFill(
            [],
            net.NextScopedBlob("zero_label"),
            value=0.0,
            dtype=core.DataType.FLOAT,
        )
        mask = net.EQ([label, label_0], broadcast=1)

        unjoinedl2dist = net.Conditional(
            [mask, neg_sample_correction, pos_sample_error],
            net.NextScopedBlob("unjoined_l2"),
        )

        if "weight" in self.input_record.fields:
            weight_blob = self.input_record.weight()
            if self.input_record.weight.field_type().base != np.float32:
                weight_blob = net.Cast(
                    weight_blob, weight_blob + "_float32", to=core.DataType.FLOAT
                )
            weight_blob = net.StopGradient(
                [weight_blob], [net.NextScopedBlob("weight_stop_gradient")]
            )
            unjoinedl2dist = net.Mul(
                [unjoinedl2dist, weight_blob],
                net.NextScopedBlob("weighted_unjoined_l2"),
            )

        net.AveragedLoss(unjoinedl2dist, self.output_schema.field_blobs())
