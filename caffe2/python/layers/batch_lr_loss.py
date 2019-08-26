## @package batch_lr_loss
# Module caffe2.python.layers.batch_lr_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
from caffe2.python.layers.tags import (
    Tags
)
import numpy as np


class BatchLRLoss(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name='batch_lr_loss',
        average_loss=True,
        jsd_weight=0.0,
        pos_label_target=1.0,
        neg_label_target=0.0,
        homotopy_weighting=False,
        log_D_trick=False,
        unjoined_lr_loss=False,
        uncertainty_penalty=1.0,
        focal_gamma=0.0,
        stop_grad_in_focal_factor=False,
        **kwargs
    ):
        super(BatchLRLoss, self).__init__(model, name, input_record, **kwargs)

        self.average_loss = average_loss

        assert (schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('logit', schema.Scalar())
            ),
            input_record
        ))

        self.jsd_fuse = False
        assert jsd_weight >= 0 and jsd_weight <= 1
        if jsd_weight > 0 or homotopy_weighting:
            assert 'prediction' in input_record
            self.init_weight(jsd_weight, homotopy_weighting)
            self.jsd_fuse = True
        self.homotopy_weighting = homotopy_weighting

        assert pos_label_target <= 1 and pos_label_target >= 0
        assert neg_label_target <= 1 and neg_label_target >= 0
        assert pos_label_target >= neg_label_target
        self.pos_label_target = pos_label_target
        self.neg_label_target = neg_label_target

        assert not (log_D_trick and unjoined_lr_loss)
        self.log_D_trick = log_D_trick
        self.unjoined_lr_loss = unjoined_lr_loss
        assert uncertainty_penalty >= 0
        self.uncertainty_penalty = uncertainty_penalty

        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output')
        )

        self.focal_gamma = focal_gamma
        self.stop_grad_in_focal_factor = stop_grad_in_focal_factor


    def init_weight(self, jsd_weight, homotopy_weighting):
        if homotopy_weighting:
            self.mutex = self.create_param(
                param_name=('%s_mutex' % self.name),
                shape=None,
                initializer=('CreateMutex', ),
                optimizer=self.model.NoOptim,
            )
            self.counter = self.create_param(
                param_name=('%s_counter' % self.name),
                shape=[1],
                initializer=(
                    'ConstantFill', {
                        'value': 0,
                        'dtype': core.DataType.INT64
                    }
                ),
                optimizer=self.model.NoOptim,
            )
            self.xent_weight = self.create_param(
                param_name=('%s_xent_weight' % self.name),
                shape=[1],
                initializer=(
                    'ConstantFill', {
                        'value': 1.,
                        'dtype': core.DataType.FLOAT
                    }
                ),
                optimizer=self.model.NoOptim,
            )
            self.jsd_weight = self.create_param(
                param_name=('%s_jsd_weight' % self.name),
                shape=[1],
                initializer=(
                    'ConstantFill', {
                        'value': 0.,
                        'dtype': core.DataType.FLOAT
                    }
                ),
                optimizer=self.model.NoOptim,
            )
        else:
            self.jsd_weight = self.model.add_global_constant(
                '%s_jsd_weight' % self.name, jsd_weight
            )
            self.xent_weight = self.model.add_global_constant(
                '%s_xent_weight' % self.name, 1. - jsd_weight
            )

    def update_weight(self, net):
        net.AtomicIter([self.mutex, self.counter], [self.counter])
        # iter = 0: lr = 1;
        # iter = 1e6; lr = 0.5^0.1  = 0.93
        # iter = 1e9; lr = 1e-3^0.1 = 0.50
        net.LearningRate([self.counter], [self.xent_weight], base_lr=1.0,
                         policy='inv', gamma=1e-6, power=0.1,)
        net.Sub(
            [self.model.global_constants['ONE'], self.xent_weight],
            [self.jsd_weight]
        )
        return self.xent_weight, self.jsd_weight

    def add_ops(self, net):
        # numerically stable log-softmax with crossentropy
        label = self.input_record.label()
        # mandatory cast to float32
        # self.input_record.label.field_type().base is np.float32 but
        # label type is actually int
        label = net.Cast(
            label,
            net.NextScopedBlob('label_float32'),
            to=core.DataType.FLOAT)
        label = net.ExpandDims(label, net.NextScopedBlob('expanded_label'),
                                dims=[1])
        if self.pos_label_target != 1.0 or self.neg_label_target != 0.0:
            label = net.StumpFunc(
                label,
                net.NextScopedBlob('smoothed_label'),
                threshold=0.5,
                low_value=self.neg_label_target,
                high_value=self.pos_label_target,
            )
        xent = net.SigmoidCrossEntropyWithLogits(
            [self.input_record.logit(), label],
            net.NextScopedBlob('cross_entropy'),
            log_D_trick=self.log_D_trick,
            unjoined_lr_loss=self.unjoined_lr_loss
        )

        if self.focal_gamma != 0:
            label = net.StopGradient(
                [label],
                [net.NextScopedBlob('label_stop_gradient')],
            )

            prediction = self.input_record.prediction()
            # focal loss = (y(1-p) + p(1-y))^gamma * orginal LR loss
            # y(1-p) + p(1-y) = y + p - 2 * yp
            y_plus_p = net.Add(
                [prediction, label],
                net.NextScopedBlob("y_plus_p"),
            )
            yp = net.Mul([prediction, label], net.NextScopedBlob("yp"))
            two_yp = net.Scale(yp, net.NextScopedBlob("two_yp"), scale=2.0)
            y_plus_p_sub_two_yp = net.Sub(
                [y_plus_p, two_yp], net.NextScopedBlob("y_plus_p_sub_two_yp")
            )
            focal_factor = net.Pow(
                y_plus_p_sub_two_yp,
                net.NextScopedBlob("y_plus_p_sub_two_yp_power"),
                exponent=float(self.focal_gamma),
            )
            if self.stop_grad_in_focal_factor is True:
                focal_factor = net.StopGradient(
                    [focal_factor],
                    [net.NextScopedBlob("focal_factor_stop_gradient")],
                )
            xent = net.Mul(
                [xent, focal_factor], net.NextScopedBlob("focallossxent")
            )

        # fuse with JSD
        if self.jsd_fuse:
            jsd = net.BernoulliJSD(
                [self.input_record.prediction(), label],
                net.NextScopedBlob('jsd'),
            )
            if self.homotopy_weighting:
                self.update_weight(net)
            loss = net.WeightedSum(
                [xent, self.xent_weight, jsd, self.jsd_weight],
                net.NextScopedBlob('loss'),
            )
        else:
            loss = xent

        if 'log_variance' in self.input_record.fields:
            # mean (0.5 * exp(-s) * loss + 0.5 * penalty * s)
            log_variance_blob = self.input_record.log_variance()

            log_variance_blob = net.ExpandDims(
                log_variance_blob, net.NextScopedBlob('expanded_log_variance'),
                dims=[1]
            )

            neg_log_variance_blob = net.Negative(
                [log_variance_blob],
                net.NextScopedBlob('neg_log_variance')
            )

            # enforce less than 88 to avoid OverflowError
            neg_log_variance_blob = net.Clip(
                [neg_log_variance_blob],
                net.NextScopedBlob('clipped_neg_log_variance'),
                max=88.0
            )

            exp_neg_log_variance_blob = net.Exp(
                [neg_log_variance_blob],
                net.NextScopedBlob('exp_neg_log_variance')
            )

            exp_neg_log_variance_loss_blob = net.Mul(
                [exp_neg_log_variance_blob, loss],
                net.NextScopedBlob('exp_neg_log_variance_loss')
            )

            penalized_uncertainty = net.Scale(
                log_variance_blob, net.NextScopedBlob("penalized_unceratinty"),
                scale=float(self.uncertainty_penalty)
            )

            loss_2x = net.Add(
                [exp_neg_log_variance_loss_blob, penalized_uncertainty],
                net.NextScopedBlob('loss')
            )
            loss = net.Scale(loss_2x, net.NextScopedBlob("loss"), scale=0.5)

        if 'weight' in self.input_record.fields:
            weight_blob = self.input_record.weight()
            if self.input_record.weight.field_type().base != np.float32:
                weight_blob = net.Cast(
                    weight_blob,
                    weight_blob + '_float32',
                    to=core.DataType.FLOAT
                )
            weight_blob = net.StopGradient(
                [weight_blob],
                [net.NextScopedBlob('weight_stop_gradient')],
            )
            loss = net.Mul(
                [loss, weight_blob],
                net.NextScopedBlob('weighted_cross_entropy'),
            )

        if self.average_loss:
            net.AveragedLoss(loss, self.output_schema.field_blobs())
        else:
            net.ReduceFrontSum(loss, self.output_schema.field_blobs())
