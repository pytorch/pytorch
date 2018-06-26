# @package adaptive_weight
# Module caffe2.fb.python.layers.adaptive_weight
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.regularizer import BoundedGradientProjection, LogBarrier


"""
Implementation of adaptive weighting: https://arxiv.org/pdf/1705.07115.pdf
"""


class AdaptiveWeight(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name="adaptive_weight",
        optimizer=None,
        weights=None,
        enable_diagnose=False,
        estimation_method=None,
        **kwargs
    ):
        super(AdaptiveWeight, self).__init__(model, name, input_record, **kwargs)
        self.output_schema = schema.Scalar(
            np.float32, self.get_next_blob_reference("adaptive_weight")
        )
        self.data = self.input_record.field_blobs()
        self.num = len(self.data)
        self.optimizer = optimizer
        if weights is not None:
            assert len(weights) == self.num
        else:
            weights = [1. / self.num for _ in range(self.num)]
        assert min(weights) > 0, "initial weights must be positive"
        self.weights = np.array(weights).astype(np.float32)
        self.estimation_method = estimation_method
        if self.estimation_method is not None:
            self.estimation_method_type = infer_thrift_union_selection(
                estimation_method
            ).lower()
            self.estimation_method_value = estimation_method.value
        else:
            self.estimation_method_type = "log_std"
            self.estimation_method_value = None
        self.enable_diagnose = enable_diagnose
        self.init_func = getattr(self, self.estimation_method_type + "_init")
        self.weight_func = getattr(self, self.estimation_method_type + "_weight")
        self.reg_func = getattr(self, self.estimation_method_type + "_reg")
        self.init_func()

    def concat_data(self, net):
        reshaped = [net.NextScopedBlob("reshaped_data_%d" % i) for i in range(self.num)]
        # coerce shape for single real values
        for i in range(self.num):
            net.Reshape(
                [self.data[i]],
                [reshaped[i], net.NextScopedBlob("new_shape_%d" % i)],
                shape=[1],
            )
        concated = net.NextScopedBlob("concated_data")
        net.Concat(
            reshaped, [concated, net.NextScopedBlob("concated_new_shape")], axis=0
        )
        return concated

    def log_std_init(self):
        """
        mu = 2 log sigma, sigma = standard variance
        per task objective:
        min 1 / 2 / e^mu X + mu / 2
        """
        values = np.log(1. / 2. / self.weights)
        initializer = (
            "GivenTensorFill",
            {"values": values, "dtype": core.DataType.FLOAT},
        )
        self.mu = self.create_param(
            param_name="mu",
            shape=[self.num],
            initializer=initializer,
            optimizer=self.optimizer,
        )

    def log_std_weight(self, x, net, weight):
        """
        min 1 / 2 / e^mu X + mu / 2
        """
        mu_neg = net.NextScopedBlob("mu_neg")
        net.Negative(self.mu, mu_neg)
        mu_neg_exp = net.NextScopedBlob("mu_neg_exp")
        net.Exp(mu_neg, mu_neg_exp)
        net.Scale(mu_neg_exp, weight, scale=0.5)

    def log_std_reg(self, net, reg):
        net.Scale(self.mu, reg, scale=0.5)

    def inv_var_init(self):
        """
        k = 1 / variance
        per task objective:
        min 1 / 2 * k  X - 1 / 2 * log k
        """
        values = 2. * self.weights
        initializer = (
            "GivenTensorFill",
            {"values": values, "dtype": core.DataType.FLOAT},
        )
        pos_optim_method = self.estimation_method_value.pos_optim_method.getType()
        pos_optim_option = self.estimation_method_value.pos_optim_method.value
        if pos_optim_method == "LOG_BARRIER":
            regularizer = LogBarrier(float(reg_lambda=pos_optim_option.reg_lambda))
        elif pos_optim_method == "POS_GRAD_PROJ":
            regularizer = BoundedGradientProjection(lb=0, left_open=True)
        else:
            raise TypeError(
                "unknown positivity optimization method: {}".format(pos_optim_method)
            )
        self.k = self.create_param(
            param_name="k",
            shape=[self.num],
            initializer=initializer,
            optimizer=self.optimizer,
            regularizer=regularizer,
        )

    def inv_var_weight(self, x, net, weight):
        net.Scale(self.k, weight, scale=0.5)

    def inv_var_reg(self, net, reg):
        log_k = net.NextScopedBlob("log_k")
        net.Log(self.k, log_k)
        net.Scale(log_k, reg, scale=-0.5)

    def add_ops(self, net):
        x = self.concat_data(net)
        weight = net.NextScopedBlob("weight")
        reg = net.NextScopedBlob("reg")
        weighted_x = net.NextScopedBlob("weighted_x")
        weighted_x_add_reg = net.NextScopedBlob("weighted_x_add_reg")
        self.weight_func(x, net, weight)
        self.reg_func(net, reg)
        net.Mul([weight, x], weighted_x)
        net.Add([weighted_x, reg], weighted_x_add_reg)
        net.SumElements(weighted_x_add_reg, self.output_schema())
        if self.enable_diagnose:
            for i in range(self.num):
                weight_i = net.NextScopedBlob("weight_%d" % i)
                net.Slice(weight, weight_i, starts=[i], ends=[i + 1])


def infer_thrift_union_selection(ttype_union):
    # TODO(xlwang): this is a hack way to infer the type str of a thrift union
    # struct
    assert ttype_union.isUnion(), "type {} is not a thrift union".format(
        type(ttype_union)
    )
    field = ttype_union.field
    for attr in dir(ttype_union):
        v = getattr(ttype_union, attr)
        if isinstance(v, int) and attr != "field" and v == field:
            return attr
    raise ValueError("Fail to infer the thrift union type")
