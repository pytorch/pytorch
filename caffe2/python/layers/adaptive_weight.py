# @package adaptive_weight
# Module caffe2.fb.python.layers.adaptive_weight


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
        estimation_method="log_std",
        pos_optim_method="log_barrier",
        reg_lambda=0.1,
        **kwargs
    ):
        super().__init__(model, name, input_record, **kwargs)
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
        self.estimation_method = str(estimation_method).lower()
        # used in positivity-constrained parameterization as when the estimation method
        # is inv_var, with optimization method being either log barrier, or grad proj
        self.pos_optim_method = str(pos_optim_method).lower()
        self.reg_lambda = float(reg_lambda)
        self.enable_diagnose = enable_diagnose
        self.init_func = getattr(self, self.estimation_method + "_init")
        self.weight_func = getattr(self, self.estimation_method + "_weight")
        self.reg_func = getattr(self, self.estimation_method + "_reg")
        self.init_func()
        if self.enable_diagnose:
            self.weight_i = [
                self.get_next_blob_reference("adaptive_weight_%d" % i)
                for i in range(self.num)
            ]
            for i in range(self.num):
                self.model.add_ad_hoc_plot_blob(self.weight_i[i])

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
        if self.pos_optim_method == "log_barrier":
            regularizer = LogBarrier(reg_lambda=self.reg_lambda)
        elif self.pos_optim_method == "pos_grad_proj":
            regularizer = BoundedGradientProjection(lb=0, left_open=True)
        else:
            raise TypeError(
                "unknown positivity optimization method: {}".format(
                    self.pos_optim_method
                )
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

    def _add_ops_impl(self, net, enable_diagnose):
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
        if enable_diagnose:
            for i in range(self.num):
                net.Slice(weight, self.weight_i[i], starts=[i], ends=[i + 1])

    def add_ops(self, net):
        self._add_ops_impl(net, self.enable_diagnose)
