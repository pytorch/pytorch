# @package optimizer
# Module caffe2.python.regularizer


from caffe2.python import core, utils
import numpy as np


class RegularizationBy(object):
    AFTER_OPTIMIZER = "after_optimizer"
    ON_LOSS = "on_loss"


class Regularizer(object):
    def __init__(self):
        self.kEpsilon = 1e-9

    """
    Adds regularization to train_net for given parameter. Its factor ahead of
    regularization is given when initialization.
    The param should be a BlobReference.
    """

    def __call__(self, net, param_init_net, param, grad=None, by=None):
        assert isinstance(param, core.BlobReference)
        by_enum = utils.EnumClassKeyVals(RegularizationBy)
        assert by in by_enum.values(), (
            "Regularizer of type {} is called with invalid by={}, "
            "not in {}".format(self.__class__, by, by_enum.values())
        )
        run_func = "_run_" + by
        assert hasattr(
            self, run_func
        ), "Regularizer of type {} does not implement function {}".format(
            self.__class__, run_func
        )
        return getattr(self, run_func)(net, param_init_net, param, grad)

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        return None

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        return None

    def _feature_grouping(self, param, net):
        # Possible alternative grouping method via summing over absolute values
        # Compute l2norm over feature weights
        # pow( sum_i { pow(theda_i, 2) } ,  0.5)
        param_mul = net.Mul([param, param], [net.NextScopedBlob("param_mul")])
        param_reduced = net.ReduceFrontSum(
            [param_mul], [net.NextScopedBlob("param_reduced")]
        )
        grouped_feature_weight_vec = net.Pow(
            [param_reduced],
            [net.NextScopedBlob("grouped_feature_weight_vec")],
            exponent=0.5,
        )

        return grouped_feature_weight_vec

    def _ensure_clipped(
        self,
        net,
        param,
        grad=None,
        min=None,
        max=None,
        open_range=False,
        left_open=False,
        right_open=False,
    ):
        min = (
            min + self.kEpsilon
            if min is not None and (open_range or left_open)
            else min
        )
        max = (
            max - self.kEpsilon
            if max is not None and (open_range or right_open)
            else max
        )
        input_blobs = (
            [param, grad.indices, grad.values]
            if isinstance(grad, core.GradientSlice)
            else [param]
        )
        net.EnsureClipped(input_blobs, [param], min=min, max=max)


class L1Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L1Norm, self).__init__()
        assert reg_lambda >= 0, "factor ahead of regularization should be 0 or positive"

        self.reg_lambda = reg_lambda

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + "_l1_regularization")
        net.LpNorm([param], [output_blob], p=1)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob

class LpNorm(Regularizer):
    def __init__(self, reg_lambda, p_value=0.5):
        """
        reg_lambda: parameter to scale regularization by

        p_value:    determines what type of Lp norm to calculate. If p > 0,
                    we will calculate Lp norm with the formula:
                    pow( sum_i { pow(theda_i, p) } ,  1/p)
        """
        super(LpNorm, self).__init__()
        assert reg_lambda > 0, "factor ahead of regularization should be greater than 0"
        assert p_value > 0, "p_value factor should be greater than 0"
        self.p_value = p_value
        self.reg_lambda = reg_lambda


    def _run_on_loss(self, net, param_init_net, param, grad=None):
        # TODO: the second dim (num of input nodes) of param is after feature preproc,
        # and does not correspond to the original num of dense features.
        # In the future, will want to create a util to reduce the input dim of param to
        # match the num of dense features.

        output_blob = net.NextScopedBlob(param + "_dense_feature_regularization")
        grouped_feature_weight_vec = self._feature_grouping(param, net)

        # Compute Lpnorm:
        # pow( sum_i { pow(theda_i, p) } ,  1/p)
        lp_vec_raised = net.Pow(
            [grouped_feature_weight_vec],
            [net.NextScopedBlob("lp_vec_raised")],
            exponent=self.p_value,
        )
        lp_vec_summed = net.ReduceFrontSum(
            [lp_vec_raised], [net.NextScopedBlob("lp_vec_summed")]
        )
        lp_norm = net.Pow(
            [lp_vec_summed],
            [net.NextScopedBlob("lp_vec")],
            exponent=(1 / self.p_value),
        )
        net.Scale([lp_norm], [output_blob], scale=self.reg_lambda)
        return output_blob


class L0ApproxNorm(Regularizer):
    def __init__(self, reg_lambda, alpha=0.01, budget=0):
        """
        reg_lambda: parameter to scale regularization by

        alpha:      hyper parameter to tune that is only used in the calculation
                    of approximate L0 norm

        budget:     desired number of features. If the number of features is greater
                    than the budget amount, then the least important features will
                    be penalized. If there are fewer features than the desired
                    budget, no penalization will be applied. Optional parameter, if
                    0, then no budget is used
        """
        super(L0ApproxNorm, self).__init__()
        assert reg_lambda > 0, "factor ahead of regularization should be greater than 0"
        assert alpha > 0, "alpha factor must be a positive value greater than 0"
        assert budget >= 0, "budget factor must be greater than or equal to 0"
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.budget = float(budget)  # budget must be float for future calculations

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        # TODO: the second dim (num of input nodes) of param is after feature preproc,
        # and does not correspond to the original num of dense features.
        # In the future, will want to create a util to reduce the input dim of param to
        # match the num of dense features.

        output_blob = net.NextScopedBlob(param + "_dense_feature_regularization")
        grouped_feature_weight_vec = self._feature_grouping(param, net)

        # compute approximate L0 norm
        # sum_i ( min ( abs (theta_i), alpha))) / alpha
        l0_abs = net.Abs([grouped_feature_weight_vec], [net.NextScopedBlob("l0_abs")])
        l0_min = net.Clip([l0_abs], [net.NextScopedBlob("l0_min")], max=self.alpha)
        l0_summed = net.ReduceFrontSum([l0_min], [net.NextScopedBlob("l0_summed")])
        l0_norm = net.Scale(
            [l0_summed], [net.NextScopedBlob("l0_norm")], scale=(1 / self.alpha)
        )

        # incorporate budget factor
        # regularization = reg_lambda * max(0, l0_norm - budget)
        if self.budget:
            budget_blob = net.ConstantFill([], "budget", shape=[1], value=self.budget)
            l0_sub_budget = net.Sub(
                [l0_norm, budget_blob], [net.NextScopedBlob("l0_budget")]
            )
            relu_l0_sub_budget = net.Relu(
                [l0_sub_budget], [net.NextScopedBlob("relu_l0_sub_budget")]
            )
            net.Scale([relu_l0_sub_budget], [output_blob], scale=self.reg_lambda)
        else:
            net.Scale([l0_norm], [output_blob], scale=self.reg_lambda)
        return output_blob

class L1NormTrimmed(Regularizer):
    """
    The Trimmed Lasso: Sparsity and Robustness. https://arxiv.org/abs/1708.04527
    """
    def __init__(self, reg_lambda, k):
        super(L1NormTrimmed, self).__init__()
        assert reg_lambda >= 0, "factor ahead of regularization should be 0 or positive"
        assert isinstance(k, int), "k should be an interger as expected #. after selection"
        assert k >= 1, "k should be larger than 1"

        self.reg_lambda = reg_lambda
        self.k = k

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + "_l1_trimmed_regularization")
        abs = net.Abs([param], [net.NextScopedBlob("abs")])
        sum_abs = net.SumElements([abs], [net.NextScopedBlob("sum_abs")], average=False)
        topk, _, _ = net.TopK([abs], [net.NextScopedBlob("topk"), net.NextScopedBlob("id"), net.NextScopedBlob("flat_id")], k=self.k)
        topk_sum = net.SumElements([topk], [net.NextScopedBlob("topk_sum")], average=False)
        net.Sub([sum_abs, topk_sum], [output_blob])
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class L2Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L2Norm, self).__init__()
        assert reg_lambda >= 0, "factor ahead of regularization should be 0 or positive"

        self.reg_lambda = reg_lambda

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + "_l2_regularization")
        net.LpNorm([param], [output_blob], p=2)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class ElasticNet(Regularizer):
    def __init__(self, l1, l2):
        super(ElasticNet, self).__init__()
        self.l1 = l1
        self.l2 = l2

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + "_elastic_net_regularization")
        l2_blob = net.NextScopedBlob(param + "_l2_blob")
        l1_blob = net.NextScopedBlob(param + "_l1_blob")
        net.LpNorm([param], [l2_blob], p=2)
        net.LpNorm([param], [l1_blob], p=1)
        net.Scale([l2_blob], [l2_blob], scale=self.l2)
        net.Scale([l1_blob], [l1_blob], scale=self.l1)
        net.Add([l1_blob, l2_blob], [output_blob])
        return output_blob


class ElasticNetL1NormTrimmed(Regularizer):
    def __init__(self, l1, l2, k):
        super(ElasticNetL1NormTrimmed, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.k = k

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + "_elastic_net_l1_trimmed_regularization")
        l2_blob = net.NextScopedBlob(param + "_l2_blob")
        net.LpNorm([param], [l2_blob], p=2)
        net.Scale([l2_blob], [l2_blob], scale=self.l2)

        l1_blob = net.NextScopedBlob(param + "_l1_blob")
        abs = net.Abs([param], [net.NextScopedBlob("abs")])
        sum_abs = net.SumElements([abs], [net.NextScopedBlob("sum_abs")], average=False)
        topk, _, _ = net.TopK([abs], [net.NextScopedBlob("topk"), net.NextScopedBlob("id"), net.NextScopedBlob("flat_id")], k=self.k)
        topk_sum = net.SumElements([topk], [net.NextScopedBlob("topk_sum")], average=False)
        net.Sub([sum_abs, topk_sum], [l1_blob])
        net.Scale([l1_blob], [l1_blob], scale=self.l1)

        net.Add([l1_blob, l2_blob], [output_blob])
        return output_blob


class MaxNorm(Regularizer):
    def __init__(self, norm=1.0):
        super(MaxNorm, self).__init__()
        self.norm = norm

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        assert self.norm > 0, "norm should be bigger than 0."
        if isinstance(grad, core.GradientSlice):
            net.SparseNormalize(
                [param, grad.indices],
                [param],
                use_max_norm=True,
                norm=self.norm,
            )
        else:
            raise NotImplementedError("MaxNorm is not supported for dense parameters")


class ConstantNorm(Regularizer):
    def __init__(self, norm=1.0):
        super(ConstantNorm, self).__init__()
        self.norm = norm

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        assert self.norm > 0, "norm should be bigger than 0."
        if isinstance(grad, core.GradientSlice):
            net.SparseNormalize(
                [param, grad.indices],
                [param],
                use_max_norm=False,
                norm=self.norm,
            )
        else:
            raise NotImplementedError(
                "ConstantNorm is not supported for dense parameters"
            )


class SparseLpNorm(Regularizer):
    def __init__(self, p, reg_lambda):
        super(SparseLpNorm, self).__init__()
        assert p in (1.0, 2.0), "Sparse Lp regularization only implemented for p = 1.0 and p = 2.0."
        assert reg_lambda > 0, "factor ahead of regularization should be greater than 0."
        self.p = p
        self.reg_lambda = reg_lambda

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        if isinstance(grad, core.GradientSlice):
            net.SparseLpRegularizer(
                [param, grad.indices],
                [param],
                p=self.p,
                reg_lambda=self.reg_lambda,
            )
        else:
            raise NotImplementedError("SparseLpNorm is not supported for dense parameters")


class SparseL1Norm(SparseLpNorm):
    def __init__(self, reg_lambda):
        super(SparseL1Norm, self).__init__(p=1.0, reg_lambda=reg_lambda)


class SparseL2Norm(SparseLpNorm):
    def __init__(self, reg_lambda):
        super(SparseL2Norm, self).__init__(p=2.0, reg_lambda=reg_lambda)


class LogBarrier(Regularizer):
    """
    Wright, S., & Nocedal, J. (1999). Numerical optimization. Springer Science,
    35(67-68), 7. Chapter 19
    """

    def __init__(self, reg_lambda, discount_policy="inv", discount_options=None):
        """
        discount is a positive weight that is decreasing, and here it is implemented
        similar to the learning rate. It is specified by a learning rate policy and
        corresponding options
        """
        super(LogBarrier, self).__init__()
        assert reg_lambda > 0, "factor ahead of regularization should be 0 or positive"
        self.reg_lambda = reg_lambda
        self.discount_policy = discount_policy
        self.discount_options = discount_options or {"gamma": 1.0, "power": 1.0}

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        iteration = utils.BuildUniqueMutexIter(param_init_net, net)
        # Since we are most likely to do a minimization
        discount = net.NextScopedBlob(param + "_log_barrier_discount")
        net.LearningRate(
            [iteration],
            [discount],
            base_lr=-self.reg_lambda,
            policy=self.discount_policy,
            **self.discount_options
        )
        # TODO(xlwang): param might still be negative at the initialization time or
        # slightly negative due to the distributed training. Enforce it's non-negativity
        # for now (at least above machine epsilon)
        param_non_neg = net.NextScopedBlob(param + "_non_neg")
        net.Clip([param], [param_non_neg], min=self.kEpsilon)
        param_log = net.NextScopedBlob(param + "_log")
        net.Log([param_non_neg], [param_log])
        param_log_sum = net.NextScopedBlob(param + "_log_sum")
        net.SumElements([param_log], [param_log_sum])
        output_blob = net.NextScopedBlob(param + "_log_barrier")
        net.Mul([param_log_sum, discount], [output_blob], broadcast=1)
        return output_blob

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        self._ensure_clipped(net, param, grad, min=0, open_range=True)


class BoundedGradientProjection(Regularizer):
    """
    Wright, S., & Nocedal, J. (1999). Numerical optimization. Springer Science,
    35(67-68), 7. Chapter 16
    """

    def __init__(
        self, lb=None, ub=None, left_open=False, right_open=False, epsilon=None
    ):
        super(BoundedGradientProjection, self).__init__()
        lb = float(lb) if lb is not None else None
        ub = float(ub) if ub is not None else None
        epsilon = float(epsilon) if epsilon is not None else self.kEpsilon
        assert epsilon > 0, "Bounded Gradient Projection with invalid eps={eps}".format(
            eps=epsilon
        )
        assert (
            (lb is None)
            or (ub is None)
            or (
                lb + (epsilon if left_open else 0.)
                <= ub - (epsilon if right_open else 0.)
            )
        ), (
            "Bounded Gradient Projection with invalid "
            "{lp}ub={ub}, lb={lb}{rp}, eps={eps}".format(
                lb=lb,
                ub=ub,
                lp="(" if left_open else "[",
                rp=")" if right_open else "]",
                eps=epsilon,
            )
        )
        self.left_open = left_open
        self.right_open = right_open
        self.kEpsilon = epsilon
        self.lb = lb
        self.ub = ub

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        self._ensure_clipped(
            net,
            param,
            grad,
            min=self.lb,
            max=self.ub,
            left_open=self.left_open,
            right_open=self.right_open,
        )


class GroupL1Norm(Regularizer):
    """
    Scardapane, Simone, et al. "Group sparse regularization for deep neural networks."
    Neurocomputing 241 (2017): 81-89.

    This regularizer computes l1 norm of a weight matrix based on groups.
    There are essentially three stages in the computation:
    1. Compute the l2 norm on all the members of each group
    2. Scale each l2 norm by the size of each group
    3. Compute the l1 norm of the scaled l2 norms
    """
    def __init__(self, reg_lambda, groups, stabilizing_val=0):
        """
        Args:
            reg_lambda: The weight of the regularization term.
            groups: A list of integers describing the size of each group.
                The length of the list is the number of groups.

        Optional Args:
            stabilizing_val: The computation of GroupL1Norm involves the Sqrt
                operator. When values are small, its gradient can be numerically
                unstable and causing gradient explosion. Adding this term to
                stabilize gradient calculation. Recommended value of this term is
                1e-8, but it depends on the specific scenarios. If the implementation
                of the gradient operator of Sqrt has taken into stability into
                consideration, this term won't be necessary.
        """
        super(GroupL1Norm, self).__init__()
        assert (
            (reg_lambda) >= 0
        ), "regularization weight should be 0 or positive"
        assert isinstance(groups, list), "groups needs to be a list"

        self.reg_lambda = (reg_lambda)
        self.groups = groups
        self.stabilizing_val = stabilizing_val

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        """
        Args:
            param: The input blob to regularize. It should be a weight matrix
                blob with shape (output_dim, input_dim). input_dim should be
                equal to the sum of self.groups.

        Returns:
            group_l1_norm: The output blob after applying regularization.

        These are the steps of computation:
            1. square all elements
            2. sum by row
            3. lengthssum by group
            4. square_root all elements
            5. normalize each group based on group size
            6. compute l1 norm of each group
            7. scale the result with the regularization lambda
        """
        squared = net.Sqr(param)
        reduced_sum = net.ReduceSum(squared, axes=[0], keepdims=0)
        lengths_sum = net.LengthsSum(
            [
                reduced_sum,
                net.GivenTensorIntFill(
                    [], 1, shape=[len(self.groups)], values=self.groups
                ),
            ]
        )

        if self.stabilizing_val:
            net.Add(
                [lengths_sum, net.ConstantFill([], 1, value=self.stabilizing_val)],
                [lengths_sum],
                broadcast=1,
            )

        sqrt = net.Sqrt(lengths_sum)

        # Here we combine step 5 and step 7 into one operator call to
        # improve efficiency: values = np.sqrt(self.groups) * self.reg_lambda
        l2_scaled = net.Mul(
            [
                sqrt,
                net.GivenTensorFill(
                    [],
                    shape=[len(self.groups)],
                    values=np.sqrt(self.groups) * self.reg_lambda
                )
            ],
            ['normalized_l2_norm_scaled']
        )

        group_l1_norm = net.LpNorm(l2_scaled, ['group_l1_nrom'], p=1)

        return group_l1_norm
