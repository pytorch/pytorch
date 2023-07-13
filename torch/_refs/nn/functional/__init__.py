import math
from functools import wraps
from typing import Callable, Optional, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    NumberType,
    ShapeType,
    TensorLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import (
    elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper,
    out_wrapper,
)
from torch._refs import _make_inplace

__all__ = [
    "alpha_dropout",
    "celu",
    "celu_",
    "dropout",
    "elu",
    "elu_",
    "gelu",
    "glu",
    "group_norm",
    "hardshrink",
    "hardtanh",
    "hinge_embedding_loss",
    "huber_loss",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "log_softmax",
    "margin_ranking_loss",
    "mish",
    "mish_",
    "mse_loss",
    "nll_loss",
    "pairwise_distance",
    "pdist",
    "poisson_nll_loss",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "selu_",
    "smooth_l1_loss",
    "softmax",
    "softmin",
    "softplus",
    "softshrink",
    "tanhshrink",
    "threshold",
    "threshold_",
    "triplet_margin_loss",
]

Tensor = torch.Tensor
aten = torch._ops.ops.aten


def _dropout_helper(
    self: TensorLikeType,
    val: float,
) -> TensorLikeType:
    """
    Helper function for all dropout-type operators. During training,
    some of the elements of the input tensor are randomly masked.

    Returns the masked tensor of the boolean values.

    """

    return (
        refs._uniform_helper(
            self.shape, low=0.0, high=1.0, dtype=torch.float32, device=self.device
        )
        < val
    )


@register_decomposition(aten.alpha_dropout)
def alpha_dropout(
    self: TensorLikeType, p: float = 0.5, training: bool = False, inplace: bool = False
) -> TensorLikeType:
    if inplace:
        raise NotImplementedError

    if not training:
        return self

    torch._check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    if p == 1:
        return torch.zeros_like(self)

    if p == 0:
        return self

    dropout_mask = _dropout_helper(self, 1 - p)

    # From paper: Self-Normalizing Neural Networks (https://arxiv.org/pdf/1706.02515.pdf)
    # alpha = - SELU.alpha * SELU.scale, here
    # SELU.alpha = 1.6732632423543772848170429916717 and
    # SELU.scale = 1.0507009873554804934193349852946
    alpha = -1.7580993408473766

    a = 1.0 / math.sqrt((alpha * alpha * p + 1) * (1 - p))
    b = torch.logical_not(dropout_mask)
    b = b * (alpha * a) + alpha * a * p
    dropout_mask = a * dropout_mask

    return self * dropout_mask + b


def _inplace_wrapper(fn):
    """
    Given a nn.functional non-linearity, implements its `inplace: bool` argument
    """

    # nb. We use the name of the first argument used in the unary references
    @wraps(fn)
    def _fn(a, *args, inplace=False, **kwargs):
        if inplace:
            torch._check(
                "out" not in kwargs,
                lambda: "Cannot set inplace=True and pass out= at the same time",
            )
            return fn(a, *args, inplace=False, out=a, **kwargs)
        else:
            return fn(a, *args, inplace=False, **kwargs)

    return _fn


# celu is implemented specially because it has an alpha argument
# celu is very similar to elu
@register_decomposition(aten.celu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def celu(
    a: TensorLikeType, alpha: Optional[NumberType] = None, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.celu
    """

    if inplace:
        raise NotImplementedError

    rhs: TensorLikeType
    if alpha is not None:
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = (
                "alpha argument of type {0} cannot be safely cast to type {1}!".format(
                    type(alpha), python_type
                )
            )
            raise ValueError(msg)
        rhs = alpha * torch.expm1(torch.true_divide(a, alpha))  # type: ignore[arg-type]
    else:
        rhs = torch.expm1(a)

    return torch.where(a > 0, a, rhs)


@register_decomposition(aten.dropout)
@_inplace_wrapper
@out_wrapper()
def dropout(
    a: TensorLikeType, p: float = 0.5, training: bool = True, inplace: bool = False
) -> TensorLikeType:
    if inplace:
        raise NotImplementedError

    if not training:
        return a

    torch._check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    if p == 1:
        return torch.zeros_like(a)

    if p == 0:
        return a

    scale = 1 / (1 - p)
    dropout_mask = _dropout_helper(a, 1 - p)

    return a * dropout_mask * scale


@register_decomposition(aten.elu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def elu(
    a: TensorLikeType,
    alpha: NumberType = 1.0,
    scale: NumberType = 1.0,
    input_scale: NumberType = 1.0,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.elu
    """
    if inplace:
        raise NotImplementedError

    # nb. This should be factored out into a can_cast aux function
    python_type = utils.dtype_to_type(a.dtype)
    torch._check(
        utils.is_weakly_lesser_type(type(input_scale), python_type),
        lambda: f"input_scale argument of type {type(input_scale)} cannot be safely cast to type {python_type}!",
    )
    torch._check(
        utils.is_weakly_lesser_type(type(scale), python_type),
        lambda: f"scale argument of type {type(scale)} cannot be safely cast to type {python_type}!",
    )
    torch._check(
        utils.is_weakly_lesser_type(type(alpha), python_type),
        lambda: f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!",
    )

    return torch.where(a > 0, scale * a, (alpha * scale) * torch.expm1(a * input_scale))


@register_decomposition(aten.relu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def relu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu
    """

    if inplace:
        raise NotImplementedError

    return torch.where(torch.le(a, 0), 0, a)


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Reference implementation of :func:`torch.nn.functional.group_norm`.
    """
    torch._check(
        input.ndim >= 2,
        lambda: f"Expected at least 2 dimensions for input tensor but received {input.ndim}",
    )

    batch_size = input.shape[0]
    num_channels = input.shape[1]
    torch._check(
        num_channels % num_groups == 0,
        lambda: "Expected number of channels in input to be divisible by num_groups, "
        + f"but got input of shape {input.shape} and num_groups = {num_groups}",
    )

    # input shape is (N, C, *), so we flatten all inner dimensions except (N, C)
    flattened_inner_size = 1
    for dim_length in input.shape[2:]:
        flattened_inner_size *= dim_length

    return torch.native_group_norm(
        input,
        weight,
        bias,
        batch_size,
        num_channels,
        flattened_inner_size,
        num_groups,
        eps,
    )[0]


def layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Reference implementation of :func:`torch.nn.functional.layer_norm`.
    """
    return torch.native_layer_norm(input, normalized_shape, weight, bias, eps)[0]


@register_decomposition(aten.leaky_relu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def leaky_relu(
    a: TensorLikeType, negative_slope: float = 0.01, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.leaky_relu
    """

    if inplace:
        raise NotImplementedError

    python_type = utils.dtype_to_type(a.dtype)
    if not utils.is_weakly_lesser_type(type(negative_slope), python_type):
        msg = f"negative_slope argument of type {type(negative_slope)} cannot be safely cast to type {python_type}!"
        raise ValueError(msg)
    return torch.where(torch.gt(a, 0), a, torch.mul(a, negative_slope))


@register_decomposition(aten.mish)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def mish(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.mish
    """

    if inplace:
        raise NotImplementedError
    return a * torch.tanh(torch.nn.functional.softplus(a))


@register_decomposition(aten.selu)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def selu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.selu
    """
    if inplace:
        raise NotImplementedError

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    rhs = alpha * torch.expm1(a)

    return scale * torch.where(a > 0, a, rhs)


# Forwarding alias: the functional variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def softmax(
    a: TensorLikeType,
    dim: Optional[int] = None,
    _stacklevel: int = 3,  # for compat when using TorchRefsMode(strict=True)
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # The error is for compat with regular PyTorch, which has this behavior
    # deprecated.  For PrimTorch, it's fine to drop support for deprecated
    # behavior because it requires explicit opt in.  This error is to inform
    # users how to update their calls.
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# CompositeImplicitAutograd - don't register decomp
def softmin(
    a: TensorLikeType,
    dim: Optional[int] = None,
    _stacklevel: int = 3,  # for compat when using TorchRefsMode(strict=True)
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # The error is for compat with regular PyTorch, which has this behavior
    # deprecated.  For PrimTorch, it's fine to drop support for deprecated
    # behavior because it requires explicit opt in.  This error is to inform
    # users how to update their calls.
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    return torch.softmax(a=-a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# softplus is implemented specially because it has beta and threshold arguments
@register_decomposition(aten.softplus)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def softplus(
    a: TensorLikeType,
    beta: Optional[NumberType] = None,
    threshold: NumberType = 20,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.softplus
    """

    if inplace:
        raise NotImplementedError

    rhs: TensorLikeType
    if beta is not None:
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(beta), python_type):
            msg = "beta argument of type {0} cannot be safely cast to type {1}!".format(
                type(beta), python_type
            )
            raise ValueError(msg)
        scaled_input = a * beta
        rhs = torch.true_divide(torch.log1p(torch.exp(scaled_input)), beta)  # type: ignore[arg-type]

    else:
        scaled_input = a
        rhs = torch.log1p(torch.exp(scaled_input))

    return torch.where(scaled_input > threshold, a, rhs)


@register_decomposition(aten.hardshrink)
@out_wrapper()
def hardshrink(a: TensorLikeType, lambd: float = 0.5):
    # Formula for reference,
    # hardshrink(x) = x if x > lambd
    #               = x if x < -lambd
    #               = 0 otherwise
    return torch.where(torch.logical_and(a >= -lambd, a <= lambd), 0, a)


@register_decomposition(aten.softshrink)
@out_wrapper()
def softshrink(a: TensorLikeType, lambd: float = 0.5):
    # Formula for reference,
    # softshrink(x) = x - lambd if x > lambd
    #               = x + lambd if x < -lambd
    #               = 0 otherwise
    torch._check(
        lambd >= 0,
        lambda: f"lambda must be greater or equal to 0, but found to be {lambd}",
    )
    ge_mask = a > lambd
    le_mask = a < -lambd
    zero_mask = torch.logical_not(torch.logical_or(ge_mask, le_mask))
    result = torch.where(ge_mask, a - lambd, a)
    result = torch.where(le_mask, a + lambd, result)
    return torch.where(zero_mask, 0, result)


# Losses
def _reduction_int_to_str(reduction: int) -> str:
    from torch._decomp.decompositions import Reduction

    if reduction == Reduction.NONE.value:
        return "none"
    elif reduction == Reduction.MEAN.value:
        return "mean"
    elif reduction == Reduction.SUM.value:
        return "sum"
    else:
        raise ValueError(f"{reduction} is not a valid value for reduction")


def _apply_loss_reduction(loss: TensorLikeType, reduction: str) -> TensorLikeType:
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss


def _check_reduction_value(reduction: str):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")


# This helper function maps depreciated arguments, "size_average" and "reduce"
# to their corresponding "reduction" string argument
def _get_string_reduction_arg(
    *, size_average: Optional[bool], reduce: Optional[bool]
) -> str:
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    return ret


# CompositeImplicitAutograd - don't register decomp
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.l1_loss
    """
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)
    loss = torch.abs(input - target)
    return _apply_loss_reduction(loss, reduction)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def smooth_l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.smooth_l1_loss
    """
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)

    if beta == 0.0:
        return torch.nn.functional.l1_loss(
            input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    else:
        loss = torch.abs(input - target)
        loss = torch.where(loss < beta, 0.5 * loss**2 / beta, loss - 0.5 * beta)
        return _apply_loss_reduction(loss, reduction)


# Forwarding alias: the functional variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def log_softmax(
    a: TensorLikeType,
    dim: Optional[int] = None,
    _stacklevel: int = 3,  # for compat when using TorchRefsMode(strict=True)
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # The error is for compat with regular PyTorch, which has this behavior
    # deprecated.  For PrimTorch, it's fine to drop support for deprecated
    # behavior because it requires explicit opt in.  This error is to inform
    # users how to update their calls.
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


@register_decomposition(aten.margin_ranking_loss)
def margin_ranking_loss(
    input1: TensorLikeType,
    input2: TensorLikeType,
    target: TensorLikeType,
    margin: float = 0.0,
    reduction: str = "mean",
) -> TensorLikeType:
    # loss_without_reduction = max(0, −target * (input1 − input2) + margin)
    if input1.ndim != input2.ndim or input1.ndim != target.ndim:
        raise RuntimeError(
            (
                "margin_ranking_loss : All input tensors should have same dimension but got sizes: "
                "input1: {}, input2: {}, target: {} ".format(
                    input1.shape, input2.shape, target.shape
                )
            )
        )
    _check_reduction_value(reduction)
    loss = torch.clamp_min(-target * (input1 - input2) + margin, 0)
    return _apply_loss_reduction(loss, reduction)


@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
def mse_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)
    loss = torch.pow(input - target, 2)
    return _apply_loss_reduction(loss, reduction)


@register_decomposition(aten.hinge_embedding_loss)
def hinge_embedding_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    margin: float = 1.0,
    reduction: str = "mean",
) -> TensorLikeType:
    # loss_without_reduction = input if y == 1
    #                        = max(0, margin - input) if y == -1
    _check_reduction_value(reduction)
    margin_clamp = torch.clamp_min(margin - input, 0)
    output_margin = torch.where(target != 1, margin_clamp, 0)
    output_self = torch.where(target != -1, input, 0)
    loss = output_margin + output_self
    return _apply_loss_reduction(loss, reduction)


def _nll_loss_nd(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: Optional[TensorLikeType],
    reduction: str,
    ignore_index: int,
) -> TensorLikeType:
    torch._check(
        input.ndim > 0 and input.ndim <= 3,
        lambda: f"Expected input dimension to be either [1, 2, 3] but received {input.ndim}.",
    )

    torch._check(
        (input.ndim == 1) or (input.shape[0] == target.shape[0]),
        lambda: f"Expected input batch size {input.shape[0]} to match target batch size {target.shape[0]}.",
    )

    _check_reduction_value(reduction)

    flat_target = torch.flatten(target)
    ignore_classes_mask = torch.eq(flat_target, ignore_index)

    # TODO: Enable data-dependent checks with debug mode
    # TODO: This check does not work with FakeTensor inputs; See Issue #85834
    # Explicit cast for class_check to bool; See Issue #78071
    """
    from torch._subclasses.fake_tensor import FakeTensor
    num_classes = input.shape[1] if input.ndim > 1 else input.shape[0]
    valid_classes_mask = torch.logical_and(
        (flat_target >= 0), (flat_target < num_classes)
    )
    class_check = torch.all(torch.logical_or(ignore_classes_mask, valid_classes_mask))
    torch._check(
        isinstance(target, FakeTensor) or bool(class_check.item()),
        lambda: "A target class is out-of-bounds and not the ignore index.",
    )
    """

    ignore_class_weight = torch.scalar_tensor(0, dtype=input.dtype, device=input.device)
    class_weight = (
        torch.scalar_tensor(1, dtype=input.dtype, device=input.device)
        if weight is None
        else weight[flat_target]
    )
    current_weight = torch.where(
        ignore_classes_mask,
        ignore_class_weight,
        class_weight,
    )

    if input.ndim == 1:
        # implicit batch size = 1
        # input (1 batch size, C classes)
        loss = -input[target] * current_weight
    elif input.ndim == 2:
        # input (N batch size, C classes)
        batch_size = input.shape[0]
        loss = -input[torch.arange(batch_size), target] * current_weight
    else:
        # 3D case (N batch size, C classe, K dimensions)
        # input (N batch size, C classes, K)
        batch_size = input.shape[0]
        extent = input.shape[2]
        numel = batch_size * extent
        indices = torch.arange(numel)
        bdx = indices // extent
        kdx = indices % extent
        loss = -input[bdx, flat_target, kdx] * current_weight
    loss = torch.reshape(loss, target.shape)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        # calculate weighted mean of the loss function
        return torch.sum(loss) / torch.sum(current_weight)


@register_decomposition(aten.nll_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: Optional[TensorLikeType] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.nll_loss
    """
    torch._check(
        input.ndim > 0,
        lambda: f"Expected input tensor to have 1 or more dimensions (got {input.ndim})",
    )

    # TODO: raise exception instead of converting value
    # msg = "size_average and reduce args are deprecated, please use reduction argument."
    # Convert these options for consistency with the eager mode
    if size_average is not None or reduce is not None:
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)

    # The expected behavior when the target and input have zero elements:
    #   reduction = 'none' --- tensor([])
    #   reduction = 'sum'  --- tensor(0.)
    #   reduction = 'mean' --- tensor(nan)
    # Mean reduction on empty tensors produces NaN. See the discussion in
    # https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if input.numel() == 0 and target.numel() == 0:
        if reduction == "none":
            return torch.zeros_like(target)
        elif reduction == "sum":
            return torch.empty_like(target)
        else:
            return torch.full_like(target, float("nan"))

    # The _nll_loss_nd helper function handles the most common cases.
    # ndim == 1 (Single Example)
    #   => Batch Size: 1, Input: (C), Target: ()
    # ndim == 2 (k = 1)
    #   => Batch Size: N, Input: (N, C), Target: (N)
    # ndim == 3 (k > 1)
    #   => Batch Size: N, Input: (N, C, K), Target: (N, K)
    if input.ndim <= 3:
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)

    # For ndim > 3, we reshape the input and target to 3-D case.
    # Input (N batch-size, C classes, k-dimensions)
    # Target (N batch-size, k-dimensions)
    torch._check(
        input.ndim > 0 and target.ndim > 0 and target.shape[1:] == input.shape[2:],
        lambda: (
            "Expected input and target to both have ndim > 0 and "
            "target.shape[1:] == input.shape[2:], but got "
            f"target.shape {target.shape} and input.shape {input.shape}"
        ),
    )

    batch_size = input.shape[0]
    num_classes = input.shape[1]
    out_size = [batch_size] + list(target.shape[1:])

    input = torch.reshape(input, [batch_size, num_classes, -1])
    target = torch.reshape(target, [batch_size, -1])
    if reduction != "none":
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)
    else:
        result = _nll_loss_nd(input, target, weight, reduction, ignore_index)
        # reshape flattened inner-dim to original k-dimensions
        return torch.reshape(result, out_size)


# TODO: This ref supports int reduction and out kwarg to be compatible with ATen:
# https://github.com/pytorch/pytorch/issues/83931
# TODO: Could be rewritten to support complex:
# https://github.com/pytorch/pytorch/pull/85041
@register_decomposition(aten.huber_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def huber_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    reduction: Union[str, int] = "mean",
    delta: float = 1.0,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.huber_loss
    """
    if type(reduction) is int:
        reduction = _reduction_int_to_str(reduction)
    _check_reduction_value(reduction)  # type: ignore[arg-type]
    torch._check(
        delta > 0,
        lambda: "huber_loss does not support non-positive values for delta.",
    )
    z = (input - target).abs()
    loss = torch.where(z < delta, 0.5 * z * z, delta * (z - 0.5 * delta))
    return _apply_loss_reduction(loss, reduction)  # type: ignore[arg-type]


# tanhshrink does not use _make_elementwise_unary_reference because it does not support out
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def tanhshrink(a: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.tanhshrink
    """
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    return a - torch.tanh(a)


@register_decomposition(aten.threshold)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def threshold(
    a: TensorLikeType,
    threshold: NumberType,
    value: Union[bool, int, float],
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.threshold
    """

    if inplace:
        raise NotImplementedError

    return torch.where(a <= threshold, value, a)


# CompositeImplicitAutograd - don't register decomp
# No elementwise type promotion - core op doesn't explicitly type promote
def triplet_margin_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)

    # torch.nn.functional.triplet_margin_with_distance_loss has no ref defined
    # since it's a pure Python implementation.  Use this helper instead.
    return _triplet_margin_with_distance_loss(
        anchor=anchor,
        positive=positive,
        negative=negative,
        distance_function=lambda x, y: torch.pairwise_distance(x, y, p, eps),
        margin=margin,
        swap=swap,
        reduction=reduction,
    )


# Pure Python impl - don't register decomp and don't add a ref.  Defined as a
# helper here since triplet_margin_loss can be nicely implemented with it.
def _triplet_margin_with_distance_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    *,
    distance_function: Optional[
        Callable[[TensorLikeType, TensorLikeType], TensorLikeType]
    ] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> TensorLikeType:
    _check_reduction_value(reduction)

    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    torch._check(
        a_dim == p_dim and p_dim == n_dim,
        lambda: (
            f"The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        ),
    )

    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)
    return _apply_loss_reduction(loss, reduction)


@register_decomposition(aten.hardtanh)
@_inplace_wrapper
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def hardtanh(
    a: TensorLikeType,
    min_val: NumberType = -1,
    max_val: NumberType = 1,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.hardtanh
    """
    if inplace:
        raise NotImplementedError
    if utils.is_boolean_dtype(a.dtype):
        raise RuntimeError("Bool inputs not supported for hardtanh")

    # preserve legacy behavior of boundaries not causing type promotion
    if utils.is_integer_dtype(a.dtype):
        min_val = int(min_val)  # type: ignore[arg-type]
        max_val = int(max_val)  # type: ignore[arg-type]
        if not (a.dtype != torch.uint8 or (min_val >= 0 and max_val >= 0)):
            raise RuntimeError(
                "Cannot do hardtanh on an unsigned type with negative limits"
            )
    return torch.clamp(a, min_val, max_val)  # type: ignore[arg-type]


@register_decomposition(aten.gelu)
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def gelu(a: TensorLikeType, approximate: str = "none") -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.gelu
    """
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        a_cube = a * a * a
        inner = kBeta * (a + kKappa * a_cube)
        return 0.5 * a * (1 + torch.tanh(inner))
    elif approximate == "none":
        kAlpha = M_SQRT1_2
        return a * 0.5 * (1 + torch.erf(a * kAlpha))
    else:
        raise RuntimeError("approximate argument must be either none or tanh.")


# CompositeImplicitAutograd - don't register decomp
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def poisson_nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    log_input: bool = True,
    full: bool = False,
    size_average: Optional[bool] = None,
    eps: float = 1e-8,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.poisson_nll_loss
    """
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    _check_reduction_value(reduction)
    if log_input:
        loss = torch.exp(input) - target * input
    else:
        loss = input - target * torch.log(input + eps)

    if full:
        stirling_term = (
            target * torch.log(target) - target + 0.5 * torch.log(2 * torch.pi * target)
        )
        # avoid inplace add
        loss = loss + stirling_term.masked_fill(target <= 1, 0)
    return _apply_loss_reduction(loss, reduction)


@register_decomposition(aten.prelu)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "weight"),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def prelu(a: TensorLikeType, weight: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.prelu
    """
    torch._check(
        isinstance(a, TensorLike),
        lambda: f"prelu: Expected `a` to be tensor, but got: {type(a)}",
    )
    torch._check(
        isinstance(weight, TensorLike),
        lambda: f"prelu: Expected `weight` to be tensor, but got: {type(weight)}",
    )

    if weight.numel() != 1:
        torch._check(a.ndim > 0, lambda: "Not allow zero-dim input tensor.")
        channel_size = a.shape[1] if a.ndim >= 2 else 1
        torch._check(
            weight.numel() == channel_size,
            lambda: f"Mismatch of parameter numbers and input channel size. Found parameter numbers ="
            f" {weight.numel()} and channel size = {channel_size}.",
        )

    torch._check(
        weight.ndim == 0 or weight.ndim == 1,
        lambda: f"prelu: Expected `weight` to be a scalar or 1D tensor, but got: "
        f"ndim = {weight.ndim}",
    )
    if a.ndim == 0:
        weight = weight[0] if weight.ndim == 1 else weight
    else:
        weight = prims.broadcast_in_dim(
            weight, a.shape, tuple() if weight.ndim == 0 else (0 if a.ndim == 1 else 1,)
        )

    return torch.where(a > 0, a, a * weight)


@register_decomposition(aten.relu6)
@_inplace_wrapper
@out_wrapper()
def relu6(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu6
    """
    if inplace:
        raise NotImplementedError

    # See https://github.com/pytorch/pytorch/pull/81142#discussion_r918220126
    # It may be better to use clamp here, but we use hardtanh to replicate
    # the behavior of the existing implementation
    return torch.nn.functional.hardtanh(a, 0, 6)


@register_decomposition(aten.glu)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def glu(a: TensorLikeType, dim: int = -1) -> TensorLikeType:
    dim = utils.canonicalize_dims(a.ndim, dim)
    torch._check(
        a.shape[dim] % 2 == 0,
        lambda: f"Halving dimension must be even, but dimension {dim} is size {a.shape[dim]}",
    )
    b, c = torch.tensor_split(a, 2, dim)

    return b * torch.sigmoid(c)


@register_decomposition(aten.pairwise_distance)
@out_wrapper()
def pairwise_distance(
    x1: TensorLikeType,
    x2: TensorLikeType,
    p: NumberType = 2.0,
    eps: NumberType = 1e-6,
    keepdim=False,
) -> TensorLikeType:
    return torch.linalg.vector_norm(x1 - x2 + eps, ord=p, dim=-1, keepdim=keepdim)


@register_decomposition(aten.pdist)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def pdist(a: TensorLikeType, p: float = 2) -> TensorLikeType:
    torch._check(a.ndim == 2, lambda: f"pdist only supports 2D tensors, got: {a.ndim}D")
    torch._check(p >= 0, lambda: "pdist only supports non-negative p values")
    # For p == 2 we can use an efficient implementation, but other values of p
    # require creating a much bigger tensor for an intermediate step
    if p == 2:
        aTa = torch.mm(a, a.T)
        aTa_diag = torch.diag(aTa)
        t = torch.sqrt(torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0))
    else:
        t = torch.linalg.vector_norm(a.unsqueeze(1) - a, ord=p, dim=2)
    i = torch.triu_indices(t.shape[0], t.shape[1], offset=1, device=a.device)
    return t.flatten().index_select(0, i[0] * t.shape[0] + i[1])


# Needed as aten.{celu_,elu_...} exist (even if they don't have the in-place kwarg)
celu_ = _make_inplace(celu)
elu_ = _make_inplace(elu)
mish_ = _make_inplace(mish)
selu_ = _make_inplace(selu)
threshold_ = _make_inplace(threshold)
