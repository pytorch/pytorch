import inspect
from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Dict, List, Sequence, Union

import torch
import torch.library
from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket
from torch.utils._pytree import tree_map

__all__ = [
    "decomposition_table",
    "pre_autograd_decomposition_table",
    "meta_table",
    "register_decomposition",
    "get_decompositions",
    "core_aten_decompositions",
]


# TODO: relax key type here; torch registrations should be possible to; but
# right now this type is accurate
global_decomposition_table: Dict[
    str, Dict[torch._ops.OperatorBase, Callable]
] = defaultdict(dict)

decomposition_table = global_decomposition_table["post_autograd"]
pre_autograd_decomposition_table = global_decomposition_table["pre_autograd"]
meta_table = global_decomposition_table["meta"]


def _add_op_to_registry(registry, op, fn):
    """
    This is an internal API for adding an op to the decomposition table.

    If op is OpOverload, it will be added to the registry directly.
    If op is OpOverloadPacket, all the valid op_overloads in the packet will be added to the registry.
    """
    overloads: List[Union[torch._ops.OperatorBase]] = []
    if isinstance(op, HigherOrderOperator):
        # There's no concept of overloads for HigherOrderOperator
        registry[op] = fn
        return
    elif isinstance(op, OpOverload):
        overloads.append(op)
    else:
        assert isinstance(op, OpOverloadPacket)
        for ol in op.overloads():
            overloads.append(getattr(op, ol))

    for op_overload in overloads:
        if op_overload in registry:
            raise RuntimeError(f"duplicate registrations for {op_overload}")
        # TorchScript dumps a bunch of extra nonsense overloads
        # which don't have corresponding dispatcher entries, we need
        # to filter those out, e.g aten.add.float_int
        if torch._C._dispatch_has_kernel(op_overload.name()):
            registry[op_overload] = fn


def _convert_out_params(f):
    sig = inspect.signature(f)
    out_annotation = f.__annotations__.get("out")
    # Hack to detect when out is a Tuple. There seems to be no pretty way of doing this
    fn = f
    if out_annotation and getattr(out_annotation, "__origin__", None) is tuple:
        out_names = sig.return_annotation._fields
        # If out is a tuple, we need to register a function that unpacks all the out
        # elements as this is what native_functions.yaml expects

        @wraps(f)
        def _fn(*args, **kwargs):
            out_kwargs = tuple(kwargs.pop(o, None) for o in out_names)
            # Either all of the out kwargs are set or none of them
            is_none = out_kwargs[0] is None
            assert all((o is None) == is_none for o in out_kwargs)
            return f(*args, **kwargs, out=None if is_none else out_kwargs)

        out_params = [
            inspect.Parameter(
                o,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=t,
            )
            for o, t in zip(out_names, out_annotation.__args__)
        ]
        # Drop the out parameter and concatenate the new kwargs in the signature
        params = chain((v for k, v in sig.parameters.items() if k != "out"), out_params)
        _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params, return_annotation=sig.return_annotation  # type: ignore[arg-type]
        )
        # Drop the out parameter and concatenate the new kwargs in the annotations
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != "out"}
        for o in out_params:
            _fn.__annotations__[o.name] = o.annotation

        fn = _fn

    return fn


def register_decomposition(
    aten_op, registry=None, *, type="post_autograd", unsafe=False
):
    """
    A decorator to register a function as a decomposition to the Python
    decomposition table.  Use it like this::

        @register_decomposition(torch.ops.aten.clamp_min)
        def clamp_min(x):
            return torch.clamp(self, min=min)

    If you are writing a new decomposition, consider contributing it
    directly to PyTorch in torch._decomp.decompositions.

    This API is experimental; we are almost certainly going to extend
    the API when we make decompositions eligible for use in transforms (e.g.,
    autograd) and not just backend tracing, where we then need to know if a
    decomposition can be used to simulate a transform.

    By default, we also will register it to the Meta key of dispatcher,
    and replace the c++ Meta implementation if there is already one.

    unsafe kwarg is for reuse of this function for registering non-function
    things
    """

    assert type in {"post_autograd", "pre_autograd", "meta"}

    def decomposition_decorator(fn: Callable) -> Callable:
        if not unsafe:
            fn = _convert_out_params(fn)

        nonlocal registry
        if registry is None:
            registry = global_decomposition_table[type]

        def register(op):
            _add_op_to_registry(registry, op, fn)

        # To handle allowing multiple aten_ops at once
        tree_map(register, aten_op)
        return fn

    return decomposition_decorator


def get_decompositions(
    aten_ops: Sequence[Union[torch._ops.OperatorBase, OpOverloadPacket]],
    type: str = "post_autograd",
) -> Dict[torch._ops.OperatorBase, Callable]:
    """
    Retrieve a dictionary of decompositions corresponding to the list of
    operator overloads and overload packets passed as input.  Overload
    packets will include all decomposed overloads in the packet.  If there is
    no decomposition for a requested operator, it is silently ignored.

    This API is experimental; we are almost certainly going to give an alternate,
    more recommended formulation, where a user provides the set of operators
    they know how to implement, and we provide decompositions for everything
    not in this set.
    """
    assert type in {"post_autograd", "pre_autograd", "meta"}

    registry = global_decomposition_table[type]
    packets_to_overloads = defaultdict(list)
    for opo in registry:
        if isinstance(opo, (OpOverload, OpOverloadPacket)):
            packets_to_overloads[opo.overloadpacket].append(opo)
    decompositions: Dict[torch._ops.OperatorBase, Callable] = {}
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket) and op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = registry[op_overload]
        elif isinstance(op, (torch._ops.OperatorBase)) and op in registry:
            decompositions[op] = registry[op]
    return decompositions


def remove_decompositions(
    decompositions: Dict[torch._ops.OperatorBase, Callable],
    aten_ops: Sequence[Union[OpOverload, OpOverloadPacket]],
) -> None:
    """
    Given a dictionary of decompositions obtained from get_decompositions(), removes
    operators associated with a list of operator overloads and overload packets passed
    as input. If the decomposition dictionary does not contain a decomposition that is
    specified to be removed, it is silently ignored.
    """
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket):
            for overload_name in op.overloads():
                opo = getattr(op, overload_name)
                decompositions.pop(opo, None)
        elif isinstance(op, OpOverload):
            decompositions.pop(op, None)


# populate the table
import torch._decomp.decompositions
import torch._refs


# See NOTE [Core ATen Ops]
#
# list was copied from torch/_inductor/decomposition.py
# excluding decompositions that results in prim ops
# Resulting opset of decomposition is core aten ops
def core_aten_decompositions() -> Dict[torch._ops.OperatorBase, Callable]:
    aten = torch.ops.aten
    return get_decompositions(
        [
            aten.addcdiv,
            aten.addcdiv_,
            aten.addcmul,
            aten.addcmul_,
            aten.addr,
            aten.affine_grid_generator,
            aten.aminmax,
            aten.arange.default,
            aten.arange.start,
            aten.avg_pool2d_backward,
            aten.binary_cross_entropy,
            aten.binary_cross_entropy_backward,
            aten.binary_cross_entropy_with_logits,
            aten.celu,
            aten.celu_,
            aten.clamp_max,
            aten.clamp_min,
            aten.col2im,
            aten.count_nonzero,
            aten.cudnn_batch_norm,
            aten.cudnn_batch_norm_backward,
            aten.deg2rad,
            aten.deg2rad_,
            aten.detach,
            aten.diag_embed,
            aten.diagonal_backward,
            aten.dot,
            aten.vdot,
            aten.elu,
            aten.elu_,
            aten.elu_backward,
            aten._embedding_bag,
            aten.embedding_dense_backward,
            aten.empty_like,
            aten._euclidean_dist.default,
            aten.expand_as,
            aten.eye,
            aten.fill,
            aten.fill_,
            aten.frac,
            aten.frac_,
            aten._fused_moving_avg_obs_fq_helper,
            aten.gelu_,
            aten.gelu_backward,
            aten.glu_backward,
            aten.hardshrink,
            aten.hardsigmoid,
            aten.hardsigmoid_,
            aten.hardsigmoid_backward,
            aten.hardswish,
            aten.hardswish_,
            aten.hardswish_backward,
            aten.hardtanh_,
            aten.hardtanh_backward,
            aten.heaviside,
            aten.heaviside_,
            aten.huber_loss,
            aten.huber_loss_backward,
            aten.im2col,
            aten.index_add,
            aten.index_add_,
            aten.index_copy,
            aten.index_copy_,
            aten.index_fill,
            aten.index_fill_,
            aten.isneginf,
            aten.isposinf,
            aten.l1_loss,
            aten.leaky_relu_,
            aten.leaky_relu_backward,
            aten.lerp,
            aten.lerp_,
            aten.linspace,
            aten.logaddexp,
            aten.logaddexp2,
            aten.logit,
            aten.logit_,
            aten.logit_backward,
            aten.log_sigmoid_backward,
            aten.log_sigmoid_forward,
            aten._log_softmax_backward_data,
            aten.logspace,
            aten.logsumexp.default,
            aten.masked_fill,
            aten.masked_fill_,
            aten.mish,
            aten.mish_,
            aten.mse_loss,
            aten.mse_loss_backward,
            aten.multi_margin_loss,
            aten.multilabel_margin_loss_forward,
            aten.mv,
            aten.mvlgamma,
            aten.mvlgamma_,
            aten.nansum,
            aten.nan_to_num,
            aten.nan_to_num_,
            aten.narrow,
            aten.native_batch_norm_backward,
            aten.native_dropout_backward,
            aten.native_group_norm_backward,
            aten.native_layer_norm_backward,
            aten.new_empty,
            aten.new_full,
            aten.new_ones,
            aten.new_zeros,
            aten.nll_loss_backward,
            aten.nll_loss_forward,
            aten.norm,
            aten.ones,
            aten.ones_like,
            aten._prelu_kernel,
            aten._prelu_kernel_backward,
            aten._reshape_alias,
            aten.rad2deg,
            aten.rad2deg_,
            aten.renorm,
            aten.renorm_,
            aten.rot90,
            aten.rrelu_with_noise,
            aten.rrelu_with_noise_,
            aten.rsub.Scalar,
            aten.rsub.Tensor,
            aten._scaled_dot_product_flash_attention.default,
            aten.select_backward,
            aten.select_scatter,
            aten.sgn,
            aten.sgn_,
            aten.sigmoid_backward,
            aten.silu,
            aten.silu_,
            aten.silu_backward,
            aten.sinc,
            aten.sinc_,
            aten.slice_backward,
            aten.smooth_l1_loss,
            aten.smooth_l1_loss_backward,
            aten.soft_margin_loss,
            aten.soft_margin_loss_backward,
            aten._softmax_backward_data,
            aten.softplus,
            aten.softplus_backward,
            aten.softshrink,
            aten.special_entr,
            aten.special_log_ndtr,
            aten.special_xlog1py,
            aten.std.correction,
            aten.stack,
            aten.t,
            aten.tanh_backward,
            aten.threshold,
            aten.threshold_,
            aten.threshold_backward,
            aten.trace,
            aten.transpose.int,
            aten.tril,
            aten.tril_,
            aten.triu,
            aten.triu_,
            aten.unfold_backward,
            aten.unfold_copy,
            aten._unsafe_index,
            aten.unsafe_split.Tensor,
            aten.upsample_bilinear2d,
            aten.upsample_nearest2d_backward,
            aten.view_as_complex,
            aten.xlogy,
            aten.xlogy_,
            aten.zero,
            aten.zero_,
            aten.zeros,
            aten.zeros_like,
        ]
    )
