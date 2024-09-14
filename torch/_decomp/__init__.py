# mypy: allow-untyped-defs
import inspect
from collections import defaultdict
from functools import lru_cache, partial, wraps
from itertools import chain
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec

import torch
import torch.library
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.utils import _pytree as pytree


__all__ = [
    "decomposition_table",
    "pre_autograd_decomposition_table",
    "meta_table",
    "register_decomposition",
    "get_decompositions",
    "core_aten_decompositions",
    "_decomp_table_to_post_autograd_aten",
    "_special_op_to_preserve_cia",
]

_T = TypeVar("_T")
_P = ParamSpec("_P")

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
    out_annotation = f.__annotations__.get("out")

    # If there are no out params, do not wrap the function.
    if not out_annotation:
        return f

    # Hack to detect when out is a Tuple. There seems to be no pretty way of doing this
    if getattr(out_annotation, "__origin__", None) is tuple:
        sig = inspect.signature(f)
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

        # Propagate that this function is wrapped by `out_wrapper`
        _fn._torch_decompositions_out_wrapper = f._torch_decompositions_out_wrapper  # type: ignore[attr-defined]

        return _fn

    # Alternatively, there may be a single tensor out parameter with a name
    # other than "out". This will need special treatment and is indicated by an
    # annotation, which we will remove here so it is not exposed after wrapping.
    custom_out_param_name = f.__annotations__.pop(CustomOutParamAnnotation, None)
    if custom_out_param_name:

        @wraps(f)
        def _fn(*args, **kwargs):
            out_kwarg = kwargs.pop(custom_out_param_name, None)
            return f(*args, **kwargs, out=out_kwarg)

        out_param = inspect.Parameter(
            custom_out_param_name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=out_annotation,
        )

        # Drop the out parameter and concatenate the new kwarg in the signature
        sig = inspect.signature(f)
        params = chain(
            (v for k, v in sig.parameters.items() if k != "out"), (out_param,)
        )
        _fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=params, return_annotation=sig.return_annotation  # type: ignore[arg-type]
        )

        # Drop the out parameter and concatenate the new kwargs in the annotations
        _fn.__annotations__ = {k: v for k, v in f.__annotations__.items() if k != "out"}
        _fn.__annotations__[out_param.name] = out_param.annotation

        return _fn

    return f


def register_decomposition(
    aten_op, registry=None, *, type="post_autograd", unsafe=False
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
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

    def decomposition_decorator(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        orig_fn = fn
        if not unsafe:
            fn = _convert_out_params(fn)

        nonlocal registry
        if registry is None:
            registry = global_decomposition_table[type]

        def register(op):
            _add_op_to_registry(registry, op, fn)

        # To handle allowing multiple aten_ops at once
        pytree.tree_map_(register, aten_op)
        return orig_fn

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


# Our strategy for deciding if we can preserve a op is following:
# 1. The op should be known statically that it is functional
# 2. If it is maybe aliasing, we decompose because we must know if an op
#    is mutating or aliasing.
# TODO (tmanlaibaatar) make this utility function and share it with functional_tensor
# decomp part. (https://github.com/pytorch/pytorch/issues/129431)
def _check_valid_to_preserve(op_overload):
    if op_overload in FunctionalTensor.maybe_aliasing_or_mutating_ops:
        return False
    if op_overload in FunctionalTensor.metadata_fns:
        return False

    alias_info = len(
        [i for i in op_overload._schema.arguments if i.alias_info is not None]
    )

    is_mutating_or_aliasing = alias_info != 0 or op_overload._schema.is_mutable

    if is_mutating_or_aliasing:
        return False

    if not torch._C._dispatch_has_kernel(op_overload.name()):
        return False

    return True


def _is_cia_op(op: "OpOverload") -> bool:
    return (
        torch._C._dispatch_has_kernel_for_dispatch_key(
            op.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        )
        or torch._C.DispatchKey.CompositeImplicitAutograd in op.py_kernels
    )


@lru_cache(maxsize=1)
def _collect_all_valid_cia_ops() -> Set["OperatorBase"]:
    """
    This is an util function that gets the all CIA functional ops.

    The algorithm is in 2 steps:
      1. We first query C++ dispatcher to get the list of CIA ops
         and then we call getattr on torch.ops.aten to lazily populate
         them.

      2. Sometimes, handful of ops have CIA registered in python dispatcher
         but not on the C++ side, these can't be caught at the first step.
         So we walk again to get the final list.

    Note that the output of this function should never be modified
    """
    # First step to lazily populate torch.ops.aten
    cia_ops = torch._C._dispatch_get_registrations_for_dispatch_key(
        "CompositeImplicitAutograd"
    )
    # Ignore quantized namespace ops
    cia_ops = [name[6:] for name in cia_ops if name.startswith("aten::")]
    # Materialize all CIA ops first
    for op in cia_ops:
        split_list = op.split(".")
        # Sometime overload could be missing
        assert len(split_list) == 1 or len(split_list) == 2
        op_name = split_list[0]
        op_overload_name = "default"
        if len(split_list) == 2:
            op_overload_name = split_list[1]

        _ = getattr(getattr(torch.ops.aten, op_name), op_overload_name)

    # Second step to finally compile the list of all valid ops
    cia_ops = set()
    for op in torch.ops.aten:
        op_packet = getattr(torch.ops.aten, op)
        for overload in op_packet.overloads():
            op_overload = getattr(op_packet, overload)
            if _check_valid_to_preserve(op_overload) and _is_cia_op(op_overload):
                cia_ops.add(op_overload)
    return cia_ops


def _get_decomp_for_cia(op):
    # [NOTE] Seperating out func.decompose
    # Ideally we should be able to just register func.decompose but
    # we can't as this decomp is gonna be registered to the py_impl.
    # As a result it will infinitely recurse. So we first check if the op
    # has py_impl entry for CIA and if it is we use that first. If not,
    # we register C++ query to py_impl.
    dk = torch._C.DispatchKey.CompositeImplicitAutograd
    if dk in op.py_kernels and not isinstance(op.py_kernels[dk], torch._C.DispatchKey):
        return op.py_kernels[dk]

    def _special_op_to_decompose_cia(*args, **kwargs):
        kernel = kwargs["kernel"]
        del kwargs["kernel"]
        # Can't call kernel.decompose due to infinite recursion as
        # we register this kernel to py_impl directly
        dk = torch._C.DispatchKey.CompositeImplicitAutograd
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            kernel.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            return kernel._op_dk(dk, *args, **kwargs)
        else:
            raise AssertionError(
                f"Expected {kernel} to have CompositeImplicitAutograd kernel"
            )

    return partial(_special_op_to_decompose_cia, kernel=op)


# See NOTE [Core ATen Ops]
#
# list was copied from torch/_inductor/decomposition.py
# excluding decompositions that results in prim ops
# Resulting opset of decomposition is core aten ops
def core_aten_decompositions() -> Dict[torch._ops.OperatorBase, Callable]:
    decomp_table = _core_aten_decompositions_post_autograd()

    # If it is fbcode change, we return the old decomposition list
    from torch._inductor import config

    if config.is_fbcode():
        return decomp_table

    aten = torch.ops.aten

    # We are deleting custom decomp in core_aten_decomp
    # for CIA ops but it should be fine technically
    # because this table is only meant to be used in export context
    # in which we really carefully control the decomp behaviour
    # In any case, C++ decomps should be preferred
    cia_ops_that_should_be_removed = [
        aten.all.dimname,
        aten.index_add.dimname,
        aten.index_copy.dimname,
        aten.index_fill.Dimname_Scalar,
        aten.index_fill.Dimname_Tensor,
        aten.norm.names_ScalarOpt_dim_dtype,
        aten.norm.names_ScalarOpt_dim,
        aten.silu_backward.default,
        aten.std.default,
        aten.std.dim,
        aten.std.names_dim,
        aten.std.correction_names,
        aten.std_mean.default,
        aten.std_mean.dim,
        aten.std_mean.names_dim,
        aten.std_mean.correction_names,
        aten.upsample_bilinear2d.vec,
        aten.upsample_trilinear3d.vec,
    ]

    for k in list(decomp_table.keys()):
        if k in cia_ops_that_should_be_removed:
            del decomp_table[k]

    for op in _collect_all_valid_cia_ops():
        decomp_table[op] = _get_decomp_for_cia(op)
    return decomp_table


# This table is a stop-gap table which replicates
# the old behaviour of post-dispatch IR.
# This table contains all functional CIA ops mapping
# to their default decomp. In old export, this will
# be decomposed implicitly.
def _decomp_table_to_post_autograd_aten():
    decomp_table = {}
    for k in _collect_all_valid_cia_ops():
        decomp_table[k] = _get_decomp_for_cia(k)
    return decomp_table


def _core_aten_decompositions_post_autograd() -> (
    Dict[torch._ops.OperatorBase, Callable]
):
    aten = torch.ops.aten
    # TODO Delete all mutating or CIA ops from this list
    return get_decompositions(
        [
            aten.addcdiv,
            aten.addcdiv_,
            aten.addcmul,
            aten.addcmul_,
            aten.addr,
            aten.affine_grid_generator,
            aten.alias_copy,
            aten.all,
            aten.aminmax,
            aten.arange.default,
            aten.arange.start,
            aten.avg_pool2d_backward,
            aten.baddbmm,
            aten.binary_cross_entropy,
            aten.binary_cross_entropy_backward,
            aten.binary_cross_entropy_with_logits,
            aten.block_diag,
            aten.celu,
            aten.celu_,
            aten.channel_shuffle,
            aten.clamp_max,
            aten.clamp_min,
            aten.col2im,
            aten.count_nonzero,
            aten.linalg_cross,
            aten.cudnn_batch_norm,
            aten.cudnn_batch_norm_backward,
            aten.miopen_batch_norm_backward,
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
            aten.expand_copy,
            aten.eye,
            aten.fill,
            aten.fill_,
            aten.floor_divide,
            aten.frac,
            aten.frac_,
            aten._fused_moving_avg_obs_fq_helper,
            aten.gelu_,
            aten.gelu_backward,
            aten.glu,
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
            aten.isin,
            aten.isneginf,
            aten.isposinf,
            aten.l1_loss,
            aten._lazy_clone,
            aten._test_parallel_materialize,
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
            aten.nll_loss2d_forward,
            aten.nll_loss2d_backward,
            aten.nll_loss_backward,
            aten.nll_loss_forward,
            aten.norm,
            aten.ones,
            aten.ones_like,
            aten.pixel_shuffle,
            aten.pixel_unshuffle,
            aten._prelu_kernel,
            aten._prelu_kernel_backward,
            aten._reshape_alias,
            aten.rad2deg,
            aten.rad2deg_,
            aten.reflection_pad1d,
            aten.reflection_pad1d_backward,
            aten.reflection_pad2d,
            aten.reflection_pad2d_backward,
            aten.reflection_pad3d,
            aten.reflection_pad3d_backward,
            aten.replication_pad1d,
            aten.replication_pad2d,
            aten.replication_pad3d,
            aten.renorm,
            aten.renorm_,
            aten.replication_pad2d,
            aten.resize_as,
            aten.roll,
            aten.rot90,
            aten.rrelu_with_noise,
            aten.rrelu_with_noise_,
            aten.rsub,
            aten._safe_softmax,
            aten._scaled_dot_product_flash_attention_for_cpu.default,
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
            aten.split.Tensor,
            aten.split_with_sizes_copy,
            aten.squeeze.default,
            aten.squeeze.dim,
            aten.std,
            aten.std_mean,
            aten.stack,
            aten.sum.default,
            aten.sum.out,
            aten.t,
            aten.t_copy,
            aten.take,
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
            aten.unbind,
            aten.unfold_backward,
            aten.unfold_copy,
            aten._unsafe_index,
            aten._unsafe_index_put,
            aten._unsafe_masked_index,
            aten._unsafe_masked_index_put_accumulate,
            aten.unsafe_split.Tensor,
            aten.unsafe_split_with_sizes,
            aten.unsqueeze_copy,
            aten._unsafe_view,
            aten.upsample_linear1d,
            aten.upsample_bilinear2d,
            aten.upsample_trilinear3d,
            aten.upsample_nearest2d_backward,
            aten.view_as_complex,
            aten.xlogy,
            aten.xlogy_,
            aten.zero,
            aten.zero_,
            aten.zeros,
            aten.zeros_like,
            aten._chunk_cat,
            aten._weight_norm_interface,
        ]
    )
