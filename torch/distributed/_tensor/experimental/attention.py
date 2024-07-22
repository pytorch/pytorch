# Copyright (c) Meta Platforms, Inc. and affiliates

import collections
import contextlib
import importlib
import itertools
import logging
import types
import weakref
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
import torch.nn.functional as F
from torch import nn
from torch.distributed._tensor import distribute_module, DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle


aten = torch.ops.aten
logger = logging.getLogger(__name__)
_rerun_forward = False
_merge_per_step = True
_convert_to_f32 = True
_enable_load_balance = True


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    """
    Calculate is_causal behavior for each KV block. The attention can either be
    calculated in full, not at all or with the causal mask applied.
    """
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    source_rank = (rank - i) % world_size
    if source_rank < rank:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP


class SDPAMerger:
    def __init__(self, merge_per_step: bool, convert_to_f32: bool):
        self._merge_per_step = merge_per_step
        self._chunks: List[torch.Tensor] = []
        self._logsumexps: List[torch.Tensor] = []
        self._out: Optional[torch.Tensor] = None
        self._softmax_lse: Optional[torch.Tensor] = None
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = None
        self._lse_dtype = None

    def _merge_one(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        lse = lse.unsqueeze(dim=-1)
        if self._softmax_lse is None:
            self._softmax_lse = lse
            self._out = out
        else:
            new_lse = self._softmax_lse + torch.log(
                1 + torch.exp(lse - self._softmax_lse)
            )
            self._out = (
                torch.exp(self._softmax_lse - new_lse) * self._out
                + torch.exp(lse - new_lse) * out
            )
            self._softmax_lse = new_lse

    def _merge_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(self._chunks) == len(self._logsumexps)

        # LSE may be padded in the sequence dimension such as with
        # memory efficient attention.
        seq_len = self._chunks[0].size(2)
        logsumexps = [lse[:, :, :seq_len] for lse in self._logsumexps]
        softmax_lse = logsumexps[0]
        for lse in logsumexps[1:]:
            max_scale = torch.max(softmax_lse, lse)
            min_scale = torch.min(softmax_lse, lse)
            softmax_lse = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
        softmax_lse = torch.stack([lse.exp() for lse in logsumexps]).sum(dim=0).log_()

        out = []
        for chunk, chunk_lse in zip(self._chunks, logsumexps):
            softmax_lse_corrected = torch.exp(chunk_lse - softmax_lse)
            out_corrected = chunk * softmax_lse_corrected.unsqueeze(-1)
            out.append(out_corrected)
        out = torch.stack(out).sum(dim=0)
        return out, softmax_lse

    def step(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype

        if self._convert_to_f32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

        if self._merge_per_step:
            self._merge_one(out, lse)
        else:
            self._chunks.append(out)
            self._logsumexps.append(lse)

    def results(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._merge_per_step:
            out, softmax_lse = self._out, self._softmax_lse.squeeze(-1)
        else:
            out, softmax_lse = self._merge_all()

        return out.to(self._out_dtype), softmax_lse.to(self._lse_dtype)


def _scaled_dot_product_ring_flash_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    return _templated_ring_attention(
        mesh,
        aten._scaled_dot_product_flash_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
    )


def _scaled_dot_product_ring_efficient_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    compute_log_sumexp: bool = True,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")
    if not compute_log_sumexp:
        raise NotImplementedError("compute_log_sumexp must be set")

    return _templated_ring_attention(
        mesh,
        aten._scaled_dot_product_efficient_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        scale=scale,
        compute_log_sumexp=compute_log_sumexp,
    )


def _scaled_dot_product_ring_cudnn_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = True,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    if not return_debug_mask:
        raise NotImplementedError("return_debug_mask must be set")

    return _templated_ring_attention(
        mesh,
        aten._scaled_dot_product_cudnn_attention,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        dropout_p=dropout_p,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )


class AttentionOp(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> Tuple[torch.Tensor, ...]:
        ...


def _ring_rotate(
    block: torch.Tensor, pg: dist.ProcessGroup, send_to_next: bool
) -> torch.Tensor:
    size = dist.get_world_size(pg)
    dsts = (
        list(range(1, size)) + [0]
        if send_to_next
        else [size - 1] + list(range(0, size - 1))
    )
    return ft_c.permute_tensor(block, dsts, pg)


def _templated_ring_attention(
    mesh: DeviceMesh,
    op: AttentionOp,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    **kwargs: object,
) -> Tuple[torch.Tensor, ...]:
    """
    This is a generalized ring attention implementation that can support multiple attention ops.

    Parameters
    ----------
    op:
        The attention op to use
    *args:
        additional args are passed to the op
    **kwargs:
        additional kwargs are passed to the op

    Returns
    -------
    out:
        The merged attention output
    softmax_lse:
        The logsumexp of the merged attention output
    """
    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )

    if isinstance(mesh, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = mesh
    else:
        pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "process group must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)

    next_kv = None

    chunks = []
    logsumexps = []

    # Without making key and value contiguous(), the lose curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = SDPAMerger(_merge_per_step, _convert_to_f32)

    rest: Tuple[Any, ...] = tuple()

    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = next_kv.wait()
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = _ring_rotate(next_kv, pg, send_to_next=True)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            out, logsumexp, *rest = op(
                query,
                key,
                value,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )

            sdpa_merger.step(out, logsumexp)

    return *sdpa_merger.results(), *rest


def sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    if op_call == aten._scaled_dot_product_flash_attention.default:
        local_results = _scaled_dot_product_ring_flash_attention(
            op_info.mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    else:
        raise NotImplementedError

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def sdpa_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    # Redistribute grad_output tensor to the same placement as output tensor
    args = list(args)
    assert isinstance(args[0], DTensor) and isinstance(args[4], DTensor)
    args[0] = args[0].redistribute(args[4].device_mesh, args[4].placements)
    args = tuple(args)

    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    local_results = _scaled_dot_product_ring_flash_attention_backward(
        op_info.mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _get_rerun_gradients(
    grad_out: torch.Tensor,
    logsumexp: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Keep this implementation for verification purpose.
    # TODO(chienchin): remove this implementation after more E2E
    # verification.
    rest: Tuple[Any, ...] = tuple()
    output, logsumexp_, *rest = aten._scaled_dot_product_flash_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal
    )
    softmax_lse_corrected = torch.exp(logsumexp_ - logsumexp)
    grad_out_ = grad_out * softmax_lse_corrected.conj().unsqueeze(-1).to(grad_out.dtype)
    return grad_out_, logsumexp_, output


def _templated_ring_attention_backward(
    mesh: DeviceMesh,
    op: AttentionOp,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)
    next_kv = None
    next_grad_kv = None
    rest: Tuple[Any, ...] = tuple()

    for i in range(size):
        if i == 0:
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = _ring_rotate(next_kv, pg, send_to_next=True)
            grad_query = torch.zeros_like(query).to(torch.float32)
            grad_key = None
            grad_value = None

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            if _rerun_forward:
                grad_out_, logsumexp_, output = _get_rerun_gradients(
                    grad_out,
                    logsumexp,
                    query,
                    key,
                    value,
                    kwargs["dropout_p"],
                    is_causal_behavior.value,
                )
            else:
                grad_out_, logsumexp_, output = (grad_out, logsumexp, out)

            grad_query_, grad_key_, grad_value_, *rest = op(
                grad_out=grad_out_,
                query=query,
                key=key,
                value=value,
                out=output,
                logsumexp=logsumexp_,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
        else:
            grad_query_ = torch.zeros_like(query)
            grad_key_ = torch.zeros_like(key)
            grad_value_ = torch.zeros_like(value)

        buffer = next_kv if i == 0 else next_grad_kv
        buffer = buffer.wait()
        pointer = 0

        # Get the new key and value for the (i + 1) round.
        if i != size - 1:
            key = buffer[pointer : pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer : pointer + value.numel()].reshape(value.shape)
            pointer += value.numel()

        # Get the grad key and grad value for the i round.
        if i == 0:
            grad_key = grad_key_
            grad_value = grad_value_
        else:
            grad_key = buffer[pointer : pointer + grad_key_.numel()].reshape(
                grad_key_.shape
            )
            pointer += grad_key_.numel()
            grad_value = buffer[pointer : pointer + grad_value_.numel()].reshape(
                grad_value_.shape
            )
            grad_key += grad_key_
            grad_value += grad_value_

        # Send the key, value, grad key, and grad value to the next rank.
        if i >= size - 2:
            next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
        else:
            next_grad_kv = torch.cat(
                [
                    key.flatten(),
                    value.flatten(),
                    grad_key.flatten(),
                    grad_value.flatten(),
                ]
            )

        next_grad_kv = _ring_rotate(next_grad_kv, pg, send_to_next=True)
        grad_query += grad_query_

    grad_query = grad_query.to(query.dtype)
    next_grad_kv = next_grad_kv.wait()
    grad_key = next_grad_kv[: grad_key_.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_value_.numel() :].reshape(grad_value.shape)
    return (
        grad_query,
        grad_key,
        grad_value,
        *rest,
    )


def _scaled_dot_product_ring_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    return _templated_ring_attention_backward(
        mesh,
        aten._scaled_dot_product_flash_attention_backward.default,
        grad_out=grad_out,
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        is_causal=is_causal,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: sdpa_backward_handler,
}


# Following APIs are experimental to allow users to enable CP.


@contextlib.contextmanager
def attention_context_parallel() -> Generator[None, None, None]:
    """
    This is a context manager that force enables attention context parallel
    optimizations for all scaled_dot_product_attention ops.

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """
    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}

    yield

    DTensor._op_dispatcher._custom_op_handlers = old_handlers


class AttentionContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer.

    This will work for nn.MultiHeadedAttention and custom attention layers that
    call F.scaled_dotproduct_attention with a simliar signature.

    This expects the `forward` method consumes either:

    * a single tensor for self attention
    * one argument for each of: query, key, value

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """

    # use a weakref dictionary to store context managers for each nn.Module
    _CONTEXT_MANAGERS: "weakref.WeakKeyDictionary[nn.Module, Any]" = (
        weakref.WeakKeyDictionary()
    )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                f"{type(device_mesh)} is not supported by {type(self)} yet."
            )

        if not device_mesh.ndim == 1:
            raise ValueError

        return distribute_module(
            module,
            device_mesh,
            input_fn=self._input_fn,  # type: ignore[arg-type]
            output_fn=self._output_fn,  # type: ignore[arg-type]
        )

    @classmethod
    def _input_fn(
        cls,
        module: nn.Module,
        inputs: Tuple[Union[torch.Tensor, int, float], ...],
        device_mesh: DeviceMesh,
    ) -> Tuple[Union[torch.Tensor, int, float], ...]:
        # TODO(d4l3k); this should be Shard(2), need to fix Linear layer rules
        placement = [Replicate()]

        def backward_hook(grad: torch.Tensor) -> None:
            if module in cls._CONTEXT_MANAGERS:
                cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
                del cls._CONTEXT_MANAGERS[module]

        # convert inputs to DTensor
        inp = []
        for input in inputs:
            if isinstance(input, torch.Tensor) and not isinstance(input, DTensor):
                input = DTensor.from_local(
                    input, device_mesh, placement, run_check=False
                )

            if isinstance(input, torch.Tensor) and input.requires_grad:
                input.register_hook(backward_hook)

            inp.append(input)

        manager = attention_context_parallel()
        manager.__enter__()
        cls._CONTEXT_MANAGERS[module] = manager

        return tuple(inp)

    @classmethod
    def _output_fn(
        cls,
        module: nn.Module,
        outputs: Union[torch.Tensor, Tuple[Union[torch.Tensor, int, float], ...]],
        device_mesh: DeviceMesh,
    ) -> Union[
        Union[torch.Tensor, int, float], Tuple[Union[torch.Tensor, int, float], ...]
    ]:
        cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
        del cls._CONTEXT_MANAGERS[module]

        def backward_hook(grad: torch.Tensor) -> None:
            if module not in cls._CONTEXT_MANAGERS:
                manager = attention_context_parallel()
                manager.__enter__()
                cls._CONTEXT_MANAGERS[module] = manager

        # back to local tensor
        out = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output

            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(backward_hook)

            out.append(output)

        if isinstance(outputs, torch.Tensor):
            return out[0]

        return tuple(out)


_original_functions: Dict[Callable, Callable] = {}
_wrapper_functions: Dict[Callable, Callable] = {}
_replaced_objs: collections.defaultdict[
    Callable, Set[Tuple[types.ModuleType, str]]
] = collections.defaultdict(set)


def _distribute_function(
    fn: Callable,
    fn_module: types.ModuleType,
    fn_callers: List[nn.Module],
    device_mesh: DeviceMesh,
    input_fn: Optional[Callable] = None,
    output_fn: Optional[Callable] = None,
) -> None:
    """
    ``distribute_function`` is an experimental API that allows users to "distribute"
    the inputs and outputs of a function. Similar to ``distribute_module``, this API
    installs hooks to the ``fn`` to convert the inputs and outputs. There are two
    major differences between ``distribute_function`` and ``distribute_module``.
    First, a function does not have parammeters and buffers, as a result,
    ``distribute_function`` itself won't convert any tensors but simply install the
    input and output hooks.  The tnesor conversion will happen in the hooks.
    Another difference is an nn.Module subclass can have several instances and each
    instance be fed into ``distribute_module`` independently with affecting other
    instance. On the other hand, function is a singleton object. So if a function
    is distributed by ``distribute_function`` all subsequent calls to the function
    will invoke the installed hooks.

    Args:
        fn (Callable): the function to be distributed.
        fn_module (types.ModuleType): the Python module that the function is declared.
            e.g., if ``fn`` is ``torch.nn.functional.scaled_dot_product_attention``,
            ``fn_module`` is ``torch.nn.functional``.
        fn_callers (nn.Module): the nn.Module that calls ``fn``.
        device_mesh (:class:`DeviceMesh`): the device mesh that will be used by the
            input and output hooks to distribute the tensors.
        input_fn (Optioinal[Callable]): the hook to distribute or convert the input
            arguments of ``fn``.
        output_fn (Optioinal[Callable]): the hook to distribute or convert the output
            arguments of ``fn``.
    """

    def wrapper(target_fn, input_fn, output_fn):
        def inner_fn(*args, **kwargs):
            if input_fn is not None:
                args, kwargs = input_fn(device_mesh, *args, **kwargs)
            output = target_fn(*args, **kwargs)
            if output_fn is not None:
                output = output_fn(device_mesh, output)
            return output

        return inner_fn

    def setattr_(module, obj_name, obj, new_obj):
        setattr(module, obj_name, new_obj)
        global _replaced_objs
        _replaced_objs[obj].add((module, obj_name))

    global _original_functions
    global _wrapper_functions
    if fn in _original_functions:
        wrapper_func = _original_functions[fn]
        original_func = fn
    elif fn in _wrapper_functions:
        wrapper_func = fn
        original_func = _wrapper_functions[fn]
    else:
        original_func = fn
        wrapper_func = wrapper(fn, input_fn, output_fn)
        setattr_(fn_module, fn.__name__, fn, wrapper_func)

    for nn_module in fn_callers:
        fn_caller_module = importlib.import_module(nn_module.__module__)
        for obj_name in dir(fn_caller_module):
            obj = getattr(fn_caller_module, obj_name)
            if obj == original_func:
                setattr_(fn_caller_module, obj_name, obj, wrapper_func)


_function_cm: weakref.WeakKeyDictionary[Callable, Any] = weakref.WeakKeyDictionary()


def enable_context_parallel(
    seq_dim: int,
    callers: List[nn.Module],
    device_mesh: DeviceMesh,
    enable_load_balance: bool = True,
) -> None:
    """
    This is an experimental API to enable context parallelism for
    ``torch.nn.functional.scaled_dot_product_attention``. This API assumes
    that the q, k, v are already sharded on the ``seq_dim`` dimension and
    will install hook to convert the q, k, v to the DTensors.

    This API will change ``scaled_dot_product_attention`` in ``torch.nn.functional``
    (short as ``F``) to a wrapped function. So any subsequent call to
    ``F.scaled_dot_product_attention`` will be redirected to the wrapped function.

    Note that it is important to include all the modules that call SDPA in the
    ``callers`` list. This can avoid the incorrect wrapping if the model code uses
    ``from ... import ..." to import SDPA.

    Args:
        seq_dim (int): the sequence dimension for q, k, v.
        callers (List[nn.Module]): the nn.Modules that call ``scaled_dot_product_attention``.
        device_mesh (:class:`DeviceMesh`, optional): the device mesh for context
            parallelism.
    """

    def attention_input_fn(device_mesh: DeviceMesh, *args, **kwargs):
        placement = [Shard(seq_dim)]
        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                arg = DTensor.from_local(arg, device_mesh, placement, run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0 : len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args) :]))
        if not _function_cm:
            _function_cm[attention_input_fn] = attention_context_parallel()
            manager = _function_cm[attention_input_fn]
            manager.__enter__()

        return new_args, new_kwargs

    def attention_output_fn(device_mesh, outputs):
        new_outputs = []
        # TODO: Convert to 1D DTensor if 2D is applied.
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output
            new_outputs.append(output)

        if isinstance(outputs, torch.Tensor):
            return new_outputs[0]

        return tuple(new_outputs)

    _distribute_function(
        F.scaled_dot_product_attention,
        F,
        callers,
        device_mesh,
        attention_input_fn,
        attention_output_fn,
    )


@contextlib.contextmanager
@torch.no_grad()
def context_parallel_buffers(
    cp_rank: int,
    cp_world_size: int,
    buffers: List[torch.Tensor],
    seq_dims: List[int],
    keep_orig_buffers: List[bool],
):
    if cp_world_size == 1:
        yield
        return

    for buffer, seq_dim, keep_orig_buffer in zip(buffers, seq_dims, keep_orig_buffers):
        if keep_orig_buffer:
            orig_buffer = getattr(buffer, "_orig_buffer", None)
            if orig_buffer is None:
                orig_buffer = buffer.clone()
                buffer._orig_buffer = orig_buffer
        else:
            orig_buffer = buffer.clone()

        shape = buffer.shape
        seq_len = shape[seq_dim]
        chunk_seq_len = seq_len // cp_world_size
        view_slices = tuple(
            slice(0, shape[i])
            if i != seq_dim
            else slice(cp_rank * chunk_seq_len, (cp_rank + 1) * chunk_seq_len)
            for i in range(len(shape))
        )
        buffer_view = orig_buffer[view_slices]
        buffer.resize_(buffer_view.shape)
        buffer.copy_(buffer_view)
        if not keep_orig_buffer:
            del buffer_view
            del orig_buffer

    yield

    for buffer, seq_dim, keep_orig_buffer in zip(buffers, seq_dims, keep_orig_buffers):
        if not keep_orig_buffer:
            continue

        buffer.resize_(buffer._orig_buffer.shape)
        buffer.copy_(buffer._orig_buffer)
