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
    DefaultDict,
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
_convert_to_f32 = True


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


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


class _SDPAMerger:
    """A class to help to merge the local SDPA result."""

    def __init__(self, convert_to_f32: bool):
        self._out: Optional[torch.Tensor] = None
        self._lse: Optional[torch.Tensor] = None
        self._convert_to_f32 = convert_to_f32
        self._out_dtype = torch.float32
        self._lse_dtype = torch.float32

    def _merge_one(self, block_out: torch.Tensor, block_lse: torch.Tensor) -> None:
        block_lse = block_lse.unsqueeze(dim=-1)
        if self._lse is None:
            self._lse = block_lse
            self._out = block_out
        else:
            new_lse = self._lse + torch.log(1 + torch.exp(block_lse - self._lse))
            self._out = (
                torch.exp(self._lse - new_lse) * self._out
                + torch.exp(block_lse - new_lse) * block_out
            )
            self._lse = new_lse

    def step(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        self._out_dtype = out.dtype
        self._lse_dtype = lse.dtype

        if self._convert_to_f32:
            out = out.to(torch.float32)
            lse = lse.to(torch.float32)

        self._merge_one(out, lse)

    def results(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._out is not None
        assert self._lse is not None
        out, lse = self._out, self._lse.squeeze(-1)
        return out.to(self._out_dtype), lse.to(self._lse_dtype)


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
    compute_log_sumexp: bool = True,
    dropout_p: float = 0.0,
    is_causal: bool = False,
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

    # Without making key and value contiguous(), the lose curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    key = key.contiguous()
    value = value.contiguous()

    sdpa_merger = _SDPAMerger(_convert_to_f32)

    rest: List[Any]
    out: torch.Tensor
    logsumexp: torch.Tensor

    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = _maybe_wait(next_kv)
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
    elif op_call == aten._scaled_dot_product_efficient_attention.default:
        local_results = _scaled_dot_product_ring_efficient_attention(
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
    # assert isinstance(args[0], DTensor) and isinstance(args[4], DTensor)
    # args[0] = args[0].redistribute(args[4].device_mesh, args[4].placements)
    args = tuple(args)

    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    logger.debug("Dispatching op_call: %s", op_info.schema)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    if op_call == aten._scaled_dot_product_flash_attention_backward.default:
        local_results = _scaled_dot_product_ring_flash_attention_backward(
            op_info.mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    elif op_call == aten._scaled_dot_product_efficient_attention_backward.default:
        local_results = _scaled_dot_product_ring_efficient_attention_backward(
            op_info.mesh,
            *op_info.local_args,  # type: ignore[arg-type]
            **op_info.local_kwargs,  # type: ignore[arg-type]
        )
    else:
        raise NotImplementedError(f"{op_call=}")

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _templated_ring_attention_backward(
    mesh: DeviceMesh,
    op: AttentionOp,
    grad_out: torch.Tensor,
    grad_out_name: str,
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
    rest: List[Any]
    grad_query_, grad_key_, grad_value_ = None, None, None

    grad_query = torch.zeros_like(query, dtype=torch.float32)
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value, dtype=torch.float32)

    key = key.contiguous()
    value = value.contiguous()
    for i in range(size):
        if next_kv is not None:
            buffer = _maybe_wait(next_kv)
            pointer = 0
            key = buffer[pointer : pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer : pointer + value.numel()].reshape(value.shape)
            pointer += value.numel()

        if i != size - 1:
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = _ring_rotate(next_kv, pg, send_to_next=True)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            kwargs[grad_out_name] = grad_out
            grad_query_, grad_key_, grad_value_, *rest = op(
                query=query,
                key=key,
                value=value,
                out=out,
                logsumexp=logsumexp,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
        else:
            grad_query_ = torch.zeros_like(query)
            grad_key_ = torch.zeros_like(key)
            grad_value_ = torch.zeros_like(value)

        # Get the grad key and grad value for the i round.
        if i > 0:
            buffer = next_grad_kv
            pointer = 0
            grad_key = buffer[pointer : pointer + grad_key.numel()].reshape(
                grad_key.shape
            )
            pointer += grad_key.numel()
            grad_value = buffer[pointer : pointer + grad_value.numel()].reshape(
                grad_value.shape
            )

        grad_key += grad_key_
        grad_value += grad_value_

        # Send the key, value, grad key, and grad value to the next rank.
        next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
        next_grad_kv = _ring_rotate(next_grad_kv, pg, send_to_next=True)
        grad_query += grad_query_

    assert grad_query is not None
    assert grad_key is not None
    assert grad_value is not None
    assert next_grad_kv is not None
    assert grad_key_ is not None
    assert grad_value_ is not None
    grad_query = grad_query.to(query.dtype)
    next_grad_kv = _maybe_wait(next_grad_kv).to(key.dtype)
    grad_key = next_grad_kv[: grad_key.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_value.numel() :].reshape(grad_value.shape)
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
        grad_out_name="grad_out",
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


def _scaled_dot_product_ring_efficient_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    dropout_p: float,
    grad_input_mask: Tuple[bool, ...],
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    return _templated_ring_attention_backward(
        mesh,
        aten._scaled_dot_product_efficient_attention_backward.default,
        grad_out=grad_out,
        grad_out_name="grad_out_",
        query=query,
        key=key,
        value=value,
        attn_bias=bias,
        out=out,
        logsumexp=logsumexp,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        grad_input_mask=grad_input_mask,
        is_causal=is_causal,
        scale=scale,
    )


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: sdpa_backward_handler,
    aten._scaled_dot_product_efficient_attention.default: sdpa_handler,
    aten._scaled_dot_product_efficient_attention_backward.default: sdpa_backward_handler,
}


# Following experimental APIs allow users to enable CP.


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
                    input.contiguous(), device_mesh, placement, run_check=False
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
_replaced_objs: DefaultDict[
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

    def wrapper(
        target_fn: Callable, input_fn: Optional[Callable], output_fn: Optional[Callable]
    ) -> Callable:
        def inner_fn(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
            if input_fn is not None:
                args, kwargs = input_fn(device_mesh, *args, **kwargs)
            output = target_fn(*args, **kwargs)
            if output_fn is not None:
                output = output_fn(device_mesh, output)
            return output

        return inner_fn

    def setattr_(
        module: types.ModuleType, obj_name: str, obj: Any, new_obj: Any
    ) -> None:
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


def enable_context_parallel(
    seq_dim: int,
    callers: List[nn.Module],
    device_mesh: DeviceMesh,
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

    def attention_input_fn(
        device_mesh: DeviceMesh, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        placement = [Shard(seq_dim)]
        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                arg = DTensor.from_local(arg, device_mesh, placement, run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0 : len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args) :]))
        return new_args, new_kwargs

    def attention_output_fn(device_mesh: DeviceMesh, outputs: Any) -> Any:
        new_outputs = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output
            new_outputs.append(output)

        if isinstance(outputs, torch.Tensor):
            return new_outputs[0]

        return tuple(new_outputs)

    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}
    _distribute_function(
        F.scaled_dot_product_attention,
        F,
        callers,
        device_mesh,
        attention_input_fn,
        attention_output_fn,
    )


def _get_sequence_shard(
    buffer: torch.Tensor, mesh: DeviceMesh, seq_dim: int
) -> torch.Tensor:
    return buffer.chunk(mesh.size(), dim=seq_dim)[mesh.get_local_rank()]


@contextlib.contextmanager
@torch.no_grad()
def context_parallel_buffers(
    mesh: Optional[DeviceMesh],
    buffers: List[torch.Tensor],
    seq_dims: List[int],
    restore_funcs: Optional[Tuple[Optional[Callable]]] = None,
) -> Generator[List[torch.Tensor], None, None]:
    if mesh is None:
        yield buffers
        return

    restore_funcs_tuple = (
        [None for _ in buffers] if restore_funcs is None else restore_funcs
    )
    original_buffers = []

    if len(buffers) != len(seq_dims):
        raise ValueError(
            "`seq_dims` must have the same number of elements as `buffers`."
        )

    if len(restore_funcs_tuple) != len(buffers):
        raise ValueError(
            "`restore_funcs_tuple` must be either None or have the same "
            "number of as `buffers`."
        )

    assert mesh is not None
    new_buffers = []
    for buffer, seq_dim, restore_func in zip(buffers, seq_dims, restore_funcs_tuple):
        original_buffers.append(buffer if restore_func else None)
        new_buffers.append(_get_sequence_shard(buffer, mesh, seq_dim))

    yield new_buffers

    for original_buffer, restore_func in zip(original_buffers, restore_funcs_tuple):
        if original_buffer is not None:
            assert restore_func is not None
            restore_func(original_buffer)
