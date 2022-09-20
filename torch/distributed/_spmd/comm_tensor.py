from dataclasses import dataclass
from functools import partial
from typing import Any, List, Tuple


import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import (
    _ProxyTensor,
    fetch_tensor_proxy,
    get_proxy_slots,
    set_proxy_slot,
    track_tensor_tree,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import (
    tree_flatten,
    tree_map,
    tree_map_only,
)


@dataclass
class _CommResult:
    # a custom type wrapping both inplace output tensor and work handle
    _tensor: torch.Tensor
    _work: torch.distributed._Work


def _wait_comm(comm_result: _CommResult):
    # This function is only used by tracing mode as a call_function node right
    # before consuming a collective result tensor.
    comm_result._work.wait()
    return comm_result._tensor


def _wrap_comm_result(result: Tuple[Any, Any]) -> Tuple[Any, Any]:
    def wrap(work, e):
        assert isinstance(e, torch.Tensor), (
            "Excepting collection of tensors as the first element in the "
            "return value of communication operations."
        )

        return _CommResult(e, work)

    # E.g.,
    # allreduce_ returns ([tensor], work)
    # allgather_ returns ([[tensor1, tensor2]], work)
    work = result[1]
    return (tree_map(partial(wrap, work), result[0]), work)


class CommTensor(torch.Tensor):
    r"""
    A Tensor subclass to wrap input tensors for collective communications. This
    Tensor subclass works for both eager and tracing mode.

    In eager mode, it will record whether the inplace collective communication
    has been launched using this Tensor and remember the corresponding work
    handle. If yes, it will expliclty call wait() in the ``__torch_dispatch__``
    function before subsequent operations consuming the value of the Tensor.

    In tracing mode, ``CommTensor`` inserts two node into the graph using the
    ``__torch_dispatch__`` function.
    1. The first node is inserted right after the
    communication, wrapping both the inplace output tensor and the returned
    work handle into a custom ``_CommResult`` type. We have to do this because
    ``ProxyTorchDispatchMode`` only handles ``torch.Tensor``, ``_ProxyTensor``,
    and ``torch.nn.Parameter`` objects and will treat the work handle
    as a constant and embed that into the graph. As a result, during execution,
    it will use the work handle created during tracing and will lead to wrong
    result. The solution in this test is to manually create a proxy on the
    return value of ``allreduce_`` which is ``([tensor], work)``, and wrap that
    to ``[(_CommResult(tensor, work)), work]``. In this way, subsequent nodes can
    directly consume ``_CommResult``.
    2. The second node is inserted right before any subsequent node reads from
    ``_CommResult``. It will call ``wait()`` on the stashed work handle to ensure
    that computation waits for communication.
    """

    _supported_comms: List[str] = [
        "allreduce_",
        "allgather_",
        "broadcast_",
        "reduce_scatter_",
        "scatter_",
    ]
    _after_comm: bool
    _tensor: torch.Tensor
    _work: torch.distributed._Work

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        r = torch.Tensor._make_subclass(
            cls,
            tensor,
            require_grad=tensor.requires_grad,
        )
        # The tensor object wrapped by this CommTensor
        r._tensor = tensor  # type: ignore[attr-defined]
        # Record whether communication has launched on this tensor.
        r._after_comm = False  # type: ignore[attr-defined]
        return r

    def __repr__(self):
        return f"CommTensor({self._tensor}, after_comm={self._after_comm})"

    # disable __torch_function__ so that CommTensor can recursively dispatch
    # with ProxyTorchDispatchMode in make_fx
    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def _is_supported(cls, op_name):
        return any([comm in op_name for comm in cls._supported_comms])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # shared states when unwrapping args
        tracer = None
        after_comm = False

        def get_tracer(obj):
            slots = get_proxy_slots(obj)
            if slots is None:
                return None
            keys = tuple(slots.keys())
            assert len(keys) == 1
            return keys[0]

        def get_proxy(obj):
            slots = get_proxy_slots(obj)
            if slots is None:
                return None
            vals = tuple(slots.values())
            assert len(vals) == 1
            return vals[0]

        # wrapped ._tensor if this is a CommTensor, and insert/call wait()
        # if communication has been launched on this tensor.
        def unwrap(e: Any):
            if isinstance(e, CommTensor):
                nonlocal tracer, after_comm

                after_comm = e._after_comm
                tracer = get_tracer(e._tensor)

                if after_comm:
                    if tracer is not None:
                        # insert a node to the traced graph.
                        proxy_res = tracer.create_proxy(
                            'call_function',
                            _wait_comm,
                            (get_proxy(e._tensor).proxy,),
                            {},
                            name="wait_comm"
                        )
                        # HACK: update the proxy for the inplace output
                        set_proxy_slot(e._tensor, tracer, proxy_res)
                    # For eager mode, simply wait.
                    # During tracing, still need to wait here, to make sure the
                    # execution during tracing is correct.
                    e._work.wait()

                return e._tensor
            else:
                return e

        def wrap(e: Any):
            return CommTensor(e) if isinstance(e, torch.Tensor) else e

        def mark_after_comm(work: torch.distributed._Work, e: Any):
            if isinstance(e, torch.Tensor) or isinstance(e, CommTensor):
                e._work = work  # type: ignore[attr-defined]
                e._after_comm = True  # type: ignore[attr-defined]

            return e

        unwrapped_args = tree_map(unwrap, args)
        unwrapped_kwargs = tree_map(unwrap, kwargs)

        if cls._is_supported(func.__name__):
            if tracer is not None:
                # in tracing mode, get proxies for args
                proxy_args, proxy_kwargs = tree_map_only(
                    _ProxyTensor,
                    lambda e: e.proxy,
                    tree_map_only(
                        torch.Tensor,
                        fetch_tensor_proxy(tracer),
                        (unwrapped_args, unwrapped_kwargs)
                    ),
                )

                # get proxy for output tuple
                proxy_res = func(*proxy_args, **proxy_kwargs)
                # insert a node that wraps the output tuple into
                # _CommResult(tensor, work)
                comm_result_proxy = tracer.create_proxy(
                    'call_function',
                    _wrap_comm_result,
                    (proxy_res, ),
                    {},
                    name="comm_result"
                )

                with no_dispatch():
                    # disable dispatch to avoid trigger ProxyTorchDispatchMode logic
                    out = func(*unwrapped_args, **unwrapped_kwargs)

                # wrap output with the proxy of _CommResult, so that subsequent
                # ops and link to it.
                track_tensor_tree(out, comm_result_proxy, constant=None, tracer=tracer)

                # N.B.: we still need to remember the work handle here, and wait
                # for it later to make sure the execution during tracing is
                # correct. Also, remember comm is already launched
                # args[0] is always the collection of output tensors
                tree_map(partial(mark_after_comm, out[1]), args[0])

                # HACK: update the proxy on the input argument as this is an
                # inplace collective communication.
                flat_args, args_spec = tree_flatten(unwrapped_args[0])
                flat_out, out_spec = tree_flatten(out[0])
                for a, o in zip(flat_args, flat_out):
                    set_proxy_slot(a, tracer, get_proxy(o))

                return out
            else:
                # in eager mode, simply remember work handle as an attribute
                out = func(*unwrapped_args, **kwargs)
                tree_map(partial(mark_after_comm, out[1]), args[0])
                return out
        else:
            if after_comm:
                return func(*unwrapped_args, **unwrapped_kwargs)
            else:
                # we need to propagate CommTensor wrapping until the first
                # subsequent operation has waited for it.
                return tree_map(wrap, func(*unwrapped_args, **unwrapped_kwargs))
