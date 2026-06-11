"""Custom Python process groups.

See CUSTOM_PG.md for full documentation and usage examples.
"""

import inspect
import weakref
from functools import wraps

from torch._C._distributed_c10d import ProcessGroup


__all__ = [
    "PassthroughProcessGroup",
    "setup_inner_pg",
]


# =====================================================================
# Internal helpers
# =====================================================================


class _DistributedBackendOpts:
    """Python wrapper around C++ ``_DistributedBackendOptions``.

    Extra fields for custom Python process groups:
    ``inner``, ``pg_options``, ``_remaining_pg_options``.
    """

    _extra_fields = frozenset(
        {
            "inner",
            "pg_options",
            "_remaining_pg_options",
        }
    )

    def __init__(self, base):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "inner", None)
        object.__setattr__(self, "pg_options", None)
        object.__setattr__(self, "_remaining_pg_options", None)

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def __setattr__(self, name: str, value):
        if name in _DistributedBackendOpts._extra_fields:
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)


def _pop_pg_options(pg_options, backend_name: str):
    """Extract per-layer and dist pg_options from *pg_options*.

    Returns ``(this_layer_opts, dist_opts, remaining_dict)``.
    """
    if not isinstance(pg_options, dict):
        return None, pg_options, None
    from torch.distributed.distributed_c10d import Backend

    valid_keys = {"dist"} | {k.lower() for k in Backend._plugins}
    unknown = set(pg_options) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown keys in pg_options dict: {unknown}. "
            f"Valid keys are registered backend names or 'dist'."
        )
    rest = dict(pg_options)
    this_layer = rest.pop(backend_name, None)
    dist = rest.pop("dist", None)
    remaining = rest if rest else None
    return this_layer, dist, remaining


def _create_process_group(dist_opts):
    """Create a process group from the backend string in *dist_opts*.

    Called internally by ``setup_inner_pg``.
    """
    from torch.distributed.distributed_c10d import _new_process_group_helper, GroupName

    inner = dist_opts.inner
    if inner is None:
        raise ValueError(
            "_create_process_group called with no inner in dist_opts. "
            "This is only for passthrough backends with a nested backend spec."
        )

    remaining = dist_opts._remaining_pg_options
    dist_pg_options = dist_opts.pg_options
    if remaining is not None:
        backend_options = dict(remaining)
        if dist_pg_options is not None:
            backend_options["dist"] = dist_pg_options
    else:
        backend_options = dist_pg_options

    store = dist_opts.store
    group_rank = dist_opts.group_rank
    group_size = dist_opts.group_size
    timeout = dist_opts.timeout
    group_name = dist_opts.group_id
    global_ranks_in_group = dist_opts.global_ranks_in_group

    inner_name = f"{group_name}/inner"
    pg, _ = _new_process_group_helper(
        group_size,
        group_rank,
        global_ranks_in_group,
        inner,
        store,
        GroupName(inner_name),
        backend_options=backend_options,
        timeout=timeout,
    )
    return pg


def _pg_bypass(group_param="group"):
    """Decorator: if the ProcessGroup overrides the function, forward directly.

    Walks the MRO (stopping at ProcessGroup) to find methods with
    matching parameter names.  The signature check (subset match,
    ignoring *args/**kwargs) distinguishes dist.* API overrides from
    C++ virtual method overrides.  Cached per class.
    """

    def decorator(fn):
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        group_pos = params.index(group_param) if group_param in params else -1
        fn_param_names = {p for p in sig.parameters if p != group_param}

        _sig_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

        def _has_matching_sig(cls):
            if cls not in _sig_cache:
                method = cls.__dict__.get(fn.__name__)
                if method is None:
                    _sig_cache[cls] = False
                elif not fn_param_names:
                    _sig_cache[cls] = True
                else:
                    try:
                        pg_sig = inspect.signature(method)
                        pg_params = {
                            p
                            for p, v in pg_sig.parameters.items()
                            if p != "self"
                            and v.kind
                            not in (
                                inspect.Parameter.VAR_POSITIONAL,
                                inspect.Parameter.VAR_KEYWORD,
                            )
                        }
                        _sig_cache[cls] = bool(pg_params) and pg_params.issubset(
                            fn_param_names
                        )
                    except (ValueError, TypeError):
                        _sig_cache[cls] = False
            return _sig_cache[cls]

        def _find_bypass(group):
            for cls in type(group).__mro__:
                if cls is ProcessGroup or cls is object:
                    break
                if _has_matching_sig(cls):
                    return cls.__dict__[fn.__name__], group
            return None, None

        @wraps(fn)
        def wrapper(*args, **kwargs):
            from torch.distributed.distributed_c10d import _get_default_group
            if group_pos < 0:
                group = None
            elif group_param in kwargs:
                group = kwargs.pop(group_param)
            elif len(args) > group_pos:
                group = args[group_pos]
                args = args[:group_pos] + args[group_pos + 1 :]
            else:
                group = None
            group = group or _get_default_group()
            pg_fn, target_pg = _find_bypass(group)
            if pg_fn is not None:
                return pg_fn(target_pg, *args, **kwargs)
            if group_pos < 0:
                return fn(*args, **kwargs)
            return fn(*args, **{group_param: group, **kwargs})

        return wrapper

    return decorator


# =====================================================================
# Public API
# =====================================================================


class PassthroughProcessGroup(ProcessGroup):
    """Base class for custom backends that wrap an inner ProcessGroup.

    See CUSTOM_PG.md for full documentation and usage examples.
    Use setup_inner_pg(pg, dist_opts) in the creator to attach
    the inner PG.  Do NOT set _inner_pg directly.
    """

    def __init__(self, rank: int, size: int):
        super().__init__(rank, size)  # pyrefly: ignore[missing-argument, bad-argument-type]
        self._inner_pg = None
        import torch.distributed as _dist

        self._dist = _dist

    def getBackendName(self) -> str:
        return type(self).__name__

    def __getattr__(self, name: str):
        inner = self.__dict__.get("_inner_pg")
        if inner is None:
            raise AttributeError(name)
        fn = getattr(inner, name, None)
        if fn is not None:
            return fn
        raise AttributeError(name)

    # -- C++ virtual method overrides (forward to inner PG) -----------
    # These ensure that direct C++ calls like group.allreduce(...)
    # are forwarded to the inner PG rather than failing with
    # "No backend type associated with device type".

    def allreduce(self, tensors, opts=None):  # pyrefly: ignore[bad-override]
        return self._inner_pg.allreduce(tensors, opts)  # pyrefly: ignore[missing-attribute]

    def allreduce_coalesced(self, tensors, opts=None):  # pyrefly: ignore[bad-override]
        return self._inner_pg.allreduce_coalesced(tensors, opts)  # pyrefly: ignore[missing-attribute]

    def allgather(  # pyrefly: ignore[bad-override]
        self, output_tensors, input_tensors, opts=None
    ):
        return self._inner_pg.allgather(output_tensors, input_tensors, opts)  # pyrefly: ignore[missing-attribute]

    def _allgather_base(
        self, output, input, opts=None
    ):  # pyrefly: ignore[bad-override]
        return self._inner_pg._allgather_base(output, input, opts)  # pyrefly: ignore[missing-attribute]

    def allgather_coalesced(
        self, output_lists, input_list, opts=None
    ):  # pyrefly: ignore[bad-override]
        return self._inner_pg.allgather_coalesced(output_lists, input_list, opts)  # pyrefly: ignore[missing-attribute]

    def allgather_into_tensor_coalesced(
        self, outputs, inputs, opts=None
    ):  # pyrefly: ignore[bad-override]
        return self._inner_pg.allgather_into_tensor_coalesced(outputs, inputs, opts)  # pyrefly: ignore[missing-attribute]

    def reduce_scatter_tensor_coalesced(
        self, outputs, inputs, opts=None
    ):  # pyrefly: ignore[bad-override]
        return self._inner_pg.reduce_scatter_tensor_coalesced(outputs, inputs, opts)  # pyrefly: ignore[missing-attribute]

    def _reduce_scatter_base(
        self, output, input, opts=None
    ):  # pyrefly: ignore[bad-override]
        return self._inner_pg._reduce_scatter_base(output, input, opts)  # pyrefly: ignore[missing-attribute]

    def alltoall_base(  # pyrefly: ignore[bad-override]
        self, output, input, output_split_sizes=None,
        input_split_sizes=None, opts=None,
    ):
        return self._inner_pg.alltoall_base(  # pyrefly: ignore[missing-attribute]
            output, input, output_split_sizes, input_split_sizes, opts)

    def alltoall(  # pyrefly: ignore[bad-override]
        self, output_tensors, input_tensors, opts=None
    ):
        return self._inner_pg.alltoall(output_tensors, input_tensors, opts)  # pyrefly: ignore[missing-attribute]

    # -- Python dist.* forwarding methods ------------------------------
    # These are found by _pg_bypass's MRO walk and call back into
    # dist.<fn>(group=self._inner_pg) so the bypass mechanism handles
    # the inner PG correctly.

    def send(self, tensor, dst=0, tag=0, group_dst=None, **kwargs):  # pyrefly: ignore[bad-override]
        return self._dist.send(tensor, dst=dst, group=self._inner_pg,
                               tag=tag, group_dst=group_dst, **kwargs)

    def isend(self, tensor, dst=0, tag=0, group_dst=None, **kwargs):
        return self._dist.isend(tensor, dst=dst, group=self._inner_pg,
                                tag=tag, group_dst=group_dst, **kwargs)

    def recv(self, tensor, src=None, tag=0, group_src=None, **kwargs):  # pyrefly: ignore[bad-override]
        return self._dist.recv(tensor, src=src, group=self._inner_pg,
                               tag=tag, group_src=group_src, **kwargs)

    def irecv(self, tensor, src=None, tag=0, group_src=None, **kwargs):
        return self._dist.irecv(tensor, src=src, group=self._inner_pg,
                                tag=tag, group_src=group_src, **kwargs)

    def broadcast(self, tensor, src=None, async_op=False,  # pyrefly: ignore[bad-override]
                  group_src=None, **kwargs):
        return self._dist.broadcast(
            tensor, src=src, group=self._inner_pg,
            async_op=async_op, group_src=group_src, **kwargs)

    def all_reduce(self, tensor, op=None, async_op=False, **kwargs):
        return self._dist.all_reduce(
            tensor, op=op, group=self._inner_pg,
            async_op=async_op, **kwargs)

    def all_reduce_coalesced(self, tensors, op=None, async_op=False,
                             **kwargs):
        return self._dist.all_reduce_coalesced(
            tensors, op=op, group=self._inner_pg,
            async_op=async_op, **kwargs)

    def reduce(self, tensor, dst=0, op=None, async_op=False,  # pyrefly: ignore[bad-override]
               group_dst=None, **kwargs):
        return self._dist.reduce(
            tensor, dst=dst, op=op, group=self._inner_pg,
            async_op=async_op, group_dst=group_dst, **kwargs)

    def all_gather_object(self, object_list, obj, **kwargs):
        return self._dist.all_gather_object(
            object_list, obj, group=self._inner_pg, **kwargs)

    def gather_object(self, obj, object_gather_list=None, dst=0,
                      group_dst=None, **kwargs):
        return self._dist.gather_object(
            obj, object_gather_list=object_gather_list, dst=dst,
            group=self._inner_pg, group_dst=group_dst, **kwargs)

    def send_object_list(self, object_list, dst=0, device=None,
                         group_dst=None, use_batch=False, **kwargs):
        return self._dist.send_object_list(
            object_list, dst=dst, group=self._inner_pg, device=device,
            group_dst=group_dst, use_batch=use_batch, **kwargs)

    def recv_object_list(self, object_list, src=0, device=None,
                         group_src=None, use_batch=False, **kwargs):
        return self._dist.recv_object_list(
            object_list, src=src, group=self._inner_pg, device=device,
            group_src=group_src, use_batch=use_batch, **kwargs)

    def broadcast_object_list(self, object_list, src=None, device=None,
                              group_src=None, **kwargs):
        return self._dist.broadcast_object_list(
            object_list, src=src, group=self._inner_pg, device=device,
            group_src=group_src, **kwargs)

    def scatter_object_list(self, scatter_object_output_list,
                            scatter_object_input_list, src=0,
                            group_src=None, **kwargs):
        return self._dist.scatter_object_list(
            scatter_object_output_list, scatter_object_input_list,
            src=src, group=self._inner_pg, group_src=group_src, **kwargs)

    def all_gather(self, tensor_list, tensor, async_op=False, **kwargs):
        return self._dist.all_gather(
            tensor_list, tensor, group=self._inner_pg,
            async_op=async_op, **kwargs)

    def all_gather_into_tensor(self, output_tensor, input_tensor,
                               async_op=False, **kwargs):
        return self._dist.all_gather_into_tensor(
            output_tensor, input_tensor, group=self._inner_pg,
            async_op=async_op, **kwargs)

    def all_gather_coalesced(self, output_tensor_lists,
                             input_tensor_list, async_op=False, **kwargs):
        return self._dist.all_gather_coalesced(
            output_tensor_lists, input_tensor_list,
            group=self._inner_pg, async_op=async_op, **kwargs)

    def gather(self, tensor, gather_list=None, dst=0, async_op=False,  # pyrefly: ignore[bad-override]
               group_dst=None, **kwargs):
        return self._dist.gather(
            tensor, gather_list=gather_list, dst=dst,
            group=self._inner_pg, async_op=async_op,
            group_dst=group_dst, **kwargs)

    def scatter(self, tensor, scatter_list=None, src=0, async_op=False,  # pyrefly: ignore[bad-override]
                group_src=None, **kwargs):
        return self._dist.scatter(
            tensor, scatter_list=scatter_list, src=src,
            group=self._inner_pg, async_op=async_op,
            group_src=group_src, **kwargs)

    def reduce_scatter(self, output, input_list, op=None, async_op=False,  # pyrefly: ignore[bad-override]
                       **kwargs):
        return self._dist.reduce_scatter(
            output, input_list, op=op, group=self._inner_pg,
            async_op=async_op, **kwargs)

    def reduce_scatter_tensor(self, output, input, op=None,
                              async_op=False, **kwargs):
        return self._dist.reduce_scatter_tensor(
            output, input, op=op, group=self._inner_pg,
            async_op=async_op, **kwargs)

    def all_to_all_single(self, output, input, output_split_sizes=None,
                          input_split_sizes=None, async_op=False,
                          **kwargs):
        return self._dist.all_to_all_single(
            output, input, output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self._inner_pg, async_op=async_op, **kwargs)

    def all_to_all(self, output_tensor_list, input_tensor_list,
                   async_op=False, **kwargs):
        return self._dist.all_to_all(
            output_tensor_list, input_tensor_list,
            group=self._inner_pg, async_op=async_op, **kwargs)

    def barrier(self, async_op=False, device_ids=None, timeout=None,  # pyrefly: ignore[bad-override]
                **kwargs):
        return self._dist.barrier(
            group=self._inner_pg, async_op=async_op,
            device_ids=device_ids, timeout=timeout, **kwargs)

    def monitored_barrier(self, timeout=None, wait_all_ranks=False,
                          **kwargs):
        return self._dist.monitored_barrier(
            group=self._inner_pg, timeout=timeout,
            wait_all_ranks=wait_all_ranks, **kwargs)


def setup_inner_pg(pg: PassthroughProcessGroup, dist_opts):
    """Create the inner PG from *dist_opts* and attach it to *pg*.

    This is called in a passthrough creator function after constructing
    the custom PG::

        def create_my_passthrough(dist_opts, pg_options=None):
            pg = MyPassthroughPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg
    """
    pg._inner_pg = _create_process_group(dist_opts)
