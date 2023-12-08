import contextlib
from typing import Optional, Union, List, Set, Dict, Any

import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
    _pop_torch_dispatch_stack, _push_on_torch_dispatch_stack, DispatchKey


# TODO: Limitations and things about enable_torch_dispatch_mode we should fix before exposing it:
# - We need a better user-facing api for _DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)

class TorchDispatchMode:
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    def __init__(self, _dispatch_key=None):
        if _dispatch_key is not None:
            assert isinstance(_dispatch_key, torch._C.DispatchKey)
            self.__dict__['_dispatch_key'] = _dispatch_key

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()

    def __enter__(self):
        _push_mode(self, self.__dict__.get("_dispatch_key", None))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mb_dk_or_mode_key = self.__dict__.get("_dispatch_key", None)
        if mb_dk_or_mode_key is None:
            # Today, mode keys are not used at all in the per-dispatch-key-mode logic (for pre-dispatch)
            # We should probably revisit this.
            mb_dk_or_mode_key = self.__dict__.get("_mode_key", None)
        _pop_mode(mb_dk_or_mode_key)

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn("`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`")
        instance = cls(*args, **kwargs)
        return instance

def _get_current_dispatch_mode():
    stack_len = _len_torch_dispatch_stack()
    # Return a user mode on the stack if there are any
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    return None


def _get_current_dispatch_mode_stack():
    stack_len = _len_torch_dispatch_stack()
    return [_get_dispatch_stack_at(i) for i in range(stack_len)]

def _push_mode(mode, k: Optional[DispatchKey] = None):
    if k is not None:
        from torch._ops import push_mode_for_key, get_cached_ops
        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        # Clear the cache of every op that has been used so far, for this particular key.
        ks = torch._C._functionality_to_backend_keys(k)
        for op in get_cached_ops():
            for key in ks:
                op._uncache_dispatch(key)
        push_mode_for_key(k, mode)
    else:
        _push_on_torch_dispatch_stack(mode)


def _pop_mode(k: Optional[Union[DispatchKey, torch._C._TorchDispatchModeKey]] = None):
    if k is None or isinstance(k, torch._C._TorchDispatchModeKey):
        return _pop_torch_dispatch_stack(k)
    from torch._ops import pop_mode_for_key
    # per-dispatch-key-mode-stack do not currently handle "always running infra modes last".
    # In practice this doesn't matter, since ProxyTorchDispatchMode is the only mode
    # that we push onto these per-dispatch-key-mode-stacks.
    return pop_mode_for_key(k)


@contextlib.contextmanager
def _pop_mode_temporarily(k: Optional[DispatchKey] = None):
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old, k)


@contextlib.contextmanager
def _disable_current_modes():
    mode_len = _len_torch_dispatch_stack()
    old_modes = [_pop_mode() for _ in range(mode_len)]

    # Manually disable proxy and fake modes, if any are active
    try:
        yield old_modes
    finally:
        for mode in reversed(old_modes):
            _push_mode(mode)


class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

def is_traceable_wrapper_subclass(t):
    """
    Returns whether or not a tensor subclass that implements __torch_dispatch__
    is 'traceable' with torch.compile.
    In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,
    It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.
    It is also expected to obey some restrictions around traceability and aliasing
    (TODO: add clear documentation around this.)
    """
    is_subclass = isinstance(t, torch.Tensor) and type(t) != torch.Tensor
    return is_subclass and hasattr(t, "__tensor_flatten__") and hasattr(t, "__tensor_unflatten__")

def transform_subclass(t, callback):
    """
    Given a traceable, wrapper tensor subclass ``t`` that implements
    ``__torch_dispatch__`` and holds some inner tensors,
    and a callback of type ``Callable[[str, torch.Tensor], torch.Tensor]``,
    `transform_subclass` will construct a fresh instance of the wrapper tensor subclass.
    It will do so by grabbing each inner tensor attribute from the wrapper,
    passing them into ``callback`` to get a transformed tensor,
    and putting each transformed tensor into the fresh tensor subclass instance.

    Note: this function will not handle ensuring that the fresh subclass
    gets the same (autograd, and aliasing) metadata as the original tensor.
    This is generally handled in other subsystems like AOTAutograd.
    """
    attrs, ctx = t.__tensor_flatten__()
    transformed_tensors_dict = {}
    for attr in attrs:
        transformed_tensors_dict[attr] = callback(attr, getattr(t, attr))
    return type(t).__tensor_unflatten__(transformed_tensors_dict, ctx)

def _correct_storage_aliasing(func, schema_info, args, outs):
    """
    Given: an OpOverload, a SchemaInfo (cached information from torchgen about schema),
    and the inputs/outputs to the OpOverload,
    this function checks to see if func is a view operator
    (by checking if any of the outputs in the op's schema
     are immutable aliases of inputs).
    If so, this function manually aliases the storage of the output tensor
    with its corresponding input tensor alias.
    It does this by unsafely overwriting the storage field of the output tensor
    to be the same storage as the input.
    """
    assert isinstance(func, torch._ops.OpOverload)
    assert isinstance(args, tuple)
    assert isinstance(outs, (list, tuple))
    flat_outs = torch.utils._pytree.tree_leaves(outs)

    def alias_non_inplace_storage(arg, ret):
        # This is hopefully a reasonable assert:
        # subclasses that rely on this API for output aliasing
        # should always return wrapper tensor subclasses for us to manually alias.
        # in theory if a subclass that needs this API wants to sometimes return
        # plain tensors, we could remove the assert and just not perform the aliasing,
        # but it seems safer to learn more about this case first.
        if is_traceable_wrapper_subclass(arg) or is_traceable_wrapper_subclass(ret):
            ret_list = ret if isinstance(ret, list) else [ret]
            for r in ret_list:
                assert type(arg) == type(r), f"""Called {str(func)} with input of type {type(arg)}
and output of type {type(ret)}. But expected types to match."""
        # Need to run under no_dispatch, because we explicitly do **not**
        # want our subclass to intercept the set_() call.
        # instead, our subclass should directly have its storage swapped out.
        with torch.utils._mode_utils.no_dispatch():
            # See Note: [Fake Tensor Dispatch Keys]
            # we're borrowing the way it modifies dispatch key TLS.
            meta_in_tls = torch._C._meta_in_tls_dispatch_include()
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                # directly calling this overload, and passing ret.shape, because we **explicitly**
                # don't want to reset the sizes on ret, if the storage implies a size change.
                # Why?
                # The purpose of this API is *not* to change the size/strides of our output- we assume it's already correct.
                # We just want to "fix up" the storage aliasing, without modifying or output's metadata.
                # Example: out = inp.expand(inp.shape[0], inp.shape[0])
                #     This requires swapping the storage of out to be the same as inp,
                #     but we do *not* want it to change the sizes/strides that were compute for out.
                if isinstance(ret, list):
                    for r in ret:
                        torch.ops.aten.set_.source_Storage_storage_offset(r, arg.untyped_storage(), r.storage_offset(), r.shape)
                else:
                    assert isinstance(ret, torch.Tensor), f"type: {type(ret)}"
                    torch.ops.aten.set_.source_Storage_storage_offset(ret, arg.untyped_storage(), ret.storage_offset(), ret.shape)
            finally:
                torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

    def is_read_only_alias_match(arg, ret):
        shared_aliases = arg.alias_set & ret.alias_set
        return len(shared_aliases) > 0 and not arg.is_write

    num_args = len(func._schema.arguments)
    num_returns = len(func._schema.returns)
    for arg_idx in range(num_args):
        for return_idx in range(num_returns):
            if is_read_only_alias_match(schema_info.args[arg_idx], schema_info.outs[return_idx]):
                alias_non_inplace_storage(args[arg_idx], outs[return_idx])

# This abstracts over the fact that in return_and_correct_aliasing,
# we sometimes use torchgen schema parsing (for aten ops, since torchscript's schema parsing is sometimes buggy),
# and sometimes use torchscript schema parsing (for custom ops, for which torchgen parsing is untested).
@dataclass
class AliasInfo:
    alias_set: Set[str]
    is_write: bool
    name: Optional[str]

@dataclass
class SchemaInfo:
    args: List[AliasInfo]
    outs: List[AliasInfo]

# Can't import torch._ops.OpOverload due to circular reference
parsed_schema_map: Dict[Any, SchemaInfo] = {}

# Given an OpOverload, returns schema information on it.
# This is cached for efficiency, since it can involve running torchgen
def get_alias_info(func) -> SchemaInfo:
    if func in parsed_schema_map:
        return parsed_schema_map[func]
    # For ATen ops: use torchgen (since torchscript parser doesn't handle alias annotations
    # properly for some ops that output tensorlists)
    if func.namespace == "aten":
        torchgen_schema_str = str(func._schema)
        assert torchgen_schema_str.startswith("aten::")
        # remove the aten:: namespace, which is added by the torchscript parser,
        # and torchgen doesn't know how to handle
        torchgen_schema_str = torchgen_schema_str[6:]
        import re
        # the torchscript parser ends up converting int[2]=1 into int[2]=[1, 1],
        # which torchgen chokes on.
        torchgen_schema_str = re.sub(r'=\[[0, ]+\]', '=0', torchgen_schema_str)
        torchgen_schema_str = re.sub(r'=\[[1, ]+\]', '=1', torchgen_schema_str)
        # for aten::rot90
        torchgen_schema_str = torchgen_schema_str.replace("=[0, 1]", "=[0,1]")
        torchgen_schema = torchgen.model.FunctionSchema.parse(torchgen_schema_str)
        arg_schemas = [AliasInfo(
            alias_set=set() if a.annotation is None else set(a.annotation.alias_set),
            is_write=a.annotation is not None and a.annotation.is_write,
            name=a.name,
        ) for a in torchgen_schema.arguments.flat_all]
        out_schemas = [AliasInfo(
            alias_set=set() if a.annotation is None else set(a.annotation.alias_set),
            is_write=a.annotation is not None and a.annotation.is_write,
            name=a.name,
        ) for a in torchgen_schema.returns]
    else:
        # For non-aten ops, torchgen is untested so we rely on torchscript schema parsing
        arg_schemas = [AliasInfo(
            alias_set=set() if a.alias_info is None else set(a.alias_info.before_set),
            is_write=a.alias_info is not None and a.alias_info.is_write,
            name=a.name,
        ) for a in func._schema.arguments]
        out_schemas = [AliasInfo(
            alias_set=set() if a.alias_info is None else set(a.alias_info.before_set),
            is_write=a.alias_info is not None and a.alias_info.is_write,
            name=a.name,
        ) for a in func._schema.returns]
    schema_info = SchemaInfo(args=arg_schemas, outs=out_schemas)
    parsed_schema_map[func] = schema_info
    return schema_info

def return_and_correct_aliasing(func, args, kwargs, out):
    """
    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses
    that would like to work with torch.compile. It ensures that the subclass
    properly implements the aliasing behavior of every op,
    which is needed for correctness in AOTAutograd.
    This function will handle:

        * When we see a view op, we will alias the storages of any
          input and output tensor subclasses

        * When we see an inplace or out= op, we will directly
          return the corresponding input tensor, instead of returning
          a (potentially) fresh output tensor.
    """

    # Caching here because torchgen parsing is definitely not fast, and this function is called
    # once for every op in the graph during functionalization.
    schema_info = get_alias_info(func)

    def get_write_alias(x):
        if len(x.alias_set) == 0:
            return None
        alias_set = list(x.alias_set)
        # torchscript allows for complicated alias sets, but our dispatcher ops only really involve simple aliasing
        assert len(alias_set) == 1
        if x.is_write:
            return alias_set[0]
        return None

    def get_arg_from_alias(output_alias, schema_info, args, kwargs):
        new_args, new_kwargs = torch.fx.operator_schemas.normalize_function(func, args=args, kwargs=kwargs)

        arg_indices = [
            i for i, a in enumerate(schema_info.args)
            if output_alias in a.alias_set
        ]
        # For any dispatcher op with an output alias, we expect it to map to exactly one alias in the schema's input arguments.
        assert len(arg_indices) == 1
        idx = arg_indices[0]
        arg_info = schema_info.args[idx]
        if arg_info.name is not None and arg_info.name in new_kwargs:
            return new_kwargs[arg_info.name]
        return new_args[idx]

    # Fix up the storages of any outs so that they point to the same storage as the input,
    # if func is a view op.
    _correct_storage_aliasing(func, schema_info, args, (out,) if not isinstance(out, tuple) else out)

    # For inplace_view ops in particular, we'll try hard to make sure that the wrapper subclass's
    # metadata is set correctly.
    if torch.Tag.inplace_view in func.tags:
        # no_dispatch() to make sure that we secretly change the metadata on the wrapper,
        # but don't end up dispatching the op anywhere else.
        mutated_args = [x for i, x in enumerate(args) if get_write_alias(schema_info.args[i]) is not None]
        # Assumption: we have a very small number of inplace_view ops that follow a strict schema:
        # there is only a single argument that gets its metadata mutated.
        assert len(mutated_args) == 1
        # This check exists because we generally *do* want to update the metadata of any wrapper subclasses,
        # but FunctionalTensor is special: it overrides all size/stride calls to plumb to the inner tensor.
        # so we don't actually need to update the metadata (and attempting to do so causes errors)
        from torch._subclasses.functional_tensor import FunctionalTensor
        if not isinstance(mutated_args[0], FunctionalTensor):
            with torch.utils._mode_utils.no_dispatch():
                # See Note: [Fake Tensor Dispatch Keys]
                # we're borrowing the way it modifies dispatch key TLS.
                meta_in_tls = torch._C._meta_in_tls_dispatch_include()
                torch._C._set_meta_in_tls_dispatch_include(True)
                try:
                    func(*args, **kwargs)
                finally:
                    torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)

    # Next: we need to make sure to return inputs directly, if the output is a mutable alias (e.g. add_()).

    # simple case: none of our outputs have mutable aliases, so we can return the output as-is
    if not any(get_write_alias(r) is not None for r in schema_info.outs):
        return out

    # simplifying assumption: we don't have **any** ops with return types like "-> (Tensor(a!), Tensor)"
    if not all(get_write_alias(r) is not None for r in schema_info.outs):
        raise RuntimeError("Unsupported schema: " + str(func._schema))

    if len(func._schema.returns) == 1:
        return get_arg_from_alias(get_write_alias(schema_info.outs[0]), schema_info, args, kwargs)

    # In the multi-return case, all aten ops return a tuple / list, so cast accordingly.
    outs_to_return = type(out)([
        get_arg_from_alias(get_write_alias(schema_info.outs[i]), schema_info, args, kwargs)
        if get_write_alias(r) is not None else o
        for ((i, r), o) in zip(enumerate(schema_info.outs), out)
    ])
    return outs_to_return
