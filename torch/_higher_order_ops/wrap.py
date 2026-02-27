# mypy: allow-untyped-defs
import inspect
import itertools
import logging
import weakref
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    redirect_to_mode,
    reenter_make_fx,
    register_fake,
)
from torch._logging import warning_once
from torch._ops import HigherOrderOperator
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.types import _dtype
from torch.utils._debug_mode import DebugMode
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


log = logging.getLogger(__name__)


uid = itertools.count(1)


# Used for testing the HigherOrderOperator mechanism
class Wrap(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("wrap")

    def __call__(self, func, *args, **kwargs):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        @disable
        def wrapper():
            result = func(*args, **kwargs)
            return result

        return wrapper()


wrap = Wrap()


class InductorCompiledCode(HigherOrderOperator):
    """
    Defines a HOP for wrapping inductor compiled functions as a callable.
    When used with torch.compile via "wrap_inductor_compiled_regions",
    this HOP will automatically be wrapped and redirect various torch dispatch modes.
    """

    def __init__(self) -> None:
        super().__init__("inductor_compiled_code", no_overloaded_args=True)

    def __call__(self, func, *args, **kwargs):
        # pyrefly: ignore [missing-attribute]
        return super().__call__(func, *args, **kwargs)


inductor_compiled_code = InductorCompiledCode()
inductor_compiled_code.fallthrough(DispatchKey.AutogradCPU)
inductor_compiled_code.fallthrough(DispatchKey.AutogradCUDA)
inductor_compiled_code.fallthrough(DispatchKey.Negative)
inductor_compiled_code.fallthrough(DispatchKey.Conjugate)


_inductor_compiled_callable_id = itertools.count()


class InductorCompiledCallable:
    """
    A wrapper class that holds both the Inductor-compiled callable and the
    original FX graph for fake tensor propagation.
    Each instance gets a globally unique idx at creation (via atomic itertools.count).
    """

    def __init__(self, compiled_callable, original_gm=None):
        self.idx = next(_inductor_compiled_callable_id)
        self.compiled_callable = compiled_callable
        self.original_gm = original_gm
        # AOT autograd needs this to know inputs are passed as a list
        self._boxed_call = True

    def __call__(self, inputs):
        return self.compiled_callable(inputs)


class InductorCodeSideTable:
    """
    Side table for storing InductorCompiledCallable objects.

    We cannot put InductorCompiledCallable objects directly into the FX graph
    as graph nodes do not support arbitrary objects. We use this side table
    and pass callable.idx instead.

    Uses WeakValueDictionary so entries are automatically removed when
    the InductorCompiledCallable is no longer referenced elsewhere
    (e.g. after dynamo.reset() drops the compiled code cache).
    """

    def __init__(self):
        self.id_to_callable: weakref.WeakValueDictionary[
            int, InductorCompiledCallable
        ] = weakref.WeakValueDictionary()

    def add_callable(self, callable_obj: InductorCompiledCallable) -> int:
        """Register a callable and return its idx."""
        self.id_to_callable[callable_obj.idx] = callable_obj
        return callable_obj.idx

    def get_callable(self, idx: int) -> InductorCompiledCallable:
        """Get the callable at the given index."""
        assert idx in self.id_to_callable, f"Invalid inductor code index: {idx}"  # noqa: S101
        return self.id_to_callable[idx]

    def __getstate__(self):
        # Convert WeakValueDictionary to regular dict for pickling
        return {"id_to_callable": dict(self.id_to_callable)}

    def __setstate__(self, state):
        self.id_to_callable = weakref.WeakValueDictionary(state["id_to_callable"])

    def reset_table(self) -> None:
        """Reset the table."""
        self.id_to_callable = weakref.WeakValueDictionary()


inductor_code_side_table = InductorCodeSideTable()


def _resolve_inductor_callable(func) -> InductorCompiledCallable:
    """
    Resolve func to an InductorCompiledCallable.

    func is either an InductorCompiledCallable directly (from post_compile)
    or an int index into the side table (from a traced FX graph node).
    """
    if isinstance(func, int):
        return inductor_code_side_table.get_callable(func)
    assert isinstance(func, InductorCompiledCallable), (  # noqa: S101
        f"Unexpected func type: {type(func)}"
    )
    return func


@inductor_compiled_code.py_impl(DispatchKey.CompositeExplicitAutograd)
def inductor_compiled_code_impl(func, inputs):
    resolved = _resolve_inductor_callable(func)
    return resolved.compiled_callable(inputs)


redirect_to_mode(inductor_compiled_code, DebugMode)
redirect_to_mode(inductor_compiled_code, _CachingTorchDispatchMode)
redirect_to_mode(inductor_compiled_code, _CachedTorchDispatchMode)


@register_fake(inductor_compiled_code)
def inductor_compiled_code_fake(func, inputs):
    resolved = _resolve_inductor_callable(func)
    if resolved.original_gm is None:
        raise RuntimeError(
            "inductor_compiled_code original_gm is None — the compiled graph may "
            "have been serialized without it. Recompile to restore."
        )
    # Run the original FX graph under FakeTensorMode to re-derive output
    # shapes, dtypes, and aliasing from the input fake tensors.
    return tuple(resolved.original_gm(*inputs))


@inductor_compiled_code.py_functionalize_impl
def inductor_compiled_code_functionalize(ctx, func, inputs):
    # Unwrap the functional tensors to get the underlying tensors
    unwrapped_inputs = ctx.unwrap_tensors(inputs)

    # Redispatch to the next handler in the dispatch chain
    with ctx.redispatch_to_next():
        result = inductor_compiled_code(func, unwrapped_inputs)
        return ctx.wrap_tensors(result)


@inductor_compiled_code.py_impl(ProxyTorchDispatchMode)
def inductor_compiled_code_proxy(mode, func, inputs):
    resolved = _resolve_inductor_callable(func)

    # Run the fake impl to get example outputs for tracing
    example_out = inductor_compiled_code(func, inputs)

    # Register in side table so the FX node stores a serializable int
    callable_idx = inductor_code_side_table.add_callable(resolved)

    proxy_inputs = pytree.tree_map(mode.tracer.unwrap_proxy, inputs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        inductor_compiled_code,
        (callable_idx, proxy_inputs),
        {},
    )

    return track_tensor_tree(example_out, out_proxy, constant=None, tracer=mode.tracer)


class WrapWithSetGradEnabled(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("wrap_with_set_grad_enabled")

    def __call__(self, enable_grad, wrapped_func, *args, **kwargs):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        @disable
        def wrapper():
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(enable_grad)
            res = wrapped_func(*args, **kwargs)
            torch.set_grad_enabled(prev)
            return res

        return wrapper()


wrap_with_set_grad_enabled = WrapWithSetGradEnabled()


class WrapWithAutocast(HigherOrderOperator):
    def __init__(self):
        super().__init__("wrap_with_autocast")

    def __call__(
        self,
        device_type: str,
        dtype: _dtype | None,
        enabled: bool,
        cache_enabled: bool | None,
        wrapped_func,
        *args,
        **kwargs,
    ):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        @disable
        def wrapper():
            with torch.autocast(device_type, dtype, enabled, cache_enabled):
                return wrapped_func(*args, **kwargs)

        return wrapper()


wrap_with_autocast = WrapWithAutocast()


# This HOP allows you to bypass dynamo tracing of the wrapper function while
# still tracing the inner function.
# Takes two callables: The first, `wrapper_fn`, accepts `inner_fn` and returns a
# callable with the same signature. The second is the `inner_fn` itself. Any
# extra *args and **kwargs are forwarded to `wrapper_fn(inner_fn)` when it is
# executed.
class DynamoBypassingWrapper(HigherOrderOperator):
    def __init__(self):
        super().__init__("dynamo_bypassing_wrapper")

    def __call__(
        self,
        wrapper_fn_or_key,
        inner_fn,
        *args,
        **kwargs,
    ):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        is_compiling = isinstance(wrapper_fn_or_key, str)
        if is_compiling:
            if not isinstance(inner_fn, torch.fx.GraphModule):
                raise AssertionError(
                    f"expected inner_fn to be torch.fx.GraphModule, got {type(inner_fn)}"
                )
            wrapper_fn = inner_fn.meta[wrapper_fn_or_key]
        else:
            wrapper_fn = wrapper_fn_or_key

        @disable
        def wrapper():
            return wrapper_fn(inner_fn)(*args, **kwargs)

        return wrapper()


dynamo_bypassing_wrapper = DynamoBypassingWrapper()


class WrapActivationCheckpoint(HigherOrderOperator):
    """
    This operator is used to wrap torch.utils.checkpoint. This avoids
    TorchDynamo to look into saved tensor hooks and directly passes the control
    to AOT Autograd, which is ok with tracing saved tensor hooks. As a result of
    AOT tracing torch.utils.checkpoint code, we have a backward graph with
    recomputed forward nodes.

    However, we might deprecate this operator soon. The difficulty arises in the
    functionalization of rng ops. Today, there are two different
    functionalization of rng ops - one at AOT autograd and other at Inductor.
    And they are difficult to map to each other. The rng states also complicate
    pattern matching in Inductor. Due to the ease of implementation, we are
    currently inclined towards functionalization at Inductor level, which means
    that duplication/recomputation is done as a compiler pass in the
    partitioners. See TagActivationCheckpoint for more information.
    """

    def __init__(self) -> None:
        super().__init__("wrap_activation_checkpoint", cacheable=False)

    def __call__(self, function, *args, **kwargs):
        # use_reentrant is set to False because this op is going to be traced.
        # And we ensure that AOT Autograd traces through the non reentrant
        # version of checkpointing.
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter

        kwargs["use_reentrant"] = False
        kwargs["preserve_rng_state"] = False
        # Using interpreter allows preservation of metadata through torch.compile stack.
        with fx_traceback.preserve_node_meta():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(Interpreter(function).run, *args, **kwargs)


wrap_activation_checkpoint = WrapActivationCheckpoint()


class TagActivationCheckpoint(HigherOrderOperator):
    """
    This operator is supposed to be used only with torch.compile stack. This
    accepts a Fx graph module which needs to be checkpointed. This operator adds
    "recomputable" tag to the nodes of the Fx graph that should be recomputed.

    The goal is to:
    1. Avoid using Dynamo to trace through saved tensor hooks.
    2. For selective checkpointing case, let AOTAutograd trace through
       saved tensor hooks but has special logic with TorchDispatchMode to override
       the usual saved_tensor_hooks fn logic in order to tag the nodes.
    3. Rely on the partitioners to actually duplicate the nodes.
    This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops (by setting fixed seed for each random op, see `replace_random_passes`).
    Therefore, the duplication of nodes, by design, respects the rng states in
    the forward and recomputed forward in backward.
    """

    def __init__(self) -> None:
        super().__init__("tag_activation_checkpoint", cacheable=True)

    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
        from torch.utils.checkpoint import checkpoint

        ckpt_signature = inspect.signature(checkpoint)
        checkpoint_keys = set()
        for name in ckpt_signature.parameters:
            if name in ("function", "args", "kwargs"):
                continue
            checkpoint_keys.add(name)

        # `preserve_rng_state` is not a regular kwarg
        checkpoint_keys.add("preserve_rng_state")

        checkpoint_kwargs = {
            name: kwargs[name] for name in kwargs if name in checkpoint_keys
        }
        gmod_kwargs = {
            name: kwargs[name] for name in kwargs if name not in checkpoint_keys
        }
        return checkpoint_kwargs, gmod_kwargs

    @staticmethod
    def tag_nodes(gmod, is_sac):
        from torch.utils.checkpoint import CheckpointPolicy

        unique_graph_id = next(uid)
        for node in gmod.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                node.meta["ac_graph_id"] = unique_graph_id
                if is_sac:
                    # For selective checkpointing, we will populate this tag later in _CachingTorchDispatchMode.
                    node.meta["recompute"] = None
                else:
                    # Under vanilla activation checkpointing, all nodes should be recomputed.
                    node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        return gmod

    def __call__(self, gmod, *args, **kwargs):
        dispatch_key_set = torch._ops._compute_keyset(
            args, kwargs, self.non_fallthrough_keys
        )
        dispatch_key = dispatch_key_set.highestPriorityTypeId()
        if dispatch_key == torch._C.DispatchKey.PreDispatch:
            # pyrefly: ignore [missing-attribute]
            return super().__call__(gmod, *args, **kwargs)

        return tag_activation_checkpoint_impl(gmod, *args, **kwargs)


tag_activation_checkpoint = TagActivationCheckpoint()


def tag_activation_checkpoint_impl(gmod, *args, **kwargs):
    import torch.fx.traceback as fx_traceback
    from torch.fx import Interpreter

    if "_checkpoint_context_fn" in gmod.meta:
        warning_once(
            log,
            """
Detected that context_fn is passed to torch.utils.checkpoint under torch.compile.
Please make sure the checkpointed region does not contain in-place ops (e.g. torch.relu_).
""",
        )
        # use_reentrant is set to False because this op is going to be traced.
        # And we ensure that AOT Autograd traces through the non reentrant
        # version of checkpointing.
        kwargs["use_reentrant"] = False
        # preserve_rng_state is set to False because we want to prevent AOTAutograd from tracing through
        # `torch.random.fork_rng` op (which is not supported yet under CUDA).
        # This doesn't mean that we don't preserve RNG state. Instead, we will always preserve RNG state
        # regardless of this flag (by doing RNG functionalization via `replace_random_passes` in Inductor
        # instead of in AOTAutograd).
        kwargs["preserve_rng_state"] = False
        kwargs["context_fn"] = gmod.meta["_checkpoint_context_fn"]
        # We first tag all nodes as "recompute" in this graph, and then we undo the "recompute" tag
        # for specific nodes in _CachingTorchDispatchMode in torch/utils/checkpoint.py.
        gmod = TagActivationCheckpoint.tag_nodes(gmod, is_sac=True)
        # Using interpreter allows preservation of metadata through torch.compile stack.
        with fx_traceback.preserve_node_meta():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(Interpreter(gmod).run, *args, **kwargs)
    else:
        gmod = TagActivationCheckpoint.tag_nodes(gmod, is_sac=False)
        # Using interpreter allows preservation of metadata through torch.compile stack.
        # TODO: We want to use the same `checkpoint(Interpreter(gmod).run, *args, **kwargs)` here
        # as the `context_fn != None` case, but that depends on in-place op support in TorchDispatchMode + torch.compile.
        # (for details on in-place op issue, run `test_compile_selective_checkpoint_inplace_op` unit test)
        with fx_traceback.preserve_node_meta():
            return Interpreter(gmod).run(*args)


@tag_activation_checkpoint.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    gmod: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    import torch.fx.traceback as fx_traceback
    from torch.fx import Interpreter

    if not proxy_mode.pre_dispatch:
        raise AssertionError(
            "post-dispatch mode should have inlined in the Autograd key"
        )
    example_out = tag_activation_checkpoint(gmod, *args, **kwargs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]
    proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)  # type: ignore[union-attr]
    qualname = proxy_mode.tracer.get_fresh_qualname("wrap_body")  # type: ignore[union-attr]

    # TODO (tmanlaibaatar) don't we need flat_apply here??
    # Dynamo already traced the gmod body without kwargs
    flat_args, _ = pytree.tree_flatten(args)
    with fx_traceback.preserve_node_meta():
        gmod_aten = reenter_make_fx(Interpreter(gmod).run)(*flat_args)
        gmod_aten.meta["_checkpoint_context_fn"] = gmod.meta["_checkpoint_context_fn"]
    proxy_mode.tracer.root.register_module(qualname, gmod_aten)  # type: ignore[union-attr]
    proxy_gmod = proxy_mode.tracer.unwrap_proxy(gmod_aten)  # type: ignore[union-attr, call-overload]
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        tag_activation_checkpoint,
        (proxy_gmod, *proxy_args),
        proxy_kwargs,
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )
