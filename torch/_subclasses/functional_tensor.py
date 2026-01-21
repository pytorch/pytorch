from __future__ import annotations

import contextlib
import warnings
import weakref
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import Self


if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Generator, Sequence
    from types import TracebackType

    from torch._functorch.pyfunctorch import FunctionalizeInterpreter
    from torch._ops import OpOverload

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch._ops import _get_dispatch_mode_pre_dispatch, TorchBindOpOverload
from torch._subclasses.meta_utils import is_sparse_any
from torch.utils._python_dispatch import (
    _detect_infra_mode,
    _disable_infra_mode,
    autograd_would_have_decomposed,
    return_and_correct_aliasing,
    TorchDispatchMode,
)


not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


# NOTE Some special handling for tensor conversion during export is needed.
# Normally, when tracing through the model with tensor.to(), the maybe-aliasing
# relationship between input and output tensors will be baked into the graph.
# For example, if we got a tensor with device cpu and call tensor.to("cpu"),
# it will become a no-op in the graph. For a whole graph capture, this is not
# sound so we need to do something different. Instead, in export we will try to
# preserve the tensor conversion by forcing a non-semantic-breaking aten::_to_copy
# operator to be traced in the graph, and subsequently banning mutations on all
# such converted tensors.
# In addition to patching .to() method call in functionalization, we will have to
# patch other similar methods like float() and cpu(), because they intentionally
# don't fall back to .to() methods, but have the same behavior as .to() according to
# pytorch document. https://pytorch.org/docs/stable/generated/torch.Tensor.float.html
# thus we simply force them to go through .to() call.
def _conversion_method_template(**extra_kwargs: Any) -> Callable[..., Any]:
    def _(self: FunctionalTensor, *args: Any, **kwargs: Any) -> Any:
        return self.to(*args, **{**kwargs, **extra_kwargs})

    return _


class FunctionalTensor(torch.Tensor):
    """
    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding
    FunctionalTensormode active, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    """

    elem: torch.Tensor
    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    _mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL

    # Note: The reason we add these extra keys to our FunctionalTensor subclass
    # is to mirror the behavior of C++ functionalization (we can choose to change this
    # later, as long as it doesn't break anything).
    # FunctionalTensorWrapper copies **all** dispatch keys from the inner tensor
    # to the wrapper, excluding functorch and python dispatch keys.
    # Here I'm trying to reuse the keyset the functorch wrapper subclasses copy,
    # except that they don't include ZeroTensor so I'm manually adding it in.
    _extra_dispatch_keys = torch._C._additional_keys_to_prop_for_wrapper_tensors.add(
        torch._C.DispatchKey.ZeroTensor
    )

    # These are all aten ops that correspond to metadata queries.
    # We want FunctionalTensor to be able to handle them directly.
    metadata_fns = [
        torch.ops.aten.is_contiguous.default,
        torch.ops.aten.is_contiguous.memory_format,
        torch.ops.aten.is_strides_like_format.default,
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.size.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.stride.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.storage_offset.default,
        torch.ops.aten.sym_storage_offset.default,
        torch.ops.aten.numel.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.dim.default,
        torch.ops.prim.device.default,
    ]

    # Used by auto_functionalize to determine base of tensors during inference mode.
    _inference_mode_base: Optional[FunctionalTensor] = None

    def __new__(cls, elem: torch.Tensor, mode: FunctionalTensorMode) -> Self:
        if not torch._is_functional_tensor(elem):
            raise AssertionError("elem must be a functional tensor")

        # In general, we'd like our functional tensor subclass to only be in charge of functionalization,
        # and defer to the inner subclass for all other functionality.
        # Example: If our inner tensor is a ZeroTensor, we would want to defer running the ZeroTensor fallback
        # until after we redispatch to our inner ZeroTensor.
        # However, there are a few keys that we need to mirror between the inner and outer tensors.
        #   Conjugate
        #   Negative
        # Why? These keys are used to test metadata queries, like `.is_conj()` and `.is_neg()`.
        # We **need** calls to is_conj() to return the same thing on the outer and inner tensors,
        # Because user code / framework code that branches like so needs to do the same thing
        # when it sees the outer FunctionalTensor:
        #     if (x.is_conj()) {
        #         return at::view_as_real(x.resolve_conj());
        #     } else {
        #         return at::view_as_real(x);
        #     }
        extra_dispatch_keys = (
            FunctionalTensor._extra_dispatch_keys & torch._C._dispatch_keys(elem)
        )

        out = torch.Tensor._make_wrapper_subclass(
            # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
            # Calling the overload that has kwargs causes us to go down the first overload path,
            # which will **always** specialize sizes.
            # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
            cls,
            elem.shape,  # sizes
            elem.stride() if not is_sparse_any(elem) else None,  # strides
            (
                elem.storage_offset() if not is_sparse_any(elem) else None
            ),  # storage_offset
            None,  # memory_format
            elem.dtype,  # dtype
            elem.layout,  # layout
            elem.device,  # device
            False,  # pin_memory
            elem.requires_grad,  # requires_grad
            None,  # dispatch_sizes_strides_policy
            False,  # dispatch_device
            False,  # dispatch_layout
            extra_dispatch_keys,  # _extra_dispatch_keys
        )
        torch._C._set_throw_on_mutable_data_ptr(out)
        out.elem = elem

        if (
            torch._export.config.enable_auto_functionalized_v2_for_export
            and torch.is_inference_mode_enabled()
            and torch._inductor.config.enable_auto_functionalized_v2
        ):
            if out.is_base_tensor():
                out._inference_mode_base = None
                # This assumes that the FunctionalTensor.elem does not change its storage after this point.
                # Otherwise this would be invalid.
                mode._storage_to_base[out.elem.untyped_storage()] = out
            else:
                out._inference_mode_base = mode._storage_to_base[
                    out.elem.untyped_storage()
                ]
                if out._inference_mode_base is None:
                    raise AssertionError("out._inference_mode_base must not be None")
        return out

    def __torch_dispatch__(  # type: ignore[override]
        self,
        func: OpOverload,
        types: Sequence[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        unrecognized_types = [
            t
            for t in types
            if t not in [torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor]
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        if kwargs is None:
            kwargs = {}
        # FunctionalTensor needs to plumb all metadata requests to the inner tensor.
        # In theory we don't have to do this - but if we want to service metadata requests here,
        # we need to carefully make sure all metadata is accurate (including metadata mutations)
        if func in FunctionalTensor.metadata_fns:
            # All metadata accesses should be plumbed to the inner tensor, that way we don't have to worry
            # about the problem of keeping metadata in sync between the wrapper and inner tensor.
            # This also alleviates us from having to manually handle metadata mutations on the wrapper.
            if len(kwargs) != 0:
                raise AssertionError("kwargs must be empty for metadata functions")
            if func in [
                torch.ops.aten.is_strides_like_format.default,
                torch.ops.aten.is_contiguous.memory_format,
            ]:
                if len(args) != 2 or not isinstance(args[0], FunctionalTensor):
                    raise AssertionError("Expected 2 args with FunctionalTensor first")
                return func(torch._from_functional_tensor(args[0].elem), args[1])
            if len(args) != 1 or not isinstance(args[0], FunctionalTensor):
                raise AssertionError("Expected 1 arg with FunctionalTensor")

            return func(torch._from_functional_tensor(args[0].elem))
        # Originally I tried to implement my subclass without giving it a torch_dispatch, but I gave up:
        # - _make_wrapper_subclass requires a __torch_dispatch__
        # - If we want to use _make_subclass(), we have a problem: the subclass will share a TensorImpl with the inner tensor,
        #   which is of type FunctionalTensorWrapper! We explicitly do not want our wrapper to be a FunctionalTensorWrapper.
        # - If we use the default tensor.__new__(), we have another problem: it returns inner_tensor.alias(),
        #   which causes every subclass created above autograd to have autograd view metadata
        #   (in addition to also being a FunctionalTensorWrapper).
        raise RuntimeError(
            "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
        )

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"FunctionalTensor({repr(self.elem)})"

    @staticmethod
    def to_functional(x: torch.Tensor) -> FunctionalTensor:
        # We will do the wrapping for the user.

        if torch._is_functional_tensor(x):
            raise AssertionError("x must not already be a functional tensor")
        # The only autograd metadata we care about on the FunctionalTensor is:
        # - requires_grad (so autograd runs)
        # - is_leaf (so that mutations on graph inputs that are not leaves are allowed by the autograd engine)
        #   this is handled by FunctionalTensor.to_functional
        x_functional = torch._to_functional_tensor(x)
        # Technically the FunctionalTensormode here is unnecessary,
        # but it avoids spurious NotImplemented logs during `ProxyTorchDispatchMode` tracing.
        # _mirror_autograd_meta_to queries tensor sizes,
        # and otherwise the sym_size() call will go to the proxy mode before hitting
        # FunctionalTensor.__torch_dispatch__

        functional_mode = _detect_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
        if functional_mode is None:
            raise AssertionError("functional_mode must not be None")

        with functional_mode:
            torch._mirror_autograd_meta_to(x, x_functional)  # type: ignore[attr-defined]
            out = FunctionalTensor(x_functional, functional_mode)
            torch._mirror_autograd_meta_to(x_functional, out)  # type: ignore[attr-defined]
        return out

    def from_functional(self) -> torch.Tensor:
        torch._sync(self)
        return torch._from_functional_tensor(self.elem)

    def is_base_tensor(self) -> bool:
        return torch._is_functional_tensor_base(self.elem)

    def replace_(self, output: torch.Tensor) -> None:
        torch._functionalize_replace(self.elem, output)

    def commit_update(self) -> None:
        torch._functionalize_commit_update(self.elem)

    def sync(self) -> None:
        torch._functionalize_sync(self.elem)

    def mark_mutation_hidden_from_autograd(self) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(self.elem)

    def tolist(self) -> Any:
        if self.elem.dim() == 0:
            return self.elem.item()
        elif self.elem.dim() == 1:
            return [elem.item() for elem in self.elem]
        else:
            return [elem.tolist() for elem in self.elem]

    def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if _detect_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL).export:
            torch.ops.aten._assert_tensor_metadata(
                self,
                dtype=self.dtype,
                device=self.device,
                layout=self.layout,
            )
        return super().to(*args, **kwargs)

    # pyrefly: ignore[bad-override]
    def cuda(
        self, device: torch.device | int | str | None = None, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        device = device or torch.cuda.current_device()
        if len(args) > 0:
            return self.to(device, *args, **kwargs)
        else:
            return self.to(device=device, **kwargs)

    char = _conversion_method_template(dtype=torch.int8)
    cpu = _conversion_method_template(device=torch.device("cpu"))
    bfloat16 = _conversion_method_template(dtype=torch.bfloat16)
    byte = _conversion_method_template(dtype=torch.uint8)
    double = _conversion_method_template(dtype=torch.float64)
    float = _conversion_method_template(dtype=torch.float32)
    bool = _conversion_method_template(dtype=torch.bool)
    half = _conversion_method_template(dtype=torch.float16)
    int = _conversion_method_template(dtype=torch.int32)
    long = _conversion_method_template(dtype=torch.int64)

    # TODO(sparse-team): fixes #133174 but can we do without the relay?
    def to_dense(
        self,
        dtype: torch.dtype | None = None,
        *,
        masked_grad: builtins.bool | None = None,
    ) -> torch.Tensor:
        return self.elem.to_dense()

    @property
    # pyrefly: ignore[bad-override]
    def layout(self) -> torch.layout:
        return self.elem.layout

    def __bool__(self) -> builtins.bool:
        return bool(self.item())


class FunctionalTensorMode(TorchDispatchMode):
    def __init__(
        self,
        pre_dispatch: bool = False,
        export: bool = False,
        _allow_token_discovery: bool = False,
    ) -> None:
        super().__init__()
        self.export = export
        self.is_on_stack = False
        self.enter_stack = []
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
        self.pre_dispatch = pre_dispatch
        # This will be turned off later for pre-dispatch functionalization
        self._dispatch_key = torch._C.DispatchKey.PreDispatch if pre_dispatch else None  # type: ignore[attr-defined]
        # Map of effect type (ex. _EffectType.ORDERED) to a token. The tokens help keep
        # track of the ordering between side effectful operations.
        self._tokens: dict[Any, torch.Tensor] = {}

        # Filled after forward tracing.
        self._tokens_forward_output: dict[Any, torch.Tensor] = {}

        # Functionalization runs twice in AOTAutograd, once in
        # `run_functionalized_fw_and_collect_metadata` to collect metadata to
        # see which tensors need to be functionalized and discover how many
        # tokens we need, and another time in `make_fx` which does the actual
        # tracing to replace ops with their functional variants and handling
        # side-effectful ops. In the second stage there should be no token
        # discovery. This flag distinguishes between the two stages.
        self._allow_token_discovery = _allow_token_discovery

        self._storage_to_base: weakref.WeakKeyDictionary[
            torch.storage.UntypedStorage, Optional[FunctionalTensor]
        ] = weakref.WeakKeyDictionary()

    # No-op if FunctionalTensorMode is already in use
    def __enter__(self) -> Self:
        def _get_prev_mode() -> Optional[FunctionalTensorMode]:
            if self._dispatch_key == torch._C.DispatchKey.PreDispatch:
                return _get_dispatch_mode_pre_dispatch(
                    torch._C._TorchDispatchModeKey.FUNCTIONAL
                )
            return torch._C._get_dispatch_mode(
                torch._C._TorchDispatchModeKey.FUNCTIONAL
            )

        if _get_prev_mode() is None:
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            self.enter_stack.append(False)
            return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        is_on_stack = self.enter_stack.pop()
        if is_on_stack:
            super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Sequence[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        unrecognized_types = [
            t
            for t in types
            if not issubclass(t, torch._subclasses.FakeTensor)
            and t not in [torch.Tensor, FunctionalTensor]
        ]

        if unrecognized_types:
            not_implemented_log.debug(
                "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        def _can_decompose(func: OpOverload) -> bool:
            # See https://github.com/pytorch/pytorch/pull/115258#issuecomment-1900755832
            # Never decompose dropout in export
            if self.export and func is torch.ops.aten.dropout.default:
                return False

            # We unconditionally decompose ops that are maybe aliasing or mutating ops
            from torch._decomp import _should_decompose_because_unsafe_op

            if _should_decompose_because_unsafe_op(func):
                return True

            # (1) we unconditionally decompose maybe-aliasing or maybe-mutating ops,
            # because we must know statically of an op mutates or aliasing in order to functionalize it properly
            # (2) for mutating ops that have CompositeImplicit decomps, we choose to decompose them today.
            # In theory, we could walk this back and avoid decomposing them later if we need to.
            alias_info_present = any(arg.alias_info for arg in func._schema.arguments)
            if alias_info_present or func._schema.is_mutable:
                return True

            # If we are here, it means we are seeing functional composite op.
            # For pre-dispatch IR, we don't want to decompose this op
            # For post-dispatch IR, we do want to decompose this op. it is fine
            # to decompose here even if you want to preserve a CIA in post-dispatch export
            # because we already override decompose behaviour so it will do the
            # right thing.
            if self.export:
                if self.pre_dispatch:
                    # If it is CIA custom op, we warn that we are assuming this op is indeed functional.
                    if func.namespace not in ["aten", "prim"] and func._can_decompose():
                        warnings.warn(
                            f"At pre-dispatch tracing, we assume that any custom op marked with "
                            f"CompositeImplicitAutograd and have functional schema are safe to not decompose. "
                            f"Found {func} to be one such op.",
                            stacklevel=2,
                        )
                    return False
                return True

            # in normal torch.compile IR, we only decompose an op if autograd
            # would have decomposed it (NB: autograd may have been skipped if
            # we are in inference mode)
            # TODO: the flatten here can potentially be deduped with the
            # unwrapping pytree_map later
            flat_args_kwargs, _ = pytree.tree_flatten((args, kwargs))
            return autograd_would_have_decomposed(func, flat_args_kwargs)

        if (
            func not in FunctionalTensor.metadata_fns
            and _can_decompose(func)
            # Not all funcs from __torch_dispatch__ are actual dispatcher ops,
            # e.g. prim.device
            and torch._C._dispatch_has_kernel(func.name())
        ):
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        def wrap(x: object) -> object:
            # Only wrap our outputs in subclasses if the inner functionalization call
            # also wrapped outputs into FunctionalTensorWrappers.
            # When can this happen? e.g. `torch.div(2, 2)`
            if isinstance(x, FunctionalTensor):
                raise AssertionError("x must not be a FunctionalTensor in wrap()")
            if isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
                return FunctionalTensor(x, self)
            return x

        def unwrap(x: FunctionalTensor) -> torch.Tensor:
            return x.elem

        from torch._higher_order_ops.auto_functionalize import (
            can_auto_functionalize,
            do_auto_functionalize,
            do_auto_functionalize_v2,
        )

        if can_auto_functionalize(
            func
        ) and not torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.Functionalize
        ):
            import torch._export.config as export_config
            import torch._inductor.config as inductor_config

            if torch.compiler.is_exporting():
                if export_config.enable_auto_functionalized_v2_for_export:
                    return do_auto_functionalize_v2(self, func, args, kwargs)

                return do_auto_functionalize(self, func, args, kwargs)

            if inductor_config.enable_auto_functionalized_v2:
                return do_auto_functionalize_v2(self, func, args, kwargs)
            return do_auto_functionalize(self, func, args, kwargs)

        from torch._higher_order_ops.effects import handle_effects, has_effects

        if has_effects(func):
            if torch._C._dispatch_has_kernel_for_dispatch_key(
                func.name(), torch._C.DispatchKey.Functionalize
            ):
                raise AssertionError(
                    f"func {func.name()} with effects should not have a kernel for Functionalize dispatch key"
                )
            return handle_effects(
                self._allow_token_discovery, self._tokens, func, args, kwargs
            )

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(
            FunctionalTensor, unwrap, (args, kwargs)
        )

        # Expectation: functionalization should not **already** be enabled above our mode.
        # Why would that be bad? when we return a FunctionalTensor here, we don't want functionalization
        # to run above this mode and further wrap that output in **another** C++ FunctionalTensorWrapper.
        is_included = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.Functionalize
        )
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.Functionalize
        )
        if not is_excluded and is_included:
            raise AssertionError(
                "Functionalization should not already be enabled above this mode"
            )
        include_to_set = (
            torch._C._dispatch_tls_local_include_set()
            | torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )
        exclude_to_set = (
            torch._C._dispatch_tls_local_exclude_set().remove(
                torch._C.DispatchKey.Functionalize
            )
            - FunctionalTensor._extra_dispatch_keys
        )

        if isinstance(func, TorchBindOpOverload):
            # When the function is a TorchBindOpOverload, meaning some of the
            # inputs are FakeScriptObjects, we need to skip c++ dispatcher and
            # dispatch in python because C++ dispatcher will check the schema
            # and cannot recognize FakeScriptObject.
            ctx = PythonFunctionalizeAPI()
            fully_unwrapped_args = ctx.unwrap_tensors(args)
            fully_unwrapped_kwargs = ctx.unwrap_tensors(
                kwargs  # pyrefly: ignore[bad-argument-type]
            )
            outs_unwrapped = func(
                *fully_unwrapped_args,
                **fully_unwrapped_kwargs,
            )
            outs_wrapped = ctx.wrap_tensors(outs_unwrapped)
        else:
            # All we want to do here is reuse the existing C++ functionalization logic.
            # This requires swizzling our TLS dispatch keys so that the Functionalize key is active.
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                try:
                    # By default for python functionalization (for AOTAutograd), we reapply views.
                    old_apply_views = torch._functionalize_enable_reapply_views(True)  # type: ignore[attr-defined]

                    # Sometimes these functions cannot be directly dispatched to functionalize key
                    # because args are sometimes not functional tensors for some reason?
                    if func in FunctionalTensor.metadata_fns:
                        outs_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)
                        outs_wrapped = pytree.tree_map_only(
                            torch.Tensor, wrap, outs_unwrapped
                        )
                    else:
                        # Note: [Functionalization View Replay Annotation]
                        # When functionalization encounters a mutation, it handles aliases by lazily regenerating the aliases
                        # at the first time they are next used.
                        # This is a problem when plumbing user annotations during tracing. We want the view ops from view replay
                        # to have the same annotation that the user specified on the original views. But view replay in
                        # functionalization happens the next time the alias is used (e.g. second_op(alias_with_pending_mutation)),
                        # so when we regenerate views before calling into second_op, those views will end up getting the metadata
                        # for second_op!
                        #
                        # Instead, we need to remember the node metadata from the original views, and ensure that this node metadata
                        # is globally set when we lazily perform view replay.
                        # The globally set metadata will be used to populate the fx node created for the replayed operation.
                        if m := torch._C._get_dispatch_mode(
                            torch._C._TorchDispatchModeKey.PROXY
                        ):
                            for a in pytree.tree_leaves([args, kwargs]):
                                if not isinstance(a, FunctionalTensor):
                                    continue
                                curr_node = m.tracer.tensor_tracker[
                                    torch._from_functional_tensor(a.elem)
                                ].proxy.node
                                with fx_traceback.set_current_replay_node(curr_node):
                                    torch._sync(a)

                        # When we dispatch to the C++ functionalization kernel, we might need to jump back to the
                        # PreDispatch mode stack afterwards, to handle any other PreDispatch modes underneath
                        # FunctionalTensorMode. If we call func() directly, we would need to exclude PreDispatch
                        # from the TLS in order to avoid infinite looping, but this would prevent us from coming
                        # back to PreDispatch later
                        outs_unwrapped = func._op_dk(
                            torch._C.DispatchKey.Functionalize,
                            *args_unwrapped,
                            **kwargs_unwrapped,
                        )

                        if self.export:
                            if func is torch.ops.aten.dropout.default:
                                torch._freeze_functional_tensor(outs_unwrapped)  # type: ignore[attr-defined]
                        outs_wrapped = pytree.tree_map_only(
                            torch.Tensor, wrap, outs_unwrapped
                        )
                finally:
                    torch._disable_functionalization()
                    torch._functionalize_enable_reapply_views(old_apply_views)  # type: ignore[attr-defined]

        is_included = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.Functionalize
        )
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.Functionalize
        )
        if not is_excluded and is_included:
            raise AssertionError(
                "Functionalization should not already be enabled above this mode after dispatch"
            )

        if (
            # If no outputs are our functional subclass, then don't try to fix up aliasing
            not any(
                isinstance(x, FunctionalTensor)
                for x in pytree.tree_leaves(outs_wrapped)
            )
            # Since lift_fresh lifts its argument into a functional tensor, we can skip the
            # aliasing correction step. Otherwise, we would be setting the storage of a
            # lifted tensor to that of an unlifted tensor.
            # Ref: https://github.com/pytorch/pytorch/issues/111506
            or func is torch.ops.aten.lift_fresh.default
        ):
            return outs_wrapped
        # for metadata mutations, need to manually mutate the metadata of the FunctionalTensor wrapper
        if (
            torch.Tag.inplace_view in func.tags
            and func is not torch.ops.aten.set_.source_Tensor
        ):
            with torch.utils._mode_utils.no_dispatch():
                func(*args, **kwargs)
        # Wrapper tensor subclasses do not have correct aliasing info! Use this util to manually correct the output aliasing.
        # inplace ops like `aten.add_()` are expected to return inputs **directly**, instead of creating fresh tensor objects.
        # Use this util to figure out the right thing to return.
        # If none of our inputs were wrapped, then we have no FunctionalTensor outputs that we need to fix up storages for.
        return return_and_correct_aliasing(func, args, kwargs, outs_wrapped)

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True


@contextlib.contextmanager
def disable_functional_mode() -> Generator[None, None, None]:
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)


# This is similar to torch.func.functionalize, but:
# - It uses FunctionalTensorMode, and FunctionalTensor (a python subclass).
#   One important advantage to using this mode is that it will let us
#   run functionalization underneath __torch_dispatch__,
#   which we need in AOTAutograd.
# - Doing so means that it does not automatically compose with other
#   functorch transforms, since these transforms always run above __torch_dispatch__.
#   That's why this util lives here, and not in functorch.
def dispatch_functionalize(
    func: Callable[..., Any], mode: FunctionalTensorMode = FunctionalTensorMode()
) -> Callable[..., Any]:
    # TODO: pull these from aot autograd
    def to_fun(t: object) -> object:
        if isinstance(t, torch.Tensor):
            return FunctionalTensor.to_functional(t)
        return t

    def from_fun(t: object) -> object:
        if not isinstance(t, FunctionalTensor):
            # quick sanity check
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    raise AssertionError(
                        "Non-FunctionalTensor torch.Tensor should not be a functional tensor"
                    )
            return t
        torch._sync(t)
        return torch._from_functional_tensor(t.elem)

    def inner(*args: Any, **kwargs: Any) -> Any:
        disable_above = torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )
        with disable_above, mode:
            func_args = pytree.tree_map_only(torch.Tensor, to_fun, args)
            func_kwargs = pytree.tree_map_only(torch.Tensor, to_fun, kwargs)
            func_outputs = func(*func_args, **func_kwargs)
            outputs = pytree.tree_map_only(FunctionalTensor, from_fun, func_outputs)

            return outputs

    return inner


class BaseFunctionalizeAPI(ABC):
    @abstractmethod
    def wrap_tensors(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        pass

    @abstractmethod
    def unwrap_tensors(self, args: torch.Tensor | tuple[torch.Tensor, ...]) -> Any:
        pass

    @abstractmethod
    def functionalize(self, inner_f: Callable[..., Any]) -> Callable[..., Any]:
        pass

    @abstractmethod
    def redispatch_to_next(self) -> AbstractContextManager[None]:
        pass

    @abstractmethod
    def replace(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        pass

    @abstractmethod
    def commit_update(self, tensor: torch.Tensor) -> None:
        pass

    @abstractmethod
    def sync(self, tensor: torch.Tensor) -> None:
        pass

    @abstractmethod
    def mark_mutation_hidden_from_autograd(self, tensor: torch.Tensor) -> None:
        pass


class PythonFunctionalizeAPI(BaseFunctionalizeAPI):
    def __init__(
        self, mode: Optional[FunctionalTensorMode] = None, pre_dispatch: bool = False
    ) -> None:
        super().__init__()
        self.mode = mode if mode else FunctionalTensorMode()
        self.pre_dispatch = pre_dispatch

    def wrap_tensors(self, args: tuple[Any]) -> tuple[Any]:
        with self.mode:
            return torch.utils._pytree.tree_map_only(
                torch.Tensor, FunctionalTensor.to_functional, args
            )

    def unwrap_tensors(
        self, args: Union[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]]
    ) -> Any:
        return torch.utils._pytree.tree_map_only(
            FunctionalTensor, FunctionalTensor.from_functional, args
        )

    def functionalize(self, inner_f: Callable) -> Callable:
        return dispatch_functionalize(inner_f, self.mode)

    def redispatch_to_next(self) -> AbstractContextManager[None]:
        # [NOTE] We don't do anything here because at the time
        # we exercise this path, we would have already popped the
        # FunctionalTensorMode from mode stack. Since FunctionalTensorMode
        # is now stateful, it is better to explicitly pass in correct mode
        # directly instead of globally setting it.
        return contextlib.nullcontext()

    def replace(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        if not isinstance(input_tensor, FunctionalTensor):
            raise AssertionError(
                f"input_tensor must be a FunctionalTensor, got {type(input_tensor)}"
            )
        if isinstance(output_tensor, FunctionalTensor):
            raise AssertionError("output_tensor must not be a FunctionalTensor")
        input_tensor.replace_(output_tensor)

    def commit_update(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, FunctionalTensor):
            raise AssertionError(
                f"tensor must be a FunctionalTensor, got {type(tensor)}"
            )
        tensor.commit_update()

    def sync(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, FunctionalTensor):
            raise AssertionError(
                f"tensor must be a FunctionalTensor, got {type(tensor)}"
            )
        tensor.sync()

    def mark_mutation_hidden_from_autograd(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, FunctionalTensor):
            raise AssertionError(
                f"tensor must be a FunctionalTensor, got {type(tensor)}"
            )
        tensor.mark_mutation_hidden_from_autograd()


class CppFunctionalizeAPI(BaseFunctionalizeAPI):
    def wrap_tensors(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional

        return _wrap_all_tensors_to_functional(args, level=0)

    def unwrap_tensors(
        self, args: Union[torch.Tensor, tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )

        return _unwrap_all_tensors_from_functional(args, reapply_views=_reapply_views())

    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(inner_f)

    def redispatch_to_next(self) -> AbstractContextManager[None]:
        return torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        )

    def replace(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        torch._functionalize_replace(input_tensor, output_tensor)

    def commit_update(self, tensor: torch.Tensor) -> None:
        torch._functionalize_commit_update(tensor)

    def sync(self, tensor: torch.Tensor) -> None:
        torch._functionalize_sync(tensor)

    def mark_mutation_hidden_from_autograd(self, tensor: torch.Tensor) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)


class FunctorchFunctionalizeAPI(BaseFunctionalizeAPI):
    def __init__(self, interpreter: FunctionalizeInterpreter) -> None:
        self.interpreter = interpreter

    def wrap_tensors(self, args: tuple[Any]) -> tuple[Any]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional

        return _wrap_all_tensors_to_functional(args, level=self.interpreter.level())

    def unwrap_tensors(
        self, args: Union[torch.Tensor, tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        from torch._functorch.eager_transforms import (
            _unwrap_all_tensors_from_functional,
        )

        return _unwrap_all_tensors_from_functional(
            args, reapply_views=self.interpreter.functionalize_add_back_views()
        )

    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(
            inner_f,
            remove=(
                "mutations_and_views"
                if self.interpreter.functionalize_add_back_views()
                else "mutations"
            ),
        )

    def redispatch_to_next(self) -> AbstractContextManager[None]:
        return self.interpreter.lower()

    def replace(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        torch._functionalize_replace(input_tensor, output_tensor)

    def commit_update(self, tensor: torch.Tensor) -> None:
        torch._functionalize_commit_update(tensor)

    def sync(self, tensor: torch.Tensor) -> None:
        torch._functionalize_sync(tensor)

    def mark_mutation_hidden_from_autograd(self, tensor: torch.Tensor) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)


def mb_unwrap_functional_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, FunctionalTensor):
        return torch._from_functional_tensor(tensor.elem)
    return tensor
