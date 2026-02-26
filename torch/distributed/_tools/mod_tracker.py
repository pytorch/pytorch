# mypy: allow-untyped-defs
import contextlib
import warnings
import weakref
from collections.abc import Callable
from typing import Any

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten, tree_unflatten


__all__ = ["ModTracker"]


# --- Compile-safe hook helpers (used by register_compile_safe_hooks) ---

# Hook types for compile-safe hooks
# pre_fw: (fqn, tensors) -> None
# pre_bw, post_bw: (fqn, tensors_or_none) -> None
# Backward hooks may receive None entries for tensors that didn't require grad.
_CompileSafeHookFn = Callable[[str, tuple[torch.Tensor | None, ...]], None]
# post_fw: (fqn, inputs, outputs) -> None
_CompileSafePostFwHookFn = Callable[
    [str, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], None
]


def _make_compile_safe_leaf(
    fqn: str,
    fw_hook: "_CompileSafeHookFn | None",
    bw_hook: "_CompileSafeHookFn | None",
):
    """Create a @leaf_function that calls fw_hook in forward and bw_hook in backward.

    The fqn and hook callables are captured in the closure of the real function.
    The fake function is a pure tensor pass-through (no closures needed).

    A per-module ``torch.autograd.Function`` (GradHook) is ALWAYS inserted for
    tensors with ``requires_grad``. This is required to maintain autograd edges
    under ``torch.compile`` — without it, gradient flow through the leaf is
    broken. The GradHook optionally calls ``bw_hook`` if one is provided.
    """
    from torch._dynamo.decorators import leaf_function

    class _GradHook(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *tensors: torch.Tensor):  # pyrefly: ignore[bad-override]
            return tensors

        @staticmethod
        def backward(ctx: Any, *grads: torch.Tensor | None):
            if bw_hook is not None:
                bw_hook(fqn, grads)
            return grads

    @leaf_function
    def _leaf(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if fw_hook is not None:
            fw_hook(fqn, tensors)

        if any(t.requires_grad for t in tensors):
            result = _GradHook.apply(*tensors)
            if not isinstance(result, tuple):
                result = (result,)
            tensors = result

        return tensors

    @_leaf.register_fake  # pyrefly: ignore[missing-attribute]
    def _fake(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tensors

    return _leaf


def _make_compile_safe_forward_hook(leaf_fn):
    """Module forward hook that routes output tensors through a leaf function.

    Input capture is handled by the pre-forward hook; this hook only processes outputs."""

    def hook(module, input, kwargs, output):
        flat, spec = tree_flatten(output)
        tensor_indices = [i for i, v in enumerate(flat) if isinstance(v, torch.Tensor)]
        if not tensor_indices:
            return output
        tensors = tuple(flat[i] for i in tensor_indices)
        new_tensors = leaf_fn(*tensors)
        for idx, new_t in zip(tensor_indices, new_tensors):
            flat[idx] = new_t
        return tree_unflatten(flat, spec)

    return hook


def _make_compile_safe_pre_forward_hook(leaf_fn):
    """Module forward pre-hook that routes input tensors through a leaf function."""

    def hook(module, args, kwargs):
        flat, spec = tree_flatten((args, kwargs))
        tensor_indices = [i for i, v in enumerate(flat) if isinstance(v, torch.Tensor)]
        if not tensor_indices:
            return args, kwargs
        tensors = tuple(flat[i] for i in tensor_indices)
        new_tensors = leaf_fn(*tensors)
        for idx, new_t in zip(tensor_indices, new_tensors):
            flat[idx] = new_t
        new_args, new_kwargs = tree_unflatten(flat, spec)
        return new_args, new_kwargs

    return hook


def _register_compile_safe_hooks(
    model: torch.nn.Module,
    pre_fw_hook: "_CompileSafeHookFn | None" = None,
    post_fw_hook: "_CompileSafePostFwHookFn | None" = None,
    pre_bw_hook: "_CompileSafeHookFn | None" = None,
    post_bw_hook: "_CompileSafeHookFn | None" = None,
    module_filter: "Callable[[str, torch.nn.Module], bool] | None" = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register compile-safe hooks on a model's submodules. Returns handles."""
    handles: list[torch.utils.hooks.RemovableHandle] = []
    need_pre = pre_fw_hook is not None or post_bw_hook is not None
    need_post = post_fw_hook is not None or pre_bw_hook is not None

    for fqn, module in model.named_modules():
        # Skip root module: under torch.compile, its hooks fire outside the
        # compiled region (at OptimizedModule level), causing dynamo errors.
        if fqn == "":
            continue
        if module_filter is not None and not module_filter(fqn, module):
            continue

        # --- Input capture for post_fw ---
        # post_fw_hook receives (fqn, inputs, outputs). Inputs are
        # captured via a pre-hook leaf. When need_pre is True, we
        # piggyback on the pre_fw leaf; otherwise a dedicated capture
        # pre-hook is created. The capture leaf MUST include a GradHook
        # (even a no-op one) to maintain autograd edges under compile.
        captured_inputs_ref: list | None = None
        if post_fw_hook is not None:
            captured_inputs_ref = [()]

        # Pre-forward leaf handles: pre_fw callback + post_bw GradHook
        # (GradHook on inputs fires AFTER module backward = post_bw)
        if need_pre:
            if captured_inputs_ref is not None:
                # Piggyback input capture onto the pre_fw leaf
                _orig_pre_fw = pre_fw_hook

                def _pre_fw_with_capture(
                    _fqn, tensors, _ref=captured_inputs_ref, _orig=_orig_pre_fw
                ):
                    _ref[0] = tensors
                    if _orig is not None:
                        _orig(_fqn, tensors)

                pre_leaf_fw = _pre_fw_with_capture
            else:
                pre_leaf_fw = pre_fw_hook

            pre_leaf = _make_compile_safe_leaf(
                fqn, fw_hook=pre_leaf_fw, bw_hook=post_bw_hook
            )
            handles.append(
                module.register_forward_pre_hook(
                    _make_compile_safe_pre_forward_hook(pre_leaf),
                    with_kwargs=True,
                )
            )

        elif captured_inputs_ref is not None:
            # No pre_fw/post_bw but post_fw needs inputs. Create a
            # dedicated capture pre-hook. The GradHook inside the leaf
            # (always present) maintains autograd edges under compile.
            capture_leaf = _make_compile_safe_leaf(
                fqn,
                # pyrefly: ignore[bad-argument-type]
                fw_hook=lambda _fqn,
                tensors,
                _ref=captured_inputs_ref: _ref.__setitem__(0, tensors),
                bw_hook=None,
            )
            handles.append(
                module.register_forward_pre_hook(
                    _make_compile_safe_pre_forward_hook(capture_leaf),
                    with_kwargs=True,
                )
            )

        # Post-forward leaf handles: post_fw callback + pre_bw GradHook
        # (GradHook on outputs fires BEFORE module backward = pre_bw)
        if need_post:
            if post_fw_hook is not None:
                assert captured_inputs_ref is not None
                _user_post_fw = post_fw_hook

                def _combined_post_fw(
                    _fqn,
                    output_tensors,
                    _ref=captured_inputs_ref,
                    _hook=_user_post_fw,
                ):
                    _hook(_fqn, _ref[0], output_tensors)

                leaf_fw_hook = _combined_post_fw
            else:
                leaf_fw_hook = None

            post_leaf = _make_compile_safe_leaf(
                fqn, fw_hook=leaf_fw_hook, bw_hook=pre_bw_hook
            )
            handles.append(
                module.register_forward_hook(
                    _make_compile_safe_forward_hook(post_leaf),
                    with_kwargs=True,
                )
            )

    return handles


class ModTracker:
    """
    ``ModTracker`` is a context manager that tracks the nn.Module hierarchy during execution
    so that other system can query which Module is currently being executed (or its backward is being
    executed).

    You can access the ``parents`` attribute on this context manager to get the set of all the
    Modules currently being executed via their fqn (fully qualified name, also used as the key within
    the state_dict).
    You can access the ``is_bw`` attribute to know if you are currently running in backward or not.

    Note that ``parents`` is never empty and always contains the "Global" key. The ``is_bw`` flag
    will remain ``True`` after the forward until another Module is executed. If you need it to be
    more accurate, please submit an issue requesting this. Adding a map from fqn to the module instance
    is possible but not done yet, please submit an issue requesting this if you need it.

    Example usage

    .. code-block:: python

        mod = torch.nn.Linear(2, 2)

        with ModTracker() as tracker:
            # Access anything during the forward pass
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias

            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    """

    parents: set[str]
    """
    A Set containing the fqn for each module currently running their forward
    """

    def __init__(self):
        self.parents = {"Global"}
        self._active_module_cnt = {}
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._seen_modules: weakref.WeakSet = weakref.WeakSet()
        self._has_callback = False
        self._post_bw_callbacks_to_enqueue: list[Callable] = []
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    def _maybe_set_engine_callback(self):
        # This assumes no concurrent calls to backward
        if self._has_callback:
            return

        for post_bw_callback in reversed(self._post_bw_callbacks_to_enqueue):
            torch.autograd.Variable._execution_engine.queue_callback(post_bw_callback)
        self._post_bw_callbacks_to_enqueue.clear()

        def callback():
            self.parents = {"Global"}
            self._has_callback = False

        torch.autograd.Variable._execution_engine.queue_callback(callback)
        self._has_callback = True

    @property
    def is_bw(self):
        """
        A boolean marking if this is currently running during the backward pass or not
        """
        return torch._C._current_graph_task_id() != -1

    def get_known_fqn(self, mod):
        """
        Return the fqn for the given module if it is known to the ``ModTracker``, otherwise ``None``.
        """
        return self._known_modules.get(mod, None)

    def register_user_hooks(
        self,
        pre_fw_hook: Callable | None = None,
        post_fw_hook: Callable | None = None,
        pre_bw_hook: Callable | None = None,
        post_bw_hook: Callable | None = None,
    ):
        """
        Registers user-specified hooks to be called before/after the forward/backward pass for each
        module tracked by the ``ModTracker``. One or more can be ``None``.
        Args:
            pre_fw_hook (Callable, optional): A hook to be called before the forward pass for the
                module. It should have the following signature:
                pre_fw_hook (module, input) -> None
            post_fw_hook (Callable, optional): A hook to be called after the forward pass for the
                module. It should have the following signature:
                post_fw_hook (module, input, output) -> None
            pre_bw_hook (Callable, optional): A multi-grad hook to be called on all the outputs of
                the module that require gradients. It should have the following signature:
                pre_bw_hook (module, grad_output) -> None
            post_bw_hook (Callable, optional): A multi-grad hook to be called on all the inputs of
                the module that require gradients. It should have the following signature:
                post_bw_hook (module, grad_input) -> None
        Raises:
            AssertionError: If a new hook is provided when one is already registered.
        Note:
            If the module is not alive during the backward pass, the pre_bw_hook and post_bw_hook will
            will receive None as the module argument.
            The module fqn will be present in the ``parents`` attribute when each of the hooks is called.
            Hooks are intended to be used as markers only not to modify the inputs/outputs.
        """

        def set_hook(hook, user_hook, hook_name):
            if hook is not None and user_hook is not None:
                raise AssertionError(
                    f"Only one {hook_name} can be registered at a time"
                    f" Clear the existing hook by calling ``clear_user_hooks`` before registering a new one"
                )
            return hook

        self._user_pre_fw_hook = set_hook(
            pre_fw_hook, self._user_pre_fw_hook, "pre_fw_hook"
        )
        self._user_post_fw_hook = set_hook(
            post_fw_hook, self._user_post_fw_hook, "post_fw_hook"
        )
        self._user_pre_bw_hook = set_hook(
            pre_bw_hook, self._user_pre_bw_hook, "pre_bw_hook"
        )
        self._user_post_bw_hook = set_hook(
            post_bw_hook, self._user_post_bw_hook, "post_bw_hook"
        )

    def clear_user_hooks(self):
        """
        Clears the user specified hooks registered with ``register_user_hooks``
        """
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    @staticmethod
    @contextlib.contextmanager
    def compile_safe_hooks(
        model: torch.nn.Module,
        pre_fw_hook: "_CompileSafeHookFn | None" = None,
        post_fw_hook: "_CompileSafePostFwHookFn | None" = None,
        pre_bw_hook: "_CompileSafeHookFn | None" = None,
        post_bw_hook: "_CompileSafeHookFn | None" = None,
        module_filter: "Callable[[str, torch.nn.Module], bool] | None" = None,
    ):
        """
        Context manager that registers compile-safe hooks at nn.Module boundaries.

        Uses ``@leaf_function`` to make hooks opaque to dynamo, so they work under
        ``torch.compile(backend="aot_eager", fullgraph=True)`` without graph breaks.
        Also compatible with ``aot_export_joint_with_descriptors``. The callback body
        runs eagerly at runtime and can do arbitrary Python (logging, assertions,
        hash computation, external library calls, etc.).

        This is a ``@staticmethod`` — it does not require a ``ModTracker`` instance
        and does not interact with ``ModTracker``'s tracking state (``parents``,
        ``is_bw``, etc.). It is independent of ``register_user_hooks``.

        The four hooks correspond to these execution points::

            Forward:   pre_fw(inputs) -> module.forward -> post_fw(inputs, outputs)
            Backward:  pre_bw(grad_outputs) -> module.backward -> post_bw(grad_inputs)

        Args:
            model: The model to register hooks on.
            pre_fw_hook: Called before each module's forward.
                Signature: ``(fqn: str, input_tensors: tuple[Tensor, ...]) -> None``
            post_fw_hook: Called after each module's forward with both inputs and outputs.
                Signature: ``(fqn: str, input_tensors: tuple[Tensor, ...], output_tensors: tuple[Tensor, ...]) -> None``
            pre_bw_hook: Called before each module's backward (receives grad_outputs).
                Signature: ``(fqn: str, grad_output_tensors: tuple[Tensor | None, ...]) -> None``
                Entries may be ``None`` for tensors that did not require grad.
            post_bw_hook: Called after each module's backward (receives grad_inputs).
                Signature: ``(fqn: str, grad_input_tensors: tuple[Tensor | None, ...]) -> None``
                Entries may be ``None`` for tensors that did not require grad.
            module_filter: Optional ``(fqn, module) -> bool`` filter. Only modules
                where this returns ``True`` get hooks. Default: all modules.

        Note:
            - Works in eager mode, and under ``torch.compile`` with ``eager``
              and ``aot_eager`` backends. The ``inductor`` backend is not yet
              supported by ``leaf_function``.
            - The root module (fqn ``""``) is automatically skipped because
              ``torch.compile(model)`` wraps the model in ``OptimizedModule``,
              causing root-level hooks to fire outside the compiled region.
              Use ``model.compile()`` (in-place) to avoid this if root hooks
              are needed.
            - Backward hooks use ``torch.autograd.Function`` inside the leaf function
              to intercept gradients. This adds identity autograd nodes but does not
              alter gradient values.
            - With ``aot_export_joint_with_descriptors``, backward hooks on the
              first submodule only fire if the tracing input has
              ``requires_grad=True``, since ``aot_export`` only captures backward
              paths for tensors that required grad at trace time.
            - Tensor subclasses (e.g., ``DTensor``) are not supported.
              ``invoke_leaf_function`` has no dispatch rule for subclasses, so hooks
              will fail even in eager mode. Use ``register_user_hooks`` instead for
              subclass tensors.

        Example::

            with ModTracker.compile_safe_hooks(
                model,
                post_fw_hook=lambda fqn, inp, out: print(
                    f"[FW] {fqn}: {[x.shape for x in inp]} -> {[x.shape for x in out]}"
                ),
                pre_bw_hook=lambda fqn, g: print(f"[BW] {fqn}: {[x.shape for x in g]}"),
            ):
                compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
                out = compiled(x)
                out.sum().backward()
        """  # noqa: B950
        handles = _register_compile_safe_hooks(
            model,
            pre_fw_hook=pre_fw_hook,
            post_fw_hook=post_fw_hook,
            pre_bw_hook=pre_bw_hook,
            post_bw_hook=post_bw_hook,
            module_filter=module_filter,
        )
        try:
            yield handles
        finally:
            for h in handles:
                h.remove()

    def _get_mod_name(self, mod):
        if mod not in self._known_modules:
            self._known_modules[mod] = type(mod).__name__
        mod_name = self._known_modules[mod]
        if mod not in self._seen_modules:
            for name, submod in mod.named_children():
                self._known_modules[submod] = f"{mod_name}.{name}"
                self._get_mod_name(submod)
            self._seen_modules.add(mod)
        return mod_name

    def _get_append_fn(self, w_mod, name, is_bw):
        def fn(*args):
            if is_bw:
                self._maybe_set_engine_callback()
            if name in self.parents and not self.is_bw:

                def custom_formatwarning(msg, category, filename, lineno, line=None):
                    return f"{filename}:{lineno}: {category.__name__}: {msg} \n"

                # pyrefly: ignore [bad-assignment]
                warnings.formatwarning = custom_formatwarning
                warnings.warn(
                    "The module hierarchy tracking maybe be messed up."
                    " Please file a bug to PyTorch, if it is the case.",
                    stacklevel=2,
                )
            if name not in self.parents:
                self._active_module_cnt[name] = 1
                self.parents.add(name)
            else:
                self._active_module_cnt[name] += 1

            if self._user_pre_bw_hook is not None and is_bw:
                self._user_pre_bw_hook(w_mod(), args)

        return fn

    def _get_pop_fn(self, w_mod, name, is_bw):
        def fn(*args):
            if self._user_post_bw_hook is not None and is_bw:
                self._user_post_bw_hook(w_mod(), args)
            if name in self.parents:
                self._active_module_cnt[name] -= 1
                if self._active_module_cnt[name] == 0:
                    self.parents.remove(name)
            elif not self.is_bw:
                # Due to some input/output not requiring gradients, we cannot enforce
                # proper nesting in backward
                raise RuntimeError(
                    "The Module hierarchy tracking is wrong. Report a bug to PyTorch"
                )

        return fn

    def _fw_pre_hook(self, mod, input):
        if torch._dynamo.eval_frame._is_in_optimized_module():
            return

        name = self._get_mod_name(mod)
        w_mod = weakref.ref(mod)
        self._get_append_fn(w_mod, name, False)()
        if self._user_pre_fw_hook is not None:
            self._user_pre_fw_hook(mod, input)
        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw:
            if tensors:
                register_multi_grad_hook(tensors, self._get_pop_fn(w_mod, name, True))
            else:
                self._post_bw_callbacks_to_enqueue.append(
                    self._get_pop_fn(w_mod, name, True)
                )

    def _fw_post_hook(self, mod, input, output):
        if torch._dynamo.eval_frame._is_in_optimized_module():
            return

        name = self._get_mod_name(mod)
        w_mod = weakref.ref(mod)
        if self._user_post_fw_hook is not None:
            self._user_post_fw_hook(mod, input, output)
        self._get_pop_fn(w_mod, name, False)()
        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw and tensors:
            register_multi_grad_hook(
                tensors, self._get_append_fn(w_mod, name, True), mode="any"
            )

    def __enter__(self):
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(
            self._fw_post_hook, always_call=True
        )
        return self

    def __exit__(self, *args):
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
