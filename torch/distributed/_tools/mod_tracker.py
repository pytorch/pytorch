# mypy: allow-untyped-defs
import warnings
import weakref
from collections.abc import Callable
from typing import Optional

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten


__all__ = ["ModTracker"]


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
        pre_fw_hook: Optional[Callable] = None,
        post_fw_hook: Optional[Callable] = None,
        pre_bw_hook: Optional[Callable] = None,
        post_bw_hook: Optional[Callable] = None,
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

                warnings.formatwarning = custom_formatwarning
                warnings.warn(
                    "The module hierarchy tracking maybe be messed up."
                    " Please file a bug to PyTorch, if it is the case."
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
