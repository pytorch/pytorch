# mypy: allow-untyped-defs
import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any

__all__ = ["RemovableHandle", "unserializable_hook", "warn_if_has_hooks", "BackwardHook"]

class RemovableHandle:
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (Union[dict, List[dict]]): An additional dictionary or list of
            dictionaries whose keys will be deleted when the same keys are
            removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref: tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (self.hooks_dict_ref(), self.id, tuple(ref() for ref in self.extra_dict_ref))

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2])

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


def unserializable_hook(f):
    """
    Mark a function as an unserializable hook with this decorator.

    This suppresses warnings that would otherwise arise if you attempt
    to serialize a tensor that has a hook.
    """
    f.__torch_unserializable__ = True
    return f


def warn_if_has_hooks(tensor) -> None:
    if tensor._backward_hooks:
        for k in tensor._backward_hooks:
            hook = tensor._backward_hooks[k]
            if not hasattr(hook, "__torch_unserializable__"):
                warnings.warn(f"backward hook {repr(hook)} on tensor will not be "
                              "serialized.  If this is expected, you can "
                              "decorate the function with @torch.utils.hooks.unserializable_hook "
                              "to suppress this warning", stacklevel=2)

class BackwardHook:
    """
    A wrapper class to implement nn.Module backward hooks.

    It handles:
      - Ignoring non-Tensor inputs and replacing them by None before calling the user hook
      - Generating the proper Node to capture a set of Tensor's gradients
      - Linking the gradients captures for the outputs with the gradients captured for the input
      - Calling the user hook once both output and input gradients are available
    """

    def __init__(self, module, user_hooks, user_pre_hooks) -> None:
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks
        self.module = module

        self.grad_outputs = None
        self.n_outputs = -1
        self.output_tensors_index = None
        self.n_inputs = -1
        self.input_tensors_index = None

    def _pack_with_none(self, indices, values, size):
        res = [None] * size
        for idx, val in zip(indices, values, strict=True):
            res[idx] = val

        return tuple(res)

    def _unpack_none(self, indices, values):
        res = [values[idx] for idx in indices]

        return tuple(res)

    def _set_user_hook(self, grad_fn) -> None:
        def hook(grad_input, _):
            if self.grad_outputs is None:
                # This happens because the gradient in your nn.Module flows to
                # the Module's input without " passing through the Module's
                # output, e.g. when you're doing double backward.
                return
            res = self._pack_with_none(self.input_tensors_index, grad_input, self.n_inputs)

            for hook in self.user_hooks:
                out = hook(self.module, res, self.grad_outputs)

                if out is None:
                    continue

                if len(out) != len(res):
                    raise RuntimeError("Backward hook returned an invalid number of grad_input, "
                                       f"got {len(out)}, but expected {len(res)}")

                res = out

            # pyrefly: ignore [bad-assignment]
            self.grad_outputs = None

            return self._unpack_none(self.input_tensors_index, res)

        grad_fn.register_hook(hook)

    def _apply_on_tensors(self, fn, args):
        # Can be used to apply the given function to the tensors contained in the
        # args. Will return updated args and the tensors indices
        tensors_idx = []
        tensors = []

        requires_grad = False
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensors_idx.append(i)
                tensors.append(arg)
                requires_grad |= arg.requires_grad

        if not (requires_grad and torch.is_grad_enabled()):
            return args, None

        new_tensors = torch.nn.modules._functions.BackwardHookFunction.apply(*tensors)
        if len(new_tensors) == 0:
            raise RuntimeError("Cannot set Module backward hook for a Module with no input Tensors.")

        grad_fns = [t.grad_fn for t in new_tensors if t.grad_fn is not None and t.grad_fn.name() == "BackwardHookFunctionBackward"]
        if len(grad_fns) == 0:
            raise RuntimeError("Error while setting up backward hooks. Please open "
                               "an issue with a code sample to reproduce this.")

        fn(grad_fns[0])

        arg_list = list(args)
        for idx, val in zip(tensors_idx, new_tensors, strict=True):
            arg_list[idx] = val

        if type(args) is tuple:
            out = tuple(arg_list)
        else:
            out = type(args)(*arg_list)
        return out, tensors_idx

    def setup_input_hook(self, args):
        def fn(grad_fn) -> None:
            self._set_user_hook(grad_fn)

        res, input_idx = self._apply_on_tensors(fn, args)
        self.n_inputs = len(args)
        self.input_tensors_index = input_idx
        return res

    def setup_output_hook(self, args):
        def fn(grad_fn) -> None:
            def hook(_, grad_output):
                self.grad_outputs = self._pack_with_none(self.output_tensors_index,
                                                         grad_output,
                                                         self.n_outputs)

                if self.user_pre_hooks:
                    expected_len = len(self.grad_outputs)
                    for user_pre_hook in self.user_pre_hooks:
                        hook_grad_outputs = user_pre_hook(self.module, self.grad_outputs)
                        if hook_grad_outputs is None:
                            continue

                        actual_len = len(hook_grad_outputs)
                        if actual_len != expected_len:
                            raise RuntimeError("Backward pre hook returned an invalid number of grad_output, "
                                               f"got {actual_len}, but expected {expected_len}")
                        self.grad_outputs = hook_grad_outputs

                # We need to be able to clear self.grad_outputs but also return it
                local_grad_outputs = self.grad_outputs

                # Special case if no input required gradients, this hook should call the user
                # hook directly
                if self.input_tensors_index is None:
                    warnings.warn("Full backward hook is firing when gradients are computed "
                                  "with respect to module outputs since no inputs require gradients. See "
                                  "https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook "  # noqa: B950
                                  "for more details.",
                                  stacklevel=5)
                    grad_inputs = self._pack_with_none([], [], self.n_inputs)
                    for user_hook in self.user_hooks:
                        res = user_hook(self.module, grad_inputs, self.grad_outputs)
                        if res is not None and not (isinstance(res, tuple) and all(el is None for el in res)):
                            raise RuntimeError("Backward hook for Modules where no input requires "
                                               "gradient should always return None or None for all gradients.")
                    self.grad_outputs = None

                if local_grad_outputs is not None:
                    if self.output_tensors_index is None:
                        raise AssertionError("output_tensors_index should not be None when grad_outputs is not None")
                    return tuple(local_grad_outputs[i] for i in self.output_tensors_index)

            grad_fn.register_hook(hook)

        is_tuple = True
        if not isinstance(args, tuple):
            args = (args,)
            is_tuple = False

        res, output_idx = self._apply_on_tensors(fn, args)
        self.n_outputs = len(args)
        self.output_tensors_index = output_idx

        if not is_tuple:
            res = res[0]
        return res
