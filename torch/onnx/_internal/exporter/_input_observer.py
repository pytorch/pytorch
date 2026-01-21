from __future__ import annotations


__all__ = ["InputObserver"]

import contextlib
import inspect
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    use_dict: bool = True,
    change_function: Callable[[torch.Tensor], Any] | None = None,
) -> Any:
    """Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    Args:
        obj:
            object from a custom class
        use_dict:
            closer to the original result but
            :func:`torch.export.export` only considers the values,
            the context gives the dictionary keys but it is not expressed
            in the dynamic shapes, these specifications seems to be different
            for the strict and non strict mode. It also preserves tuple.
        change_function:
            to modify the tensor in the structure itself,
            like replace them by a shape

    Returns:
        the serialized object
    """
    if isinstance(obj, torch.Tensor):
        return change_function(obj) if change_function else obj
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    start = 0
    end = 0
    subtrees = []
    for subspec in (
        spec.children() if hasattr(spec, "children") else spec.children_specs
    ):
        end += subspec.num_leaves
        value = subspec.unflatten(flat[start:end])
        value = flatten_unflatten_for_dynamic_shapes(
            value, use_dict=use_dict, change_function=change_function
        )
        subtrees.append(value)
        start = end
    if use_dict:
        if spec.type is dict:
            # This is a dictionary.
            return dict(zip(spec.context, subtrees))
        if spec.type is tuple:
            return tuple(subtrees)
        if spec.type is list:
            return list(subtrees)
        if spec.type is None and not subtrees:
            return None
        if spec.context:
            # This is a custom class with attributes.
            # It is returned as a list.
            return list(subtrees)
        raise ValueError(
            f"Unable to interpret spec type {spec.type} "
            f"(type is {type(spec.type)}, context is {spec.context}), "
            f"spec={spec}, subtrees={subtrees}"
        )
    # This is a list.
    return subtrees


def infer_dynamic_dimensions(shape_list: Sequence[tuple[int, ...]]) -> list[int]:
    """Returns the list of dynamic dimensions given a list of shapes
    corresponding to the same tensor.

    Args:
        shape_list:
            list of shapes, they must all have the same length

    Returns:
        list of dynamic dimensions
    """
    unique_ranks = {len(shape) for shape in shape_list}
    torch._check(
        len(unique_ranks) == 1,
        lambda: "all shapes in shape_list must have the same rank",
    )
    rank = unique_ranks.pop()
    dynamic = []
    for i in range(rank):
        dims = [shape[i] for shape in shape_list]
        if len(set(dims)) > 1:
            dynamic.append(i)
    return dynamic


class InputObserverInfo:
    def __init__(self, signature: inspect.Signature):
        # pyrefly: ignore
        self.inputs_specs: list[torch.utils._pytree.PyTreeSpec] = []
        self.flat_inputs: list[list[torch.Tensor | None]] = []
        # pyrefly: ignore
        self.outputs_specs: list[torch.utils._pytree.PyTreeSpec] = []
        self.flat_outputs: list[list[torch.Tensor]] = []
        self.signature = signature

        self._max_args: tuple[Any, torch.Tensor] | None = None
        self._max_kwargs: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.flat_inputs)

    def add_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None and not isinstance(v, (int, float, bool))
        }
        flat_args, spec = torch.utils._pytree.tree_flatten((args, kwargs))
        self.inputs_specs.append(spec)
        cloned = [(None if t is None else t.clone().detach()) for t in flat_args]
        self.flat_inputs.append(cloned)

        cloned_args, cloned_kwargs = torch.utils._pytree.tree_unflatten(cloned, spec)
        if self._max_args is None or len(cloned_args) > len(self._max_args):
            self._max_args = cloned_args
        if self._max_kwargs is None or len(cloned_kwargs) > len(self._max_kwargs):
            self._max_kwargs = cloned_kwargs

    def add_outputs(self, res: torch.Tensor | tuple[torch.Tensor, ...]):
        flat_res, spec = torch.utils._pytree.tree_flatten(res)
        self.outputs_specs.append(spec)
        self.flat_outputs.append([t.clone().detach() for t in flat_res])

    def build_inputs_completed_with_none_values(self) -> list[list[torch.Tensor]]:
        # Let's compute the sizes of each independently.
        if not self.flat_inputs or self._max_args is None or self._max_kwargs is None:
            raise RuntimeError("No inputs were captured.")
        arg_sizes = [
            len(torch.utils._pytree.tree_flatten(a)[0]) for a in self._max_args
        ]
        kwarg_sizes = {
            k: len(torch.utils._pytree.tree_flatten(v)[0])
            for k, v in self._max_kwargs.items()
        }

        # Let's reprocess everything.
        captured_inputs: dict[int | str, int] = {}
        new_flat_inputs = []
        for args_kwargs, spec in zip(self.flat_inputs, self.inputs_specs):
            args, kwargs = torch.utils._pytree.tree_unflatten(args_kwargs, spec)
            if len(set(kwargs) | set(self._max_kwargs)) > len(self._max_kwargs):
                raise RuntimeError(
                    "At least one call to the observed model "
                    "must contain all the named arguments."
                )
            flat = []
            for i in range(len(self._max_args)):
                if i < len(args):
                    ts = torch.utils._pytree.tree_flatten(args[i])[0]
                    if i in captured_inputs and captured_inputs[i] != len(ts):
                        raise RuntimeError(
                            f"Positional argument {i} has {len(ts)} tensors "
                            f"but previously got {captured_inputs[i]} tensors. "
                            f"Inference is impossible in that case."
                        )
                    captured_inputs[i] = len(ts)
                    flat.extend(ts)
                else:
                    flat.extend([None for _ in range(arg_sizes[i])])
            for k in self._max_kwargs:
                if k in kwargs:
                    ts = torch.utils._pytree.tree_flatten(kwargs[k])[0]
                    if k in captured_inputs and captured_inputs[k] != len(ts):
                        raise RuntimeError(
                            f"Named argument {k!r} has {len(ts)} tensors "
                            f"but previously got {captured_inputs[k]} tensors. "
                            f"Inference is impossible in that case."
                        )
                    captured_inputs[k] = len(ts)
                    flat.extend(ts)
                else:
                    flat.extend([None for _ in range(kwarg_sizes[k])])
            new_flat_inputs.append(flat)
        return new_flat_inputs

    def infer_dynamic_shapes(
        self,
    ) -> tuple[dict[int, Any], ...] | dict[str, dict[int, Any]]:
        flat_inputs = self.build_inputs_completed_with_none_values()
        # This is already checked by build_inputs_completed_with_none_values
        # but this is not always well captured by tools checking types.
        assert self._max_args is not None and self._max_kwargs is not None
        if len({len(flat) for flat in flat_inputs}) != 1:
            raise NotImplementedError(
                "infer_dynamic_shapes is not implemented "
                "when the number of input tensors are not the same."
            )
        shape_lists = [
            [(None if t is None else t.shape) for t in tensors]
            for tensors in flat_inputs
        ]
        n_tensors = len(shape_lists[0])
        dynamic_shapes = [
            infer_dynamic_dimensions(
                [s for s in [shapes[index] for shapes in shape_lists] if s is not None]
            )
            for index in range(n_tensors)
        ]
        cst = torch.export.Dim.DYNAMIC
        flat_dynamic_shapes = [dict.fromkeys(dims, cst) for dims in dynamic_shapes]
        if len(flat_dynamic_shapes) == len(self._max_args) + len(self._max_kwargs):
            # It means forward method is called with tensors only.
            if not self._max_kwargs:
                # only positional arguments
                return tuple(flat_dynamic_shapes)
            if not self._max_args:
                # only named arguments
                return dict(zip(list(self._max_kwargs), flat_dynamic_shapes))
            # positional arguments needs to be moved to the named arguments
            n_args = len(self._max_args)
            pos_names = list(self.signature.parameters)[:n_args]
            return {
                **dict(zip(pos_names, flat_dynamic_shapes[:n_args])),
                **dict(zip(list(self._max_kwargs), flat_dynamic_shapes[n_args:])),
            }

        # nested types, here comes the fun part because the shapes cannot be unflattened,
        # custom classes must appear in their flattened shape.
        # This does not work in all cases but every time every available argument is flattened
        # with the same number of tensors. The function does not check
        # if that assumption is true.
        flat_inputs, _max_spec = torch.utils._pytree.tree_flatten(
            (self._max_args, self._max_kwargs)
        )
        torch._check(
            len(flat_inputs) == len(flat_dynamic_shapes),
            (
                f"Length mismatch len(flat_inputs)={len(flat_inputs)}, "
                f"len(flat_dynamic_shapes)={len(flat_dynamic_shapes)}"
            ),
        )
        mapping = {id(t): shape for t, shape in zip(flat_inputs, flat_dynamic_shapes)}
        ds_args, ds_kwargs = flatten_unflatten_for_dynamic_shapes(
            (self._max_args, self._max_kwargs), change_function=lambda t: mapping[id(t)]
        )
        if not ds_kwargs:
            return tuple(ds_args)
        if not ds_args:
            return tuple(ds_kwargs)
        pos_names = list(self.signature.parameters)[: len(ds_args)]
        return {**dict(zip(pos_names, ds_args)), **ds_kwargs}

    def infer_arguments(
        self, index: int | None = None
    ) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        # This is already checked by build_inputs_completed_with_none_values
        # but this is not always well captured by tools checking types.
        assert self._max_args is not None and self._max_kwargs is not None
        candidate = None
        if index is None:
            for i, (args_kwargs, spec) in enumerate(
                zip(self.flat_inputs, self.inputs_specs)
            ):
                args, kwargs = torch.utils._pytree.tree_unflatten(args_kwargs, spec)
                if len(args) == len(self._max_args) and len(kwargs) == len(
                    self._max_kwargs
                ):
                    index = i
                    candidate = args, kwargs
                    break
        if index is not None:
            # found one available set.
            args, kwargs = candidate or torch.utils._pytree.tree_unflatten(
                self.flat_inputs[index], self.inputs_specs[index]
            )
            if not kwargs:
                return args
            if not args:
                return kwargs
            # We need to move args to kwargs
            pos_names = list(self.signature.parameters)[: len(args)]
            return {**dict(zip(pos_names, args)), **kwargs}

        raise NotImplementedError(
            "We could not find a good set of inputs/outputs. "
            "We need to replace none by empty tensors."
        )


class InputObserver:
    """Steals forward method to collect inputs and outputs.
    This information is used to infer dynamic shapes and
    export arguments.
    """

    def __init__(self, store_n_calls: int = 3):
        self.store_n_calls = store_n_calls
        self.info: InputObserverInfo | None = None

    def _forward_captured(self, *args, _captured_forward=None, **kwargs):
        assert _captured_forward is not None, "_captured_forward cannot be None"
        assert self.info is not None, "info cannot be None"
        n_stored = len(self.info)
        if n_stored < self.store_n_calls:
            self.info.add_inputs(args, kwargs)
        res = _captured_forward(*args, **kwargs)
        if n_stored < self.store_n_calls:
            self.info.add_outputs(res)
        return res

    @contextlib.contextmanager
    def __call__(self, model: torch.nn.Module):
        if self.info is not None:
            raise RuntimeError(
                "This class was already used to capture a model. Please create a new one."
            )
        self.info = InputObserverInfo(signature=inspect.signature(model.forward))
        forward_method = model.forward
        model.forward = (
            lambda *args,
            _captured_forward=forward_method,
            **kwargs: self._forward_captured(
                *args, _captured_forward=_captured_forward, **kwargs
            )
        )
        try:
            yield self
        finally:
            model.forward = forward_method

    def _check_captured(self):
        if self.info is None:
            raise RuntimeError("No inputs were captured.")

    def infer_dynamic_shapes(
        self,
    ) -> tuple[dict[int, Any], ...] | dict[str, dict[int, Any]]:
        self._check_captured()
        assert self.info is not None  # missed by type checking
        return self.info.infer_dynamic_shapes()

    def infer_arguments(
        self, index: int | None = None
    ) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        self._check_captured()
        assert self.info is not None  # missed by type checking
        return self.info.infer_arguments(index=index)
