from __future__ import annotations


__all__ = ["InputObserver"]

import contextlib
import inspect
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    change_function: Callable[[torch.Tensor], Any] | None = None,
) -> Any:
    """Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    Args:
        obj: Object from a custom class.
        change_function: Function to modify the tensor in the structure itself,
            like replace them by a shape.

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
        value = _flatten_unflatten_for_dynamic_shapes(
            value, change_function=change_function
        )
        subtrees.append(value)
        start = end
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


def _infer_dynamic_dimensions(
    shape_list: Sequence[tuple[int, ...]], set_batch_dimension: bool = False
) -> list[int]:
    """Returns the list of dynamic dimensions given a list of shapes
    corresponding to the same tensor.

    Args:
        shape_list:
            list of shapes, they must all have the same length
        set_batch_dimension:
            forces the first dimension to be treated as dynamic, even if all shapes have the same value for that dimension

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
        if len(set(dims)) > 1 or (i == 0 and set_batch_dimension):
            dynamic.append(i)
    return dynamic


class InputObserverInfo:
    """Contains all the necessary information to infer dynamic shapes
    and the arguments to send to :func:`torch.export.export`.

    Args:
        signature_names: Names of the arguments of the method
            the collector tensors come from. They are used if it becomes
            necessary to move positional arguments to named ones.
    """

    def __init__(self, signature_names: list[str]):
        self.inputs_specs: list[torch.utils._pytree.PyTreeSpec] = []
        self.flat_inputs: list[list[torch.Tensor | None]] = []
        self.outputs_specs: list[torch.utils._pytree.PyTreeSpec] = []
        self.flat_outputs: list[list[torch.Tensor]] = []
        self.signature_names = signature_names

        self._max_args: tuple[Any, torch.Tensor] | None = None
        self._max_kwargs: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        """Returns the number of collected set of inputs/outputs."""
        return len(self.flat_inputs)

    def add_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        """Stores one set of inputs. They are deepcopied.

        Args:
            args: Positional arguments.
            kwargs: Named arguments.
        """
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None and not isinstance(v, (int, float, bool))
        }
        flat_args, spec = torch.utils._pytree.tree_flatten((args, kwargs))
        self.inputs_specs.append(spec)
        cloned = [
            (None if not isinstance(t, torch.Tensor) else t.clone().detach())
            for t in flat_args
        ]
        self.flat_inputs.append(cloned)

        cloned_args, cloned_kwargs = torch.utils._pytree.tree_unflatten(cloned, spec)
        if self._max_args is None or len(cloned_args) > len(self._max_args):
            self._max_args = cloned_args
        if self._max_kwargs is None or len(cloned_kwargs) > len(self._max_kwargs):
            self._max_kwargs = cloned_kwargs

    def add_outputs(self, res: torch.Tensor | tuple[torch.Tensor, ...]):
        """Stores outputs. They are deepcopied."""
        flat_res, spec = torch.utils._pytree.tree_flatten(res)
        self.outputs_specs.append(spec)
        self.flat_outputs.append([t.clone().detach() for t in flat_res])

    def _build_inputs_completed_with_none_values(
        self,
    ) -> tuple[list[int | str], list[list[torch.Tensor]]]:
        # Let's compute the sizes of each independently.
        if not self.flat_inputs or self._max_args is None or self._max_kwargs is None:
            raise RuntimeError("No inputs were captured.")

        flat_index_to_args: list[int | str] = []
        arg_sizes = []
        for index_args, a in enumerate(self._max_args):
            size = len(torch.utils._pytree.tree_flatten(a)[0])
            arg_sizes.append(size)
            flat_index_to_args.extend([index_args] * size)
        kwarg_sizes = {}
        for k, v in self._max_kwargs.items():
            size = len(torch.utils._pytree.tree_flatten(v)[0])
            kwarg_sizes[k] = size
            flat_index_to_args.extend([k] * size)

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
        return flat_index_to_args, new_flat_inputs

    def infer_dynamic_shapes(
        self, set_batch_dimension_for: set[int | str] | None = None
    ) -> tuple[dict[int, Any], ...] | dict[str, dict[int, Any]]:
        """Infers dynamic shapes based on the collected tensors.
        Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).

        Args:
            set_batch_dimension_for (set[int | str] | None): Set of input identifiers,
                by name (``str``) or position (``int``), for which the first dimension
                should be treated as a dynamic batch dimension. If ``None`` or empty,
                no additional batch dimensions are marked as dynamic.
        """

        def _set_batch_dimension(name_or_position):
            if not set_batch_dimension_for:
                return False
            if name_or_position in set_batch_dimension_for:
                return True
            if isinstance(name_or_position, int):
                torch._check(
                    name_or_position < len(self.signature_names),
                    lambda: f"argument at position {name_or_position} is out of boundary",
                )
                if self.signature_names[name_or_position] in set_batch_dimension_for:
                    return True
            return False

        flat_index_to_args, flat_inputs = (
            self._build_inputs_completed_with_none_values()
        )

        def _set_batch_dimension_for_flat_index(index):
            return _set_batch_dimension(flat_index_to_args[index])

        # This is already checked by build_inputs_completed_with_none_values
        # but this is not always well captured by tools checking types.
        if self._max_args is None or self._max_kwargs is None:
            raise AssertionError("_max_args and _max_kwargs must be non-None")
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
            _infer_dynamic_dimensions(
                [s for s in [shapes[index] for shapes in shape_lists] if s is not None],
                set_batch_dimension=_set_batch_dimension_for_flat_index(index),
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
            pos_names = self.signature_names[:n_args]
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
        ds_args, ds_kwargs = _flatten_unflatten_for_dynamic_shapes(
            (self._max_args, self._max_kwargs), change_function=lambda t: mapping[id(t)]
        )
        if not ds_kwargs:
            return tuple(ds_args)
        if not ds_args:
            return tuple(ds_kwargs)
        pos_names = self.signature_names[: len(ds_args)]
        return {**dict(zip(pos_names, ds_args)), **ds_kwargs}

    def infer_arguments(
        self, index: int | None = None
    ) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Infers arguments based on the collected tensors."""
        # This is already checked by _build_inputs_completed_with_none_values
        # but this is not always well captured by tools checking types.
        if self._max_args is None or self._max_kwargs is None:
            raise AssertionError("_max_args and _max_kwargs must be non-None")
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
            pos_names = self.signature_names[: len(args)]
            return {**dict(zip(pos_names, args)), **kwargs}

        raise NotImplementedError(
            "We could not find a good set of inputs/outputs. "
            "We need to replace none by empty tensors. "
            "This will be soon implemented."
        )


class InputObserver:
    """Steals forward method to collect inputs and outputs.
    This information is used to infer dynamic shapes and
    export arguments.

    Examples
    --------
    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model(x1, y1)
    >>>     model(x2, y2)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    With LLM:
    >>> input_observer = InputObserver()
    >>> with input_observer(model):
    >>>     model.generate(input_ids)
    >>> ep = torch.export.export(  # or torch.onnx.export
    >>>     model,
    >>>     ()
    >>>     kwargs=input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    .. versionadded:: 2.11.0
    """

    def __init__(self):
        self.info: InputObserverInfo | None = None

    def _replaced_method(
        self,
        *args,
        _captured_method: Callable | None = None,
        _store_n_calls: int = 3,
        **kwargs,
    ):
        if _captured_method is None:
            raise AssertionError("_captured_forward cannot be None")
        if self.info is None:
            raise AssertionError("info cannot be None")
        n_stored = len(self.info)
        if n_stored < _store_n_calls:
            self.info.add_inputs(args, kwargs)
        res = _captured_method(*args, **kwargs)
        if n_stored < _store_n_calls:
            self.info.add_outputs(res)
        return res

    @contextlib.contextmanager
    def __call__(
        self,
        model: torch.nn.Module,
        store_n_calls: int = 3,
        method_name: str = "forward",
    ):
        """Starts collecting inputs and outputs of a specific method.
        The model method is replaced by a new one collecting tensors
        before and after the inner one is called.
        The original method is restored after the collection.

        Args:
            model: Model
            store_n_calls: The collection stops after this many calls
                to avoid taking too much memory.
            method_name: Method name to spy on.
        """
        if self.info is not None:
            raise RuntimeError(
                "This class was already used to capture a model. Please create a new one."
            )
        if not hasattr(model, method_name):
            raise ValueError(
                f"Model type {model} does not have a method {method_name!r}"
            )
        captured_method = getattr(model, method_name)
        self.info = InputObserverInfo(
            signature_names=list(inspect.signature(captured_method).parameters)
        )
        setattr(
            model,
            method_name,
            lambda *args,
            _cm=captured_method,
            _snc=store_n_calls,
            **kwargs: self._replaced_method(
                *args,
                _captured_method=_cm,
                _store_n_calls=_snc,
                **kwargs,
            ),
        )
        try:
            yield self
        finally:
            setattr(model, method_name, captured_method)

    def _check_captured(self):
        if self.info is None:
            raise RuntimeError("No inputs were captured.")

    def infer_dynamic_shapes(
        self, set_batch_dimension_for: set[int | str] | None = None
    ) -> tuple[dict[int, Any], ...] | dict[str, dict[int, Any]]:
        """
        Infers dynamic shapes. Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).

        Args:
            set_batch_dimension_for (set[int | str] | None): A set of input
                identifiers (by position as ``int`` or by name as ``str``) for
                which the first dimension should be treated as a dynamic batch
                dimension. If ``None``, no dimensions are explicitly marked as
                dynamic.
        """
        self._check_captured()
        if self.info is None:
            raise AssertionError("info must be non-None")
        return self.info.infer_dynamic_shapes(
            set_batch_dimension_for=set_batch_dimension_for
        )

    def infer_arguments(
        self, index: int | None = None
    ) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor]:
        """Infers arguments based on the collected tensors."""
        self._check_captured()
        if self.info is None:
            raise AssertionError("info must be non-None")
        return self.info.infer_arguments(index=index)
