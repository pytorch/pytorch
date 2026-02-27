from __future__ import annotations


__all__ = ["InputObserver"]

import contextlib
import inspect
import time
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx._internal.exporter import _onnx_program


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import onnxruntime as ort


def _flatten_unflatten_for_dynamic_shapes(
    obj: Any,
    change_function: Callable[[torch.Tensor], Any] | None = None,
) -> Any:
    """Returns the object in a different structure similar to what
    the definition of the dynamic shapes should use.

    Args:
        obj: Object from a custom class.
        change_function: If not None, this function is called to modify the tensors
            in the structure itself, like replace them by a shape.

    Returns:
        The flattened object.
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
            List of shapes, they must all have the same length.
        set_batch_dimension:
            Forces the first dimension to be treated as dynamic,
            even if all shapes have the same value for that dimension.

    Returns:
        list of dynamic dimensions
    """
    unique_ranks = {len(shape) for shape in shape_list}
    torch._check(
        len(unique_ranks) == 1,
        lambda: f"All shapes in shape_list must have the same rank but {shape_list=}.",
    )
    rank = unique_ranks.pop()
    dynamic = []
    for i in range(rank):
        dims = [shape[i] for shape in shape_list]
        if len(set(dims)) > 1 or (i == 0 and set_batch_dimension):
            dynamic.append(i)
    return dynamic


class InputCandidate:
    """Retains one set of inputs given to the forward method or any
    other method the class :class:`InputObserver` is stealing from.
    Any class is allowed as long as it can be flattened.

    Args:
        args: Positional arguments.
        kwargs: Optional arguments.
        clone: Clone the inputs before storing them. Some tensors
            may be modified inplace, the original value must be retained.
        cst_kwargs: Any optional arguments constant over multiple calls.
            int, float, str, bool values must be stored here.

    The constructor flattens the received arguments.
    Any necessary flattening function should have been registered first.
    """

    def __init__(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        clone: bool,
        cst_kwargs: dict[str, int | str | float | bool],
    ):
        self.args = args
        self.kwargs = kwargs
        self.flat_list, self.spec = torch.utils._pytree.tree_flatten((args, kwargs))
        self._position_to_args_kwargs: list[int | str] | None = None
        self._n_tensors_for_args_kwargs: dict[int | str, int] | None = None
        self.cst_kwargs = cst_kwargs.copy()

        if clone:
            self.flat_list = [
                (None if not isinstance(t, torch.Tensor) else t.clone().detach())
                for t in self.flat_list
            ]
            self.args, self.kwargs = torch.utils._pytree.tree_unflatten(
                self.flat_list, self.spec
            )

        self.aligned_spec: torch.utils._pytree.PyTreeSpec | None = None
        self.aligned_flat_list: list[torch.Tensor | None] | None = None

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self.args)} args, "
            f"{len(self.kwargs)} kwargs, {len(self.flat_list)} tensors, "
            f"{len(self.aligned_flat_list or [])} aligned tensors)"
        )

    def __len__(self) -> int:
        """Returns the number of flattened tensors, None tensors are included."""
        return len(self.flat_list)

    def str_obs(self) -> str:
        """Prints out some information about the observations."""
        return (
            f"InputCandidate(args=#{len(self.args)}(...), "
            f"kwargs=#{len(self.kwargs)}{{...}}, cst_kwargs={self.cst_kwargs})"
        )

    def build_mappings(self) -> list[int | str]:
        if self._position_to_args_kwargs is not None:
            return self._position_to_args_kwargs
        self._n_tensors_for_args_kwargs = {}

        flat_index_to_args: list[int | str] = []
        for index_args, a in enumerate(self.args):
            size = len(torch.utils._pytree.tree_flatten(a)[0])
            self._n_tensors_for_args_kwargs[index_args] = size
            flat_index_to_args.extend([index_args] * size)
        for k, v in self.kwargs.items():
            size = len(torch.utils._pytree.tree_flatten(v)[0])
            self._n_tensors_for_args_kwargs[k] = size
            flat_index_to_args.extend([k] * size)

        self._position_to_args_kwargs = flat_index_to_args
        return self._position_to_args_kwargs

    @property
    def position_to_args_kwargs(self) -> list[int | str]:
        """Returns the corresponding args or kwargs
        for every tensor in the flattened inputs.
        """
        if self._position_to_args_kwargs is None:
            self.build_mappings()
        # pyrefly: ignore [bad-return]
        return self._position_to_args_kwargs

    @property
    def n_tensors_for_args_kwargs(self) -> dict[int | str, int]:
        """Returns the number of flat tensors in every args or kwargs."""
        if self._n_tensors_for_args_kwargs is None:
            self.build_mappings()
        # pyrefly: ignore [bad-return]
        return self._n_tensors_for_args_kwargs

    def _set_aligned_flat_list(
        self,
        aligned_flat_list: list[torch.Tensor | None],
        aligned_spec: torch.utils._pytree.PyTreeSpec,
    ):
        self.aligned_flat_list = aligned_flat_list
        self.aligned_spec = aligned_spec

    def align_with(
        self,
        best_candidate: InputCandidate,
        captured_inputs: dict[int | str, int],
        signature_names: list[str],
    ):
        """Two candidates are considered as aligned if after being flattened
        if they have the same number of tensors (None allowed)."""
        if self.cst_kwargs != best_candidate.cst_kwargs:
            raise RuntimeError(
                f"Two calls were made with different constant values, "
                f"{self.cst_kwargs} != {best_candidate.cst_kwargs}"
            )

        args = self.args
        if len(self.args) > len(best_candidate.args):
            # We need to move some args to kwargs as the best_candidate does.
            new_kwargs = {}
            for i in range(len(best_candidate.args), len(self.args)):
                new_kwargs[signature_names[i]] = args[i]
            args = args[: len(best_candidate.args)]
            kwargs = {**new_kwargs, **self.kwargs}
        else:
            kwargs = self.kwargs

        flat = []
        for i in range(len(best_candidate.args)):
            if i < len(args) and (isinstance(args[i], torch.Tensor) or args[i]):
                ts = torch.utils._pytree.tree_flatten(self.args[i])[0]
                if i in captured_inputs and captured_inputs[i] != len(ts):
                    raise RuntimeError(
                        f"Positional argument {i} has {len(ts)} tensors "
                        f"but previously got {captured_inputs[i]} tensors. "
                        f"Inference is impossible in that case."
                    )
                captured_inputs[i] = len(ts)
                flat.extend(ts)
                continue
            # If the argument i is not specified or is None or an empty container.
            flat.extend(
                [None for _ in range(best_candidate.n_tensors_for_args_kwargs[i])]
            )

        for k in best_candidate.kwargs:
            if k in kwargs and (isinstance(kwargs[k], torch.Tensor) or kwargs[k]):
                ts = torch.utils._pytree.tree_flatten(kwargs[k])[0]
                if k in captured_inputs and captured_inputs[k] != len(ts):
                    raise RuntimeError(
                        f"Named argument {k!r} has {len(ts)} tensors "
                        f"but previously got {captured_inputs[k]} tensors in "
                        f"kwargs={list(kwargs)}. "
                        f"Inference is impossible in that case."
                    )
                captured_inputs[k] = len(ts)
                flat.extend(ts)
                continue
            # If the argument k is not specified or is None or an empty container.
            flat.extend(
                [None for _ in range(best_candidate.n_tensors_for_args_kwargs[k])]
            )

        self._set_aligned_flat_list(flat, best_candidate.spec)

    @property
    def n_aligned_tensors(self) -> int:
        if self.aligned_flat_list is None:
            raise RuntimeError("This input was not aligned with the others.")
        return len(self.aligned_flat_list)


class InputObserverInfo:
    """Contains all the necessary information to infer dynamic shapes
    and the arguments to send to :func:`torch.export.export`.

    Args:
        signature_names: Names of the arguments of the method
            the collector tensors come from. They are used if it becomes
            necessary to move positional arguments to named ones.
            They are used a second time because :func:`torch.export.export`
            cares about the order in kwargs and dynamic shapes, it needs
            to be the same in the ordered dictionaries `add_inputs` receive.
        default_values: Default values defined by the signature of the function,
            any value equal to that is ignored to simplify the export.
        value_if_missing: If an argument is missing,
            a default value will be taken in this dictionary,
            this is used when after the prefill step, an argument
            disappears (such as `pixel_values`) and another one
            is added (such as `past_key_values`).
            The values are only to infer dynamic shapes and arguments,
            not to run the model.
        args_name_and_position: Name of parameter `*args`
            and its position if it exists.
        kwargs_name: Name of the variable keyword parameter `**kwargs` if it exists.

    This is used by class :class:`InputObserver`.
    """

    def __init__(
        self,
        signature_names: list[str],
        default_values: dict[str, int | bool | str | float],
        value_if_missing: dict[str | int, Any],
        args_name_and_position: tuple[str, int] | None,
        kwargs_name: str | None,
    ):
        self.default_values = default_values
        self.value_if_missing = value_if_missing
        self.inputs: list[InputCandidate] = []
        self.outputs_specs: list[torch.utils._pytree.PyTreeSpec] = []
        self.flat_outputs: list[list[torch.Tensor | None]] = []
        self.latencies: list[float] = []
        self.args_name_and_position = args_name_and_position
        self.kwargs_name = kwargs_name
        self.signature_names = signature_names
        self._best_candidate: InputCandidate | None = None
        self._captured_inputs: dict[int | str, int] | None = None

    def __len__(self) -> int:
        """Returns the number of collected set of inputs/outputs."""
        return len(self.inputs)

    def add_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        """Stores one set of inputs. They are deepcopied.

        Args:
            args: Positional arguments.
            kwargs: Named arguments.
        """
        cst_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self.signature_names
            and isinstance(v, (int, float, bool, str))
            and v != self.default_values.get(k, None)
            and self.default_values.get(k, None) is not None
        }
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if v is not None and not isinstance(v, (int, float, bool, str))
        }

        # adds value_if_missing attributes
        for k, v in self.value_if_missing.items():
            if isinstance(k, str):
                if k not in kwargs:
                    # Validate that `value_if_missing` keys are compatible
                    # with the observed signature.
                    # If the function does not accept **kwargs,
                    # all value_if_missing keys must be
                    # present in the observed signature names.
                    if k not in self.signature_names and not self.kwargs_name:
                        raise ValueError(
                            f"Unexpected keyword argument {k!r} "
                            f"provided as a value_if_missing input "
                            "for a function that does not accept it. "
                            f"All value_if_missing keys must "
                            f"be in the observed signature: {tuple(self.signature_names)}."
                        )
                    kwargs[k] = v
            elif isinstance(k, int):
                if k >= len(self.signature_names):
                    raise ValueError(
                        f"Unexpected keyword argument {k=} "
                        f"provided as a value_if_missing input "
                        "for a function that does not accept it. "
                        f"All value_if_missing indices must "
                        f"be in the observed signature: {tuple(self.signature_names)}."
                    )
                if k >= len(args):
                    raise NotImplementedError(
                        f"Unexpected keyword argument {k=} "
                        f"provided as a value_if_missing input "
                        "for a function that does not accept it. "
                        f"All value_if_missing indices must "
                        f"be in the observed signature: {tuple(self.signature_names)}, "
                        f"only {len(args)} were given."
                    )
                list_args = list(args)
                list_args[k] = v
                args = tuple(list_args)
            else:
                raise TypeError(
                    f"Unexpected type {type(k)} for a missing value. The key is {k!r}."
                )

        # kwargs may come in a different order each time.
        # dictionaries are ordered and torch.export.export expects
        # dynamic shapes and kwargs to follow the same order.

        ordered_kwargs = {k: kwargs[k] for k in self.signature_names if k in kwargs}
        for k, v in kwargs.items():
            if k not in ordered_kwargs:
                ordered_kwargs[k] = v

        candidate = InputCandidate(
            args, ordered_kwargs, clone=True, cst_kwargs=cst_kwargs
        )
        self.inputs.append(candidate)
        if self._best_candidate is None or len(self._best_candidate) < len(candidate):
            self._best_candidate = candidate

    def add_outputs(self, res: torch.Tensor | tuple[torch.Tensor, ...], latency: float):
        """Stores outputs. They are deepcopied."""
        flat_res, spec = torch.utils._pytree.tree_flatten(res)
        self.outputs_specs.append(spec)
        self.flat_outputs.append(
            [(None if t is None else t.clone().detach()) for t in flat_res]
        )
        self.latencies.append(latency)

    def align_inputs_none_values(self):
        """Once the best candidate is chosen, this method aligns every set of inputs
        on the best candidate, it inserts None at the right position when
        optional inputs are not specified. We consider a set of inputs is aligned
        if this method does not change the original flattened inputs.
        """
        if not self.inputs or self._best_candidate is None:
            raise RuntimeError("No inputs were captured.")

        if all(candidate.aligned_flat_list is not None for candidate in self.inputs):
            # No new inputs, no alignment is necessary.
            return

        # Let's reprocess everything.
        self._captured_inputs = {}
        for candidate in self.inputs:
            if len(set(candidate.kwargs) | set(self._best_candidate.kwargs)) > len(
                self._best_candidate.kwargs
            ):
                raise RuntimeError(
                    f"At least one call to the observed model "
                    f"must contain all the named arguments. "
                    f"candidate kwargs={list(candidate.kwargs)}, "
                    f"best candidate kwargs={list(self._best_candidate.kwargs)}."
                )
            candidate.align_with(
                self._best_candidate, self._captured_inputs, self.signature_names
            )

    def infer_dynamic_shapes(
        self,
        set_batch_dimension_for: set[int | str] | bool | None = None,
        return_flat: bool = False,
    ) -> tuple[dict[int, Any] | None, ...] | dict[str, dict[int, Any] | None]:
        """Infers dynamic shapes based on the collected tensors.
        Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).

        Args:
            set_batch_dimension_for (set[int | str] | bool | None): Set of input identifiers,
                by name (``str``) or position (``int``), for which the first dimension
                should be treated as a dynamic batch dimension. If ``None`` or empty,
                no additional batch dimensions are marked as dynamic.
            return_flat: Tells the function to return a flat tuple instead of
                nested structured. This option is used internally to infer arguments.
        """
        self.align_inputs_none_values()
        assert self._best_candidate is not None  # noqa: S101
        assert self._best_candidate.flat_list is not None  # noqa: S101
        assert self._best_candidate.aligned_flat_list is not None  # noqa: S101

        def _set_batch_dimension(name_or_position) -> bool:
            if not set_batch_dimension_for:
                return False
            if (
                isinstance(set_batch_dimension_for, bool) and set_batch_dimension_for
            ) or name_or_position in set_batch_dimension_for:
                return True
            if isinstance(name_or_position, int):
                torch._check(
                    name_or_position < len(self.signature_names),
                    lambda: f"argument at position {name_or_position} is out of boundary",
                )
                if self.signature_names[name_or_position] in set_batch_dimension_for:
                    return True
            return False

        def _set_batch_dimension_for_flat_index(index) -> bool:
            return _set_batch_dimension(
                # pyrefly: ignore[missing-attribute]
                self._best_candidate.position_to_args_kwargs[index]
            )

        if len(self._best_candidate.flat_list) != len(
            self._best_candidate.aligned_flat_list
        ):
            raise NotImplementedError(
                "infer_dynamic_shapes is not implemented "
                "when the best candidate is not 'aligned'. "
                "This happens when there is no stored set of inputs where "
                "all optional inputs showing in other sets are defined."
            )

        if len({inputs.n_aligned_tensors for inputs in self.inputs}) != 1:
            raise NotImplementedError(
                f"infer_dynamic_shapes is not implemented "
                f"when the number of input tensors are not the same in "
                f"every set of inputs "
                f"{[inputs.n_aligned_tensors for inputs in self.inputs]}."
            )
        shape_lists = [
            [(None if t is None else t.shape) for t in candidate.aligned_flat_list]
            for candidate in self.inputs
            if candidate.aligned_flat_list is not None
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
        if return_flat:
            return tuple(flat_dynamic_shapes)

        # Let's regroup.
        if len(flat_dynamic_shapes) == len(self._best_candidate.args) + len(
            self._best_candidate.kwargs
        ):
            # It means forward method is called with tensors only.
            if (
                not self._best_candidate.kwargs
                and not self._best_candidate.cst_kwargs
                and not self.args_name_and_position
            ):
                # only positional arguments
                return tuple(flat_dynamic_shapes)
            if not self._best_candidate.args:
                # only named arguments
                ds = dict(zip(list(self._best_candidate.kwargs), flat_dynamic_shapes))
                return self._post_process_for_kwargs(
                    {**ds, **dict.fromkeys(self._best_candidate.cst_kwargs, None)}
                )
            if not self.args_name_and_position:
                # positional arguments needs to be moved to the named arguments
                n_args = len(self._best_candidate.args)
                pos_names = self.signature_names[:n_args]
                return self._post_process_for_kwargs(
                    {
                        **dict(zip(pos_names, flat_dynamic_shapes[:n_args])),
                        **dict(
                            zip(
                                list(self._best_candidate.kwargs),
                                flat_dynamic_shapes[n_args:],
                            )
                        ),
                        **dict.fromkeys(self._best_candidate.cst_kwargs, None),
                    }
                )
            # positional arguments needs to be moved to the named arguments
            n_args = min(len(self._best_candidate.args), self.args_name_and_position[1])
            i_kwargs = max(
                len(self._best_candidate.args), self.args_name_and_position[1]
            )
            var_pos = self.args_name_and_position[0]
            pos_names = self.signature_names[:n_args]
            return self._post_process_for_kwargs(
                {
                    **dict(zip(pos_names, flat_dynamic_shapes[:n_args])),
                    var_pos: tuple(flat_dynamic_shapes[n_args:i_kwargs]),
                    **dict(
                        zip(
                            list(self._best_candidate.kwargs),
                            flat_dynamic_shapes[i_kwargs:],
                        )
                    ),
                    **dict.fromkeys(self._best_candidate.cst_kwargs, None),
                }
            )

        # nested types, here comes the fun part because the shapes cannot be unflattened,
        # custom classes must appear in their flattened shape.
        # This does not work in all cases but every time every available argument is flattened
        # with the same number of tensors. The function does not check
        # if that assumption is true.
        flat_inputs, _max_spec = torch.utils._pytree.tree_flatten(
            (self._best_candidate.args, self._best_candidate.kwargs)
        )
        torch._check(
            len(flat_inputs) == len(flat_dynamic_shapes),
            (
                f"Length mismatch len(flat_inputs)={len(flat_inputs)}, "
                f"len(flat_dynamic_shapes)={len(flat_dynamic_shapes)}"
            ),
        )

        index = 0

        def change_function(t):
            nonlocal index
            if index >= len(flat_dynamic_shapes):
                raise RuntimeError(
                    f"Flattened {index} tensors when there are only "
                    f"{len(flat_dynamic_shapes)}."
                )
            res = flat_dynamic_shapes[index]
            index += 1
            return res

        ds_args, ds_kwargs = _flatten_unflatten_for_dynamic_shapes(
            (self._best_candidate.args, self._best_candidate.kwargs),
            change_function=change_function,
        )
        if self._best_candidate.cst_kwargs:
            ds_kwargs = {
                **ds_kwargs,
                **dict.fromkeys(self._best_candidate.cst_kwargs, None),
            }
        if not ds_kwargs and not self.args_name_and_position:
            return tuple(ds_args)
        if not ds_args:
            return self._post_process_for_kwargs(ds_kwargs)

        if not self.args_name_and_position:
            pos_names = self.signature_names[: len(ds_args)]
            return self._post_process_for_kwargs(
                {**dict(zip(pos_names, ds_args)), **ds_kwargs}
            )

        n_args = min(len(ds_args), self.args_name_and_position[1])
        pos_names = self.signature_names[:n_args]
        return self._post_process_for_kwargs(
            {
                **dict(zip(pos_names, ds_args[:n_args])),
                self.args_name_and_position[0]: tuple(ds_args[n_args:]),
                **ds_kwargs,
            }
        )

    def infer_arguments(
        self,
        index_or_candidate: InputCandidate | int | None = None,
        /,
        flat: bool = False,
        as_args_kwargs: bool = False,
    ) -> (
        list[torch.Tensor | None]
        | tuple[torch.Tensor, ...]
        | dict[str, torch.Tensor]
        | tuple[list[torch.Tensor] | tuple[torch.Tensor, ...], dict[str, torch.Tensor]]
    ):
        """Infers arguments based on the collected tensors.

        Args:
            index_or_candidate: If missing, the method selects one set of inputs
                among the available ones, usually the set of inputs containing
                with the highest number of tensors.
                It then replaces None values and missing tensors with empty tensors.
                If not missing, it can be an integer to fetch one of the stored set
                or some inputs.
            flat: If True, it returns a flattened list of tensors,
                if False, it returns a tuple or a dictionary preserving
                the nested structures. The flat version is used internally.
                It produces a single list of tensors easier to process or modify
                rather than a nested structure holding the same tensors.
                The original structure can be restored with
                ``torch.utils._pytree.tree_unflatten(flat_list, self.aligned_spec)``.
                This mechanism is used to replace None values by empty tensors.
            as_args_kwargs: If True, the method always returns `(args, kwargs)`,
                otherwise, it returns either a tuple (only args) or a dictionary
                (only kwargs) or raises an exception if it cannot do so.
        Returns:
            Inferred arguments, every optional tensor is replaced by an empty tensor.
        """
        # This is already checked by _build_inputs_completed_with_none_values
        # but this is not always well captured by tools checking types.
        self.align_inputs_none_values()
        assert self._best_candidate is not None  # noqa: S101
        candidate = None
        if index_or_candidate is None:
            for cand in self.inputs:
                args, kwargs = cand.args, cand.kwargs
                if len(args) == len(self._best_candidate.args or ()) and len(
                    kwargs
                ) == len(self._best_candidate.kwargs or {}):
                    candidate = cand
                    break
        elif isinstance(index_or_candidate, int):
            torch._check(
                index_or_candidate < len(self.inputs),
                lambda: (
                    f"No stored input set for index="
                    f"{index_or_candidate}<{len(self.inputs)}."
                ),
            )
            candidate = self.inputs[index_or_candidate]
        else:
            candidate = index_or_candidate

        assert candidate is not None  # noqa: S101
        if candidate.aligned_flat_list is None:
            raise RuntimeError(
                f"Candidate {candidate} has no aligned flat list of tensors, "
                f"index_or_candidate={index_or_candidate}. You should call "
                f"method 'align_with'."
            )

        aligned_flat_list = candidate.aligned_flat_list
        assert aligned_flat_list is not None  # noqa: S101
        if any(t is None for t in aligned_flat_list):
            dynamic_shapes = self.infer_dynamic_shapes(return_flat=True)
            assert isinstance(dynamic_shapes, tuple)  # noqa: S101
            aligned_flat_list = list(aligned_flat_list)
            for index in range(len(aligned_flat_list)):
                if aligned_flat_list[index] is not None:
                    continue
                shape = dynamic_shapes[index]
                all_non_empty_tensors = [
                    c.aligned_flat_list[index]
                    for c in self.inputs
                    if c.aligned_flat_list is not None
                ]
                all_non_empty_tensors_not_none = [
                    t for t in all_non_empty_tensors if t is not None
                ]
                if not all_non_empty_tensors_not_none:
                    raise RuntimeError(
                        f"There is no tensor at position {index} in any flattened inputs."
                    )
                tensor = all_non_empty_tensors_not_none.pop()
                if tensor.numel() == 0:
                    aligned_flat_list[index] = tensor
                    continue
                if not shape:
                    aligned_flat_list[index] = torch.zeros(
                        tensor.shape, dtype=tensor.dtype, device=tensor.device
                    )
                    continue
                dim = max(shape)
                torch._check(
                    dim < tensor.ndim,
                    lambda index=index, shape=shape, tshape=tensor.shape: (
                        f"Tensor shape {tshape} does not match the "
                        f"dynamic shape {shape} at position {index}."
                    ),
                )
                new_shape = list(tensor.shape)
                new_shape[dim] = 0
                aligned_flat_list[index] = torch.empty(
                    tuple(new_shape), dtype=tensor.dtype, device=tensor.device
                )
        if flat:
            return aligned_flat_list
        args, kwargs = torch.utils._pytree.tree_unflatten(
            aligned_flat_list,
            # pyrefly: ignore[bad-argument-type]
            candidate.aligned_spec,
        )
        if self._best_candidate.cst_kwargs:
            kwargs = {**kwargs, **self._best_candidate.cst_kwargs}

        if not as_args_kwargs:
            if not kwargs:
                return args
            if not args:
                return kwargs

            # We need to move args to kwargs
            if self.args_name_and_position:
                raise RuntimeError(
                    "Cannot return arguments "
                    "as a single tuple or a single dictionary "
                    "because of '*args' in the function signature. "
                    "You need to set `as_args_kwargs=True`."
                )
            n_args = len(args)
            pos_names = self.signature_names[:n_args]
            return {**dict(zip(pos_names, args[:n_args])), **kwargs}

        # Generic case.
        return tuple(args), kwargs

    def _post_process_for_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """:func:`torch.export.export` requires dynamic shapes and keyword arguments
        that are not part of the explicit function signature but are collected via
        ``**<kwargs_name>`` to be wrapped under the corresponding parameter name
        (``self.kwargs_name``) as ``{<kwargs_name>: {'param': shape or tensor}}``.
        This function ensures this wrapping is performed when ``self.kwargs_name`` is set.
        """
        if not self.kwargs_name:
            # Nothing to do here.
            return kwargs
        to_be_moved = {k for k in kwargs if k not in self.signature_names}
        if not to_be_moved:
            return kwargs
        keywords = {k: v for k, v in kwargs.items() if k in to_be_moved}
        new_kwargs = {k: v for k, v in kwargs.items() if k not in to_be_moved}
        if self.kwargs_name in new_kwargs:
            raise ValueError(
                f"Keyword argument name collision: received a keyword argument "
                f"'{self.kwargs_name}' which conflicts with the **{self.kwargs_name} "
                "parameter used to collect extra keyword arguments. "
                "Passing a keyword argument with this name is not supported."
            )
        return {**new_kwargs, self.kwargs_name: keywords}


class InputObserver:
    """Steals forward method to collect inputs and outputs.
    This information is used to infer dynamic shapes and
    export arguments.

    Args:
        value_if_missing: If an argument is missing,
            a default value will be taken in this dictionary,
            this is used when after the prefill step, an argument
            disappears (such as `pixel_values`) and another one
            is added (such as `past_key_values`).
            The values are only to infer dynamic shapes and arguments,
            not to run the model.

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
    >>>     (),
    >>>     kwargs=input_observer.infer_arguments(),
    >>>     dynamic_shapes.input_observer.infer_dynamic_shapes(),
    >>> )

    The last example considers an LLM taking images and text as inputs.
    The first call to the forward method which we try to export has `pixel_values`
    but no `past_key_values`. The next calls do not have `pixel_values` but
    `past_key_values`. The observer understands `pixel_values` and `past_key_values`
    are needed but they may not be both specified at the same time.
    Since `pixel_values` only appears in the first call, the observer cannot
    tell how to infer an empty tensor for this argument. That's what the argument
    `value_if_missing` is for. The following example is more than a dummy example
    but shows how to use it with ``transformers``.

    .. code-block:: python

        from transformers import pipeline

        model_id = "tiny-random/gemma-3"
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device="cpu",
            trust_remote_code=True,
            max_new_tokens=3,
            dtype=torch.float16,
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                    },
                    {"type": "text", "text": "What animal is on the candy?"},
                ],
            },
        ]
        observer = InputObserver(
            value_if_missing=dict(
                pixel_values=torch.empty((0, 3, 896, 896), dtype=torch.float16)
            )
        )
        with observer(pipe.model):
            pipe(text=messages, max_new_tokens=4)

    .. versionadded:: 2.11.0
    """

    def __init__(self, value_if_missing: dict[str | int, Any] | None = None):
        self.info: InputObserverInfo | None = None
        self.value_if_missing = value_if_missing or {}

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
        begin = time.perf_counter()
        res = _captured_method(*args, **kwargs)
        duration = time.perf_counter() - begin
        if n_stored < _store_n_calls:
            self.info.add_outputs(res, latency=duration)
        return res

    def num_obs(self) -> int:
        """Returns the number of stored set of inputs."""
        return 0 if not self.info else len(self.info)

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
        if not hasattr(model, method_name):
            raise ValueError(
                f"Model type {model} does not have a method {method_name!r}."
            )
        captured_method = getattr(model, method_name)
        sig = inspect.signature(captured_method)
        if self.info is None:
            kwargs_names = [
                p
                for p in sig.parameters
                if sig.parameters[p].kind == inspect.Parameter.VAR_KEYWORD
            ]
            args_names = [
                (p, i)
                for (i, p) in enumerate(sig.parameters)
                if sig.parameters[p].kind == inspect.Parameter.VAR_POSITIONAL
            ]
            self.info = InputObserverInfo(
                signature_names=list(sig.parameters),
                default_values={
                    p.name: p.default
                    for p in sig.parameters.values()
                    if p.default != inspect.Parameter.empty
                    and isinstance(p.default, (int, bool, str, float))
                },
                value_if_missing=self.value_if_missing,
                args_name_and_position=args_names[0] if args_names else None,
                kwargs_name=kwargs_names[0] if kwargs_names else None,
            )
        n_already_stored = len(self.info)
        lambda_method = lambda *args, _cm=captured_method, _snc=(  # noqa: E731
            store_n_calls + n_already_stored
        ), **kwargs: self._replaced_method(
            *args, _captured_method=_cm, _store_n_calls=_snc, **kwargs
        )

        # It may happen that the signature of the forward is used to trigger a preprocessing.
        # This is used in GenerationMixin (transformers):
        #   position_ids_key = "decoder_position_ids" if ... else "position_ids"
        #   if position_ids_key in set(inspect.signature(self.forward).parameters.keys()):
        lambda_method.__signature__ = sig  # type: ignore[attr-defined]

        setattr(model, method_name, lambda_method)

        try:
            yield self
        finally:
            setattr(model, method_name, captured_method)

    def _check_captured(self):
        if self.info is None:
            raise RuntimeError("No inputs were captured.")

    def infer_dynamic_shapes(
        self, set_batch_dimension_for: set[int | str] | bool | None = None
    ) -> tuple[dict[int, Any] | None, ...] | dict[str, dict[int, Any] | None]:
        """
        Infers dynamic shapes. Most of the time, models do support a batch dimension
        but this batch dimension has the same value for every input sample.
        Instead of running inference on new samples, argument `set_batch_dimension_for`
        can be used to tell the first dimension is a dynamic dimension for a particular
        set of inputs referenced by their name (str) or their position (int).

        Args:
            set_batch_dimension_for (set[int | str] | bool | None): A set of input
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
        self,
        index_or_args_or_kwargs: tuple[Any] | dict[str, Any] | int | None = None,
        flat: bool = False,
        as_args_kwargs: bool = False,
    ) -> (
        list[torch.Tensor | None]
        | tuple[torch.Tensor, ...]
        | dict[str, torch.Tensor]
        | tuple[list[torch.Tensor] | tuple[torch.Tensor, ...], dict[str, torch.Tensor]]
    ):
        """Infers arguments based on the collected tensors.

        Args:
            index_or_args_or_kwargs: If missing, the method selects one set of inputs
                among the available ones, usually the set of inputs containing
                with the highest number of tensors.
                It then replaces None values and missing tensors with empty tensors.
                If not missing, it can be an integer to fetch one of the stored set
                or some inputs.
            flat: If True, it returns a flattened list of tensors,
                if False, it returns a tuple or a dictionary preserving
                the nested structures. The flat version is used internally.
                It produces a single list of tensors easier to process or modify
                rather than a nested structure holding the same tensors.
                The original structure can be restored with
                ``torch.utils._pytree.tree_unflatten(flat_list, self.aligned_spec)``.
                This mechanism is used to replace None values by empty tensors.
            as_args_kwargs: If True, the method always returns `(args, kwargs)`,
                otherwise, it returns either a tuple (only args) or a dictionary
                (only kwargs) or raises an exception if it cannot do so.
        Returns:
            Inferred arguments, every optional tensor is replaced by an empty tensor.
        """
        self._check_captured()
        assert self.info is not None  # noqa: S101
        index_or_candidate: int | InputCandidate | None = None
        if index_or_args_or_kwargs is None or isinstance(index_or_args_or_kwargs, int):
            index_or_candidate = index_or_args_or_kwargs
        else:
            if isinstance(index_or_args_or_kwargs, tuple):
                index_or_candidate = InputCandidate(
                    args=index_or_args_or_kwargs, kwargs={}, clone=False, cst_kwargs={}
                )
            elif isinstance(index_or_args_or_kwargs, dict):
                index_or_candidate = InputCandidate(
                    args=(),
                    kwargs={
                        k: v
                        for k, v in index_or_args_or_kwargs.items()
                        if k not in self.info.default_values
                    },
                    clone=False,
                    cst_kwargs={
                        k: v
                        for k, v in index_or_args_or_kwargs.items()
                        if k in self.info.default_values
                    },
                )
            else:
                raise ValueError(
                    f"Unexpected type {type(index_or_args_or_kwargs)} "
                    f"for index_or_args_or_kwargs."
                )
            self.info.align_inputs_none_values()
            index_or_candidate.align_with(
                # pyrefly: ignore[bad-argument-type]
                self.info._best_candidate,
                # pyrefly: ignore[bad-argument-type]
                self.info._captured_inputs,
                self.info.signature_names,
            )
        return self.info.infer_arguments(
            index_or_candidate,
            flat=flat,
            as_args_kwargs=as_args_kwargs,
        )

    def check_discrepancies(
        self,
        onnx_program: torch.onnx.ONNXProgram,
        atol: float = 1e-4,
        rtol: float = 0.1,
        progress_bar: bool = False,
        initializer: Callable[
            [str | bytes], ort.InferenceSession
        ] = _onnx_program._ort_session_initializer,
        skip_none: bool = True,
    ) -> list[dict[str, str | int | float | bool]]:
        """Computes the discrepancies between the saved inputs and outputs
        with the saved onnx model.

        Args:
            onnx_program: Exported Model to verify.
            atol: Absolute tolerance, recommended values, 1e-4 for float, 1e-2 for float16.
            rtol: Relative tolerance.
            progress_bar: Shows a progress bar (requires `tqdm`).
            initializer: The function called to initialize the ONNX Runtime inference
                session with the specified model. By default, it uses the
                `_ort_session_initializer` function.
            skip_none: Does not check discrepancies when an output is None.

        Returns:
            A list of dictionaries, ready to be consumed by a dataframe.

        The function catches exceptions, it shows the error in the returned
        summary.
        """
        # For big models, we should consider taking a filename to avoid the users
        # creating the model proto twice.
        self._check_captured()
        assert self.info is not None  # noqa: S101

        onnx_program.initialize_inference_session(initializer)

        input_names = [i.name for i in onnx_program.model.graph.inputs]
        io_sets = list(
            zip(self.info.inputs, self.info.flat_outputs, self.info.latencies)
        )
        if progress_bar:
            from tqdm import tqdm

            loop = tqdm(io_sets)
        else:
            loop = io_sets
        data: list[dict[str, Any]] = []
        for inputs, outputs, latency in loop:
            assert inputs.aligned_flat_list is not None  # noqa: S101
            if len(input_names) != len(inputs.aligned_flat_list):
                raise RuntimeError(
                    f"There are ({len(inputs.aligned_flat_list)}) "
                    f"tensors but the model expects {len(input_names)}."
                )
            n_none = sum(t is None for t in inputs.aligned_flat_list)
            n_empty = sum(t is None or t.numel() == 0 for t in inputs.aligned_flat_list)

            feeds = dict(zip(input_names, self.info.infer_arguments(inputs, flat=True)))

            begin = time.perf_counter()
            try:
                ort_outputs = onnx_program(**feeds)
                error = None
            except Exception as e:
                error = str(e)
                ort_outputs = None

            duration = time.perf_counter() - begin
            if error:
                diff: dict[str, str | int | float | bool] = dict(
                    error=error, SUCCESS=False
                )
            elif ort_outputs is None or len(outputs) != len(ort_outputs):
                diff = dict(SUCCESS=False, error="not the same number of outputs")
            else:
                success = True
                err_abs = 0.0
                err_rel = 0.0
                error = ""
                for torch_tensor, ort_tensor in zip(outputs, ort_outputs):
                    if torch_tensor is None or ort_tensor is None:
                        if type(torch_tensor) is not type(ort_tensor) and not skip_none:
                            success = False
                            error = "missing output"
                            break
                        continue
                    if torch_tensor.shape != ort_tensor.shape:
                        success = False
                        error = "not the same shape"
                        break
                    if torch_tensor.dtype != ort_tensor.dtype:
                        success = False
                        error = "not the same type"
                        break
                    err = (torch_tensor - ort_tensor).abs().max().item()
                    err_abs = max(err_abs, err)
                    if err_abs > atol:
                        success = False
                    err = (
                        (
                            (torch_tensor - ort_tensor).abs()
                            / (torch_tensor.abs() + rtol)
                        )
                        .max()
                        .item()
                    )
                    err_rel = max(err_rel, err)
                    if err_rel > rtol:
                        success = False
                diff = dict(SUCCESS=success, abs=err_abs, rel=err_rel)
            diff.update(
                dict(
                    index=len(data),
                    duration_torch=latency,
                    ort_duration=duration,
                    n_inputs=len(input_names),
                    n_none=n_none,
                    n_empty=n_empty,
                )
            )
            data.append(diff)
        onnx_program.release()
        return data
