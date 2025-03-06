import re
from typing import Any, Callable, Union

import torch
from torch.utils._pytree import tree_flatten_with_path, tree_map


KeyPath = tuple[Any, ...]
NonTensorShapeFn = Callable[[Union[int, float]], tuple[Any, ...]]

__all__ = [
    "normalize_source_name",
    "module_to_nested_dict",
    "track_dynamism_across_examples",
    "clone_and_convert_to_meta",
]


def normalize_source_name(name: str) -> str:
    # Match attribute access like .x and replace with ['x']
    return re.sub(r"\.([a-zA-Z_][a-zA-Z0-9_]*)", r"['\1']", name)


def module_to_nested_dict(module: torch.nn.Module) -> dict[str, Any]:
    """Recursively converts an nn.Module into a nested dictionary with explicit 'parameters' and 'modules' keys."""
    self_dict: dict[str, Any] = {}

    self_dict["_parameters"] = {}
    self_dict["_modules"] = {}

    for attr_name in dir(module):
        if not attr_name.startswith("_") and not callable(getattr(module, attr_name)):
            attr_value = getattr(module, attr_name)
            if (
                not isinstance(attr_value, torch.nn.Module)
                and isinstance(attr_value, (int, float, torch.Tensor))
                and type(attr_value) is not bool
            ):
                self_dict[attr_name] = attr_value

    for name, param in module.named_parameters(recurse=False):
        self_dict["_parameters"][name] = param
    for name, buffer in module.named_buffers(recurse=False):
        self_dict["_parameters"][name] = buffer

    for name, submodule in module.named_children():
        self_dict["_modules"][name] = module_to_nested_dict(submodule)

    return self_dict


def track_dynamism_across_examples(
    example_inputs: list[Any],
) -> dict[Any, Any]:
    """
    This function analyzes a list of example inputs to determine the dynamism of their shapes.
    It tracks whether the dimensions of tensors or non-tensor values change across
    different examples. The function returns a dictionary where each key represents
    a path to a value in the input examples, and the corresponding value is a tuple
    indicating which dimensions are dynamic (i.e., change across examples). This
    helps in understanding how the structure of data varies across different instances.
    """
    tracking: dict[KeyPath, tuple[list[set[Any]], bool]] = {}

    for ex in example_inputs:
        if "self" in ex and isinstance(ex["self"], torch.nn.Module):
            ex["self"] = module_to_nested_dict(ex["self"])
        leaves_with_paths, _ = tree_flatten_with_path(ex)
        for key_path, value in leaves_with_paths:
            if not isinstance(value, (int, float, torch.Tensor)):
                continue
            if isinstance(value, torch.Tensor):
                shape: tuple[int | float, ...] = tuple(value.shape)
                is_tensor = True
            else:
                shape = (value,)
                is_tensor = False
            if key_path not in tracking:
                tracking[key_path] = ([set() for _ in range(len(shape))], is_tensor)
            else:
                dim_sets, flag = tracking[key_path]
                if flag != is_tensor:
                    pass
                while len(dim_sets) < len(shape):
                    dim_sets.append(set())
            for i, dim in enumerate(shape):
                tracking[key_path][0][i].add(dim)

    output: dict[Any, Any] = {}
    for key_path, (dim_sets, _is_tensor) in tracking.items():
        final_dyn = tuple(len(s) > 1 for s in dim_sets)
        key_str = "L" + "".join(f"{str(k)}" for k in key_path)
        key = key_path[0].key  # type: ignore[attr-defined]
        if key not in output:
            output[key] = {}
        output[key][key_str] = final_dyn
    return output


def clone_and_convert_to_meta(example_input: Any) -> Any:
    """
    This function takes a list of example inputs and for each tensor, clones it and converts it to device=meta.
    For non-tensor values, it keeps the reference. It uses pytree to handle nested structures recursively.
    """

    def transform_fn(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone().to(device="meta")
        return value

    return tree_map(transform_fn, example_input)
