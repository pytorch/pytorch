from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
from torch.utils._pytree import tree_flatten_with_path, tree_map


KeyPath = Tuple[Any, ...]
NonTensorShapeFn = Callable[[Union[int, float]], Tuple[Any, ...]]


def track_dynamism_across_examples(
    example_inputs: List[Any],
) -> Dict[Any, Any]:
    """
    This function analyzes a list of example inputs to determine the dynamism of their shapes.
    It tracks whether the dimensions of tensors or non-tensor values change across
    different examples. The function returns a dictionary where each key represents
    a path to a value in the input examples, and the corresponding value is a tuple
    indicating which dimensions are dynamic (i.e., change across examples). This
    helps in understanding how the structure of data varies across different instances.
    """
    tracking: Dict[KeyPath, Tuple[List[Set[Any]], bool]] = {}

    for ex in example_inputs:
        leaves_with_paths, _ = tree_flatten_with_path(ex)
        for key_path, value in leaves_with_paths:
            if not isinstance(value, (int, float, torch.Tensor)):
                continue
            if isinstance(value, torch.Tensor):
                shape = tuple(value.shape)
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

    output: Dict[Any, Any] = {}
    for key_path, (dim_sets, is_tensor) in tracking.items():
        dyn = tuple(len(s) > 1 for s in dim_sets)
        if is_tensor:
            final_dyn = dyn
        else:
            dyn_list = list(dyn)
            while dyn_list and not dyn_list[-1]:
                dyn_list.pop()
            final_dyn = tuple(dyn_list) if dyn_list else dyn
        key_str = "L" + "".join(f"{str(k)}" for k in key_path)
        output[key_path[0].key] = {key_str: final_dyn}
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
