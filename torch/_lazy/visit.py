import torch


def visit_structure(structure, select_fn, apply_fn, scope=None, strict=True):
    """Recursively traverse nested structureure and calls the apply_fn callable on
    the items accepted by the selector.

    Args:
        structure: A nested data structure to traverse recursively.
        select_fn: A callable that returns true if the item passed should be
            selected.
        apply_fn: A callable that performs some action using the selected item
        strict: Strictly checks that an item in the nested structure is either
            a list/dict/tuple or selected by the select_fn. Otherwise, raises
            an error. Defaults to False.
        scope: The current hierarchical scope of the data structure. Defaults
            to None.
    Returns:
        The values returned from apply_fn with the same structure as the original
        structure
    """
    scope = scope or []

    if isinstance(structure, (list, tuple)):
        return type(structure)(
            visit_structure(val, select_fn, apply_fn, scope + [i], strict)
            for i, val in enumerate(structure)
        )
    if isinstance(structure, dict):
        return {
            k: visit_structure(val, select_fn, apply_fn, scope + [k], strict)
            for k, val in structure.items()
        }
    if select_fn(structure):
        return apply_fn(structure, scope)
    if strict:
        raise TypeError(f"Unknown type: {type(structure)}")


def visit_tensors(tensors, apply_fn, scope=None, strict=True):
    return visit_structure(
        tensors,
        lambda t: isinstance(t, torch.Tensor),
        apply_fn,
        scope,
        strict,
    )


def visit_lazy_tensors(tensors, apply_fn, scope=None, strict=True):
    return visit_structure(
        tensors,
        lambda t: isinstance(t, torch.Tensor) and t.device.type == "lazy",
        apply_fn,
        scope,
        strict,
    )


def flatten_tensors(tensors, scope=None, strict=True):
    flattened = []

    def apply_fn(tensor, scope):
        flattened.append(tensor)

    visit_tensors(tensors, apply_fn, scope, strict)
    return flattened
