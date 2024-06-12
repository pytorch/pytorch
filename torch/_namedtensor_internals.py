from collections import OrderedDict

"""
This file contains helper functions that implement experimental functionality
for named tensors in python. All of these are experimental, unstable, and
subject to change or deletion.
"""


def check_serializing_named_tensor(tensor):
    if tensor.has_names():
        raise RuntimeError(
            "NYI: Named tensors don't support serialization. Please drop "
            "names via `tensor = tensor.rename(None)` before serialization."
        )


def build_dim_map(tensor):
    """Returns a map of { dim: dim_name } where dim is a name if the dim is named
    and the dim index otherwise."""
    return OrderedDict(
        [(idx if name is None else name, name) for idx, name in enumerate(tensor.names)]
    )


def unzip_namedshape(namedshape):
    if isinstance(namedshape, OrderedDict):
        namedshape = namedshape.items()
    if not hasattr(namedshape, "__iter__") and not isinstance(namedshape, tuple):
        raise RuntimeError(
            f"Expected namedshape to be OrderedDict or iterable of tuples, got: {type(namedshape)}"
        )
    if len(namedshape) == 0:
        raise RuntimeError("Expected namedshape to non-empty.")
    return zip(*namedshape)


def namer_api_name(inplace):
    if inplace:
        return "rename_"
    else:
        return "rename"


def is_ellipsis(item):
    return item == Ellipsis or item == "..."


def single_ellipsis_index(names, fn_name):
    ellipsis_indices = [i for i, name in enumerate(names) if is_ellipsis(name)]
    if len(ellipsis_indices) >= 2:
        raise RuntimeError(
            f"{fn_name}: More than one Ellipsis ('...') found in names ("
            f"{names}). This function supports up to one Ellipsis."
        )
    if len(ellipsis_indices) == 1:
        return ellipsis_indices[0]
    return None


def expand_single_ellipsis(numel_pre_glob, numel_post_glob, names):
    return names[numel_pre_glob : len(names) - numel_post_glob]


def replace_ellipsis_by_position(ellipsis_idx, names, tensor_names):
    globbed_names = expand_single_ellipsis(
        ellipsis_idx, len(names) - ellipsis_idx - 1, tensor_names
    )
    return names[:ellipsis_idx] + globbed_names + names[ellipsis_idx + 1 :]


def resolve_ellipsis(names, tensor_names, fn_name):
    """
    Expands ... inside `names` to be equal to a list of names from `tensor_names`.
    """
    ellipsis_idx = single_ellipsis_index(names, fn_name)
    if ellipsis_idx is None:
        return names
    return replace_ellipsis_by_position(ellipsis_idx, names, tensor_names)


def update_names_with_list(tensor, names, inplace):
    # Special case for tensor.rename(None)
    if len(names) == 1 and names[0] is None:
        return tensor._update_names(None, inplace)

    return tensor._update_names(
        resolve_ellipsis(names, tensor.names, namer_api_name(inplace)), inplace
    )


def update_names_with_mapping(tensor, rename_map, inplace):
    dim_map = build_dim_map(tensor)
    for old_dim in rename_map.keys():
        new_dim = rename_map[old_dim]
        if old_dim in dim_map.keys():
            dim_map[old_dim] = new_dim
        else:
            raise RuntimeError(
                f"{namer_api_name(inplace)}: Tried to rename dim '{old_dim}' to dim "
                f"{new_dim} in Tensor[{tensor.names}] but dim '{old_dim}' does not exist"
            )
    return tensor._update_names(tuple(dim_map.values()), inplace)


def update_names(tensor, names, rename_map, inplace):
    """There are two usages:

    tensor.rename(*names) returns a view on tensor with named dims `names`.
    `names` must be of length `tensor.dim()`; otherwise, if '...' is in `names`,
    then it is expanded greedily to be equal to the corresponding names from
    `tensor.names`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename('...', 'height', 'width').names
    ('N', 'C', 'height', 'width')

    >>> # xdoctest: +SKIP
    >>> x.rename('batch', '...', 'width').names
    ('batch', 'C', 'H', 'width')

    ```

    tensor.rename(**rename_map) returns a view on tensor that has rename dims
        as specified in the mapping `rename_map`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename(W='width', H='height').names
    ('N', 'C', 'height', 'width')

    ```

    Finally, tensor.rename has an in-place version called tensor.rename_.
    """
    has_names = len(names) > 0
    has_rename_pairs = bool(rename_map)
    if has_names and has_rename_pairs:
        raise RuntimeError(
            f"{namer_api_name(inplace)}: This function takes either positional "
            f"args or keyword args, but not both. Use tensor.{namer_api_name(inplace)}(*names) "
            f"to name dims and tensor.{namer_api_name(inplace)}(**rename_map) to rename "
            "dims."
        )

    # Special case for tensor.rename(*[]), which is valid for a 0 dim tensor.
    if not has_names and not has_rename_pairs:
        return update_names_with_list(tensor, names, inplace)

    if has_names:
        return update_names_with_list(tensor, names, inplace)
    return update_names_with_mapping(tensor, rename_map, inplace)
