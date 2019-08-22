import torch
from collections import OrderedDict

"""
This file contains helper functions that implement experimental functionality
for named tensors in python. All of these are experimental, unstable, and
subject to change or deletion.
"""


def _assert_namedtensor_build(api_name):
    if not torch._C._BUILD_NAMEDTENSOR:
        raise RuntimeError('NYI: {} is experimental and a part '
                           'of our named tensors project.'.format(api_name))


def _build_dim_map(tensor):
    """Returns a map of { dim: dim_name } where dim is a name if the dim is named
    and the dim index otherwise."""
    return OrderedDict([(idx if name is None else name, name)
                        for idx, name in enumerate(tensor.names)])


def _namer_api_name(inplace):
    if inplace:
        return 'names_'
    else:
        return 'view_names'


def _expand_single_glob(numel_pre_glob, numel_post_glob, names):
    return names[numel_pre_glob:len(names) - numel_post_glob]


def _update_names_with_list(tensor, names, inplace):
    # Special case for tensor.view_names(None)
    if len(names) == 1 and names[0] is None:
        return tensor._update_names(None, inplace)

    glob_indices = [i for i, x in enumerate(names) if x == '*']
    if len(glob_indices) >= 2:
        raise RuntimeError('{}: More than one \'*\' found in names ('
                           '{}). This function supports up to one \'*\'.'
                           .format(_namer_api_name(inplace), names))
    elif len(glob_indices) == 1:
        glob_idx = glob_indices[0]
        globbed_names = _expand_single_glob(glob_idx, len(names) - glob_idx - 1, tensor.names)
        return tensor._update_names(
            names[:glob_idx] + globbed_names + names[glob_idx + 1:], inplace)
    else:
        return tensor._update_names(names, inplace)


def _update_names_with_mapping(tensor, rename_map, inplace):
    dim_map = _build_dim_map(tensor)
    for old_dim in rename_map.keys():
        new_dim = rename_map[old_dim]
        if old_dim in dim_map.keys():
            dim_map[old_dim] = new_dim
        else:
            raise RuntimeError(('{api_name}: Tried to rename dim \'{old_dim}\' to dim '
                                '{new_dim} in Tensor[{dims}] but dim \'{old_dim}\' does not exist')
                               .format(old_dim=old_dim, new_dim=new_dim, dims=tensor.names,
                                       api_name=_namer_api_name(inplace)))
    return tensor._update_names(tuple(dim_map.values()), inplace)


def _update_names(tensor, names, rename_map, inplace):
    """There are two usages:

    tensor.view_names(*names) returns a view on tensor with named dims `names`.
    `names` must be of length `tensor.dim()`; otherwise, if '*' is in `names`,
    then it is expanded greedily to be equal to the corresponding names from
    `tensor.names`.

    For example,
    ```
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.view_names('*', 'height', 'width').names
    ('N', 'C', 'height', 'width')

    >>> x.view_names('batch', '*', 'width').names
    ('batch', 'C', 'H', 'width')
    ```

    tensor.view_names(**rename_map) returns a view on tensor that has renamed dims
        as specified in the mapping `rename_map`.

    For example,
    ```
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.view_names(W='width', H='height').names
    ('N', 'C', 'height', 'width')
    ```

    Finally, tensor.view_names has an in-place version called tensor.names_.
    """
    _assert_namedtensor_build(_namer_api_name(inplace))

    has_names = len(names) > 0
    has_rename_pairs = bool(rename_map)
    if has_names and has_rename_pairs:
        raise RuntimeError('{api_name}: This function takes either positional '
                           'args or keyword args, but not both. Use tensor.{api_name}(*names) '
                           'to name dims and tensor.{api_name}(**rename_map) to rename '
                           'dims.'.format(api_name=_namer_api_name(inplace)))

    # Special case for tensor.view_names(*[]), which is valid for a 0 dim tensor.
    if not has_names and not has_rename_pairs:
        return _update_names_with_list(tensor, names, inplace)

    if has_names:
        return _update_names_with_list(tensor, names, inplace)
    return _update_names_with_mapping(tensor, rename_map, inplace)
