import collections
import copy
import functools
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist

# Import the entire FSDP file to avoid circular imports
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor
from torch.distributed.fsdp._runtime_utils import _clear_grads_if_needed
from torch.distributed.fsdp._shard_utils import _gather_state_dict
from torch.distributed.fsdp.flat_param import FlatParameter, FlatParamHandle


def sorted_items(dictionary: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    keys = sorted(dictionary.keys())
    for k in keys:
        yield k, dictionary[k]


class _ConsolidatedOptimState:
    """
    This holds the consolidated optimizer state on the target rank. Positive-
    dimension tensor state is communicated across ranks, while zero-dimension
    tensor state and non-tensor state is taken directly from the target rank.

    PyTorch version 1.12 moved to using zero-dimension tensors for scalar
    values, but user implemented optimizers may still use float (i.e. a
    non-tensor). Thus, we support both and handle them identically.

    Attributes:
        tensor_state (Dict[str, torch.Tensor]): Mapping from positive-dimension
            tensor state name to the unsharded flattened tensor representing
            the state.
        zero_dim_tensor_state (Dict[str, torch.Tensor]): Mapping from zero-
            dimension tensor state name to its value.
        non_tensor_state (Dict[str, Any]): Mapping from non-tensor state
            name to its value.
    """

    tensor_state: Dict[str, torch.Tensor] = {}
    zero_dim_tensor_state: Dict[str, torch.Tensor] = {}
    non_tensor_state: Dict[str, Any] = {}


class _PosDimTensorInfo(NamedTuple):
    """
    Meatadata for positive-dimension tensors used internally for
    :meth:`scatter_full_optim_state_dict`.

    Attributes:
        shape (torch.Size): Sharded tensor shape (which is equal to the
            unsharded tensor shape if the tensor is optimizer state for a
            non-FSDP parameter and is hence not sharded).
        dtype (torch.dtype): Data type of the tensor.
    """

    shape: torch.Size
    dtype: torch.dtype


class _OptimStateKey(NamedTuple):
    """
    This represents an optimizer state key that may be used commonly across
    ranks. It is based on the unflattened parameter names rather than parameter
    IDs to make it indepenendent of each rank's own optimizer construction.
    """

    unflat_param_names: Tuple[str, ...]
    is_flat_param: bool


def _unflatten_optim_state(
    flat_param: FlatParameter,
    flat_param_state: Dict[str, Any],
    fsdp_module,
    to_save: bool,
    shard_state: bool,
) -> List[Dict[str, Any]]:
    """
    Unflattens the optimizer state, consisting of the "state" part and the
    "param_groups" part. Unflattening the "state" part involves consolidating
    the state on the target rank and remapping from flattened to unflattened
    parameter IDs, and the "param_groups" part only involves remapping from
    flattened to unflattened parameter IDs.

    Args:
        flat_param (FlatParameter): The flattened parameter.
        flat_param_state (Dict[str, Any]): Entry for the flattened parameter
            in the "state" part of the optimizer state dict.
        fsdp_module (FullyShardedDataParallel): FSDP module that owns
            ``flat_param``, i.e. holds it in ``self.params``.
        to_save (bool): Whether to save the state on this rank.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flattened parameter
        ``flat_param`` if on the target rank or an empty :class:`list`
        otherwise. The final optimizer state dict will need to map these
        entries using the proper unflattened parameter IDs.
    """
    _clear_grads_if_needed(fsdp_module._fsdp_handles(fsdp_module))
    consolidated_state = _communicate_optim_state(
        flat_param,
        flat_param_state,
        fsdp_module,
        to_save,
    )
    unflat_param_state = (
        _unflatten_communicated_optim_state(
            fsdp_module,
            flat_param,
            consolidated_state,
            shard_state,
        )
        if to_save or shard_state
        else []
    )
    if to_save:
        for optim_state in unflat_param_state:
            for key in list(optim_state.keys()):
                state = optim_state[key]
                if isinstance(state, torch.Tensor):
                    optim_state[key] = state.cpu()
    return unflat_param_state


def _communicate_optim_state(
    flat_param: FlatParameter,
    flat_param_state: Dict[str, Any],
    fsdp_module,
    to_save: bool,
) -> _ConsolidatedOptimState:
    """
    Communicates the optimizer state for a flattened parameter ``flat_param``
    across ranks so that the target rank holds the entire non-sharded optimizer
    state.

    If ``N`` is the number of tensor optimizer states in the optimizer state
    dict, then the communication complexity is 0 if ``N = 0`` and ``N + 1``
    otherwise (where the plus 1 comes from all-gathering the padding per rank).

    Args:
        flat_param (FlatParameter): The flattened parameter.
        flat_param_state (Dict[str, Any]): The entry in the "state" part of the
            optimizer state dict corresponding to the flattened parameter.
        fsdp_module (FullyShardedDataParallel): FSDP module that owns
            ``flat_param``, i.e. holds it in ``self.params``.
        to_save (bool): Whether to save the state on this rank.

    Returns:
        ConsolidatedOptimState: Consolidated optimizer state for
        ``flat_param``; the state is not populated for non-target ranks.
    """
    state = _ConsolidatedOptimState()
    tensor_state, zero_dim_tensor_state, non_tensor_state = (
        state.tensor_state,
        state.zero_dim_tensor_state,
        state.non_tensor_state,
    )
    group = fsdp_module.process_group

    for state_name, value in sorted_items(flat_param_state):
        # Positive-dimension tensor state: communicate across ranks
        if torch.is_tensor(value) and value.dim() > 0:
            # If the parameter is not sharded, then neither is the
            # positive-dimension tensor state, so no need to communicate it --
            # we take the target rank's value
            if (
                fsdp_module.world_size == 1
                or fsdp_module.sharding_strategy == fsdp_file.ShardingStrategy.NO_SHARD
            ):
                tensor_state[state_name] = value
                continue
            if not value.is_cuda:
                value = value.to(fsdp_module.compute_device)
            # Assume that positive-dimension tensor optimizer state
            # has the same shape as the sharded flattened parameter
            buffer_size = flat_param._full_param_padded.size()  # type: ignore[attr-defined]
            tensor_buffer = value.new_zeros(*buffer_size)
            dist.all_gather_into_tensor(tensor_buffer, value, group=group)
            torch.cuda.synchronize()
            if to_save:
                unpadded_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
                tensor_state[state_name] = tensor_buffer[:unpadded_numel]
        # Zero-dimension tensor state and non-tensor state: take this rank's
        # value directly
        elif to_save:
            if _is_zero_dim_tensor(value):
                zero_dim_tensor_state[state_name] = value
            else:
                non_tensor_state[state_name] = value
    return state


def _unflatten_communicated_optim_state(
    fsdp_module,
    flat_param: FlatParameter,
    state: _ConsolidatedOptimState,
    shard_state: bool,
) -> List[Dict[str, Any]]:
    """
    Unflattens the communicated optimizer state (given by ``tensor_state``,
    ``non_tensor_state``, and ``zero_dim_tensor_state``) for a single flattened
    parameter ``flat_param``. This should only be called on the target rank.

    Args:
        flat_param (FlatParameter): The flattened parameter.
        state (_ConsolidatedOptimState): Consolidated optimizer state.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flattened parameter
        ``flat_param``. The final optimizer state dict will need to map these
        entries using the proper unflattened parameter IDs.
    """
    unflat_param_state: List[Dict[str, Any]] = []
    flat_param_views: Dict[str, Iterator] = {}
    num_unflat_params = flat_param._num_params
    tensor_state, zero_dim_tensor_state, non_tensor_state = (
        state.tensor_state,
        state.zero_dim_tensor_state,
        state.non_tensor_state,
    )

    for _ in range(num_unflat_params):
        unflat_state_param = {}
        # Add positive-dimension tensor state: unflatten with views
        for state_name, flat_tensor in sorted_items(tensor_state):
            views_generated = state_name in flat_param_views
            if not views_generated:
                views = FlatParamHandle._get_unflat_views(flat_param, flat_tensor)
                flat_param_views[state_name] = views
            else:
                views = flat_param_views[state_name]
            optim_state: Union[torch.Tensor, ShardedTensor] = next(views)
            if shard_state:
                optim_state = _ext_chunk_tensor(
                    optim_state,
                    fsdp_module.rank,
                    fsdp_module.world_size,
                    torch.cuda.device_count(),
                    fsdp_module.process_group,
                )
            unflat_state_param[state_name] = optim_state

        # Add zero-dimension tensor state: take the target rank's value
        for state_name, zero_dim_tensor in sorted_items(zero_dim_tensor_state):
            unflat_state_param[state_name] = zero_dim_tensor
        # Add non-tensor state: take the target rank's value
        for state_name, non_tensor in sorted_items(non_tensor_state):
            unflat_state_param[state_name] = non_tensor
        unflat_param_state.append(unflat_state_param)
    return unflat_param_state


def _flatten_optim_state_dict(
    optim_state_dict: Dict[str, Any],
    model: torch.nn.Module,
    shard_state: bool,
) -> Dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened
    parameter names. If ``shard_state=True``, then FSDP-managed
    ``FlatParameter`` 's optimizer states are sharded, and otherwise, they are
    kept unsharded.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    unflat_osd = optim_state_dict
    if "state" not in unflat_osd or "param_groups" not in unflat_osd:
        raise ValueError(
            '`optim_state_dict` must have the keys "state" and '
            '"param_groups" to be a valid optimizer state dict'
        )
    flat_param_to_fsdp_module = _get_flat_param_to_fsdp_module(model)
    param_to_unflat_param_names = fsdp_file._get_param_to_unflat_param_names(model)

    # Construct the "state" part
    flat_osd_state: Dict[_OptimStateKey, Any] = {}
    unflat_osd_state = unflat_osd["state"]
    for param, unflat_param_names in param_to_unflat_param_names.items():
        if isinstance(param, FlatParameter):  # flatten FSDP parameters' states
            assert (
                param in flat_param_to_fsdp_module
            ), f"Check the `flat_param_to_fsdp_module` construction\nparam: {param}"
            fsdp_module = flat_param_to_fsdp_module[param]
            flat_state = _flatten_optim_state(
                unflat_osd_state,
                unflat_param_names,
                fsdp_module,
                param,
                shard_state,
            )
            key = _OptimStateKey(tuple(unflat_param_names), True)
            flat_osd_state[key] = flat_state
        else:  # do not flatten non-FSDP parameters' states
            assert len(unflat_param_names) == 1
            unflat_param_name = unflat_param_names[0]
            if unflat_param_name not in unflat_osd_state:
                # The state dict may not have an entry for a parameter if it
                # was not passed into the optimizer (e.g. if it is not an
                # FSDP-managed parameter)
                continue
            key = _OptimStateKey(tuple(unflat_param_names), False)
            flat_osd_state[key] = copy.copy(unflat_osd_state[unflat_param_name])

    # Construct the "param_groups" part -- copy as is since it will be
    # rekeyed later according to the target rank's optimizer
    flat_osd_param_groups = copy.deepcopy(unflat_osd["param_groups"])
    return {"state": flat_osd_state, "param_groups": flat_osd_param_groups}


def _flatten_optim_state(
    unflat_osd_state: Dict[str, Dict[str, Any]],
    unflat_param_names: List[str],
    fsdp_module,
    flat_param: FlatParameter,
    shard_state: bool,
) -> Dict[str, Any]:
    """
    Flattens the optimizer state in ``full_optim_state_dict`` for a single
    flattened parameter ``flat_param`` in ``fsdp_module`` corresponding to
    the unflattened parameter names in ``unflat_param_names``.

    Args:
        unflat_osd_state (Dict[str, Dict[str, Any]]): The "state" part of the
            optimizer state dict corresponding to the unflattened parameters.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the flattened parameter
            ``flat_param``.
        fsdp_module (FullyShardedDataParallel): FSDP module owning the
            flattened parameter.
        flat_param (FlatParameter): The flattened parameter.
        shard_state (bool): Whether to shard flattened positive-dimension
            tensor state; if ``False``, then the full flattened tensor is
            kept in the returned :class:`dict.

    Returns:
        Dict[str, Any]: A :class:`dict` mapping state names to their values for
        a particular flattened parameter. The sharded optimizer state dict's
        "state" part will map a key to this returned value.
    """
    num_unflat_params = len(unflat_param_names)
    assert num_unflat_params > 0, (
        "Expects at least one unflattened parameter corresponding to the "
        "flattened parameter"
    )
    unflat_param_shapes = flat_param._shapes
    num_unflat_param_shapes = len(unflat_param_shapes)
    assert (
        num_unflat_params == num_unflat_param_shapes
    ), f"Expects {num_unflat_params} shapes but got {num_unflat_param_shapes}"

    # Check if these unflattened parameters have any optimizer state
    has_state = [
        bool(unflat_param_name in unflat_osd_state)
        for unflat_param_name in unflat_param_names
    ]
    # If none of the unflattened parameters comprising this flattened parameter
    # have any state, then we do not want an entry in the optimizer state dict
    if not any(has_state):
        return {}  # no need to flatten any state
    # There may still be some unflattened parameters with state and some
    # without
    unflat_param_states = [
        _gather_state_dict(
            unflat_osd_state[unflat_param_name], pg=fsdp_module.process_group
        )
        if unflat_param_name in unflat_osd_state
        else None
        for unflat_param_name in unflat_param_names
    ]
    # Check that the unflattened parameters have the same state names
    state_names = None
    for unflat_param_state in unflat_param_states:
        if unflat_param_state is None:
            continue
        if state_names is None:
            state_names = set(unflat_param_state.keys())
        else:
            if state_names != set(unflat_param_state.keys()):
                raise ValueError(
                    "Differing optimizer state names for the unflattened "
                    f"parameters: {unflat_param_names}"
                )
    assert state_names is not None

    # Flatten the state
    flat_state: Dict[str, Any] = {}
    for state_name in state_names:
        state_values = [
            unflat_param_state[state_name] if unflat_param_state is not None else None
            for unflat_param_state in unflat_param_states
        ]
        non_none_state_values = [v for v in state_values if v is not None]
        are_pos_dim_tensors = are_zero_dim_tensors = are_non_tensors = True
        for v in non_none_state_values:
            are_pos_dim_tensors &= torch.is_tensor(v) and v.dim() > 0
            are_zero_dim_tensors &= _is_zero_dim_tensor(v)
            are_non_tensors &= not torch.is_tensor(v)
        types = set(type(v) for v in non_none_state_values)
        if len(types) != 1 or not (
            are_pos_dim_tensors or are_zero_dim_tensors or are_non_tensors
        ):
            raise ValueError(
                f"Differing optimizer state types for state {state_name}, "
                f"values {non_none_state_values}, and unflattened parameter "
                f"names {unflat_param_names}"
            )
        if are_pos_dim_tensors:
            flat_tensor = _flatten_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
                unflat_param_shapes,
                flat_param,
            )
            if shard_state:
                # Shard the flattened tensor immediately to minimize max memory
                # usage
                sharded_flat_tensor, _ = FlatParamHandle._get_shard(
                    flat_tensor,
                    fsdp_module.rank,
                    fsdp_module.world_size,
                )
                flat_state[state_name] = sharded_flat_tensor
            else:
                flat_state[state_name] = flat_tensor
        elif are_zero_dim_tensors:
            flat_state[state_name] = _flatten_zero_dim_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
            )
        else:
            assert are_non_tensors
            flat_state[state_name] = _flatten_non_tensor_optim_state(
                state_name,
                state_values,
                unflat_param_names,
            )

    return flat_state


def _flatten_tensor_optim_state(
    state_name: str,
    pos_dim_tensors: List[torch.Tensor],
    unflat_param_names: List[str],
    unflat_param_shapes: Sequence[torch.Size],
    flat_param: FlatParameter,
) -> torch.Tensor:
    """
    Flattens the positive-dimension tensor optimizer state given by the values
    ``tensors`` for the state ``state_name`` for a single flattened parameter
    ``flat_param`` corresponding to the unflattened parameter names
    ``unflat_param_names`` and unflatted parameter shapes
    ``unflat_param_shapes``. This flattens each unflattened parameter's tensor
    state into one tensor.

    NOTE: We use zero tensors for any unflattened parameters without state
    since some value is required to fill those entries. This assumes that the
    zero tensor is mathematically equivalent to having no state, which is true
    for Adam's "exp_avg" and "exp_avg_sq" but may not be true for all
    optimizers.

    Args:
        state_name (str): Optimizer state name.
        pos_dim_tensors (List[torch.Tensor]): Positive-dimension tensor
            optimizer state values for the unflattened parameters corresponding
            to the single flattened parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flattened parameter.
        unflat_param_shapes (List[torch.Size]): Unflattened parameter shapes
            corresponding to the single flattened parameter.
        flat_param (FlatParameter): The flattened parameter.

    Returns:
        torch.Tensor: A flattened tensor containing the optimizer state
        corresponding to ``state_name`` constructed by concatenating the
        unflattened parameter tensor states in ``pos_dim_tensors`` (using zero
        tensors for any unflattened parameters without the state).
    """
    non_none_tensors = [t for t in pos_dim_tensors if t is not None]
    # Check that all are tensors with the same dtype
    dtypes = set(t.dtype for t in non_none_tensors)
    if len(dtypes) != 1:
        raise ValueError(
            "All unflattened parameters comprising a single flattened "
            "parameter must have positive-dimension tensor state with the "
            f"same dtype but got dtypes {dtypes} for state {state_name} and "
            f"unflattened parameter names {unflat_param_names}"
        )
    dtype = next(iter(dtypes))
    # Check that each tensor state matches its parameter's shape
    for tensor, shape in zip(pos_dim_tensors, unflat_param_shapes):
        if tensor is None and len(shape) == 0:
            raise ValueError("Flattening a zero-dimension parameter is not supported")
        elif tensor is not None and tensor.shape != shape:
            raise ValueError(
                "Tensor optimizer state does not have same shape as its "
                f"parameter: {tensor.shape} {shape}"
            )
    # Flatten the tensor states: we do not need to add any padding since the
    # flattened optimizer state tensor sharded via `_get_shard()`, which pads
    # the shard as needed (just like for the flattened parameter)
    cpu_device = torch.device("cpu")
    tensors = [
        torch.flatten(state_value.to(cpu_device))
        if state_value is not None
        else torch.flatten(
            torch.zeros(
                size=shape,
                dtype=dtype,
                device=cpu_device,
            )
        )
        for state_value, shape in zip(pos_dim_tensors, unflat_param_shapes)
    ]
    flat_tensor = torch.cat(tensors)
    flat_param_shape = flat_param._unpadded_unsharded_size  # type: ignore[attr-defined]
    assert flat_tensor.shape == flat_param_shape, (
        f"tensor optim state: {flat_tensor.shape} "
        f"flattened parameter: {flat_param_shape}"
    )
    return flat_tensor


def _flatten_zero_dim_tensor_optim_state(
    state_name: str,
    zero_dim_tensors: List[torch.Tensor],
    unflat_param_names: List[str],
) -> torch.Tensor:
    """
    Flattens the zero-dimension tensor optimizer state given by the values
    ``zero_dim_tensors`` for the state ``state_name`` for a single flattened
    parameter corresponding to the unflattened parameter names
    ``unflat_param_names`` by enforcing that all tensors are the same and using
    that common value.

    NOTE: The requirement that the tensors are the same across all unflattened
    parameters comprising the flattened parameter is needed to maintain the
    invariant that FSDP performs the same computation as its non-sharded
    equivalent. This means that none of the unflattened parameters can be
    missing this state since imposing a value may differ from having no value.
    For example, for Adam's "step", no value means maximum bias correction,
    while having some positive value means less bias correction.

    Args:
        state_name (str): Optimizer state name.
        zero_dim_tensors (List[torch.Tensor]): Zero-dimension optimizer state
            for the unflattened parameters corresponding to the single
            flattened parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flattened parameter.

    Returns:
        torch.Tensor: A zero-dimensional tensor giving the value of the state
        ``state_name`` for all unflattened parameters corresponding to the
        names ``unflat_param_names``.
    """
    non_none_tensors = [t for t in zero_dim_tensors if t is not None]
    # Enforce that all have the same value and dtype
    values_set = set(t.item() if t is not None else None for t in zero_dim_tensors)
    dtypes = set(t.dtype if t is not None else None for t in zero_dim_tensors)
    if (
        len(non_none_tensors) != len(zero_dim_tensors)
        or len(values_set) != 1
        or len(dtypes) != 1
    ):
        raise ValueError(
            "All unflattened parameters comprising a single flattened "
            "parameter must have scalar state with the same value and dtype "
            f"but got values {values_set} and dtypes {dtypes} for state "
            f"{state_name} and unflattened parameter names "
            f"{unflat_param_names}"
        )
    value = next(iter(values_set))
    dtype = next(iter(dtypes))
    return torch.tensor(value, dtype=dtype, device=torch.device("cpu"))


def _flatten_non_tensor_optim_state(
    state_name: str,
    non_tensors: List[Any],
    unflat_param_names: List[str],
) -> Any:
    """
    Flattens the non-tensor optimizer state given by the values ``non_tensors``
    for the state ``state_name`` for a single flattened parameter corresponding
    to the unflattened parameter names ``unflat_param_names`` by enforcing that
    all values are the same and using that common value.

    See the note in :func:`_flatten_zero_dim_tensor_optim_state`.

    Args:
        state_name (str): Optimizer state name.
        non_tensors (List[Any]): Non-tensor optimizer state for the unflattened
            parameters corresponding to the single flattened parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flattened parameter.

    Returns:
        Any: A non-tensor giving the value of the state ``state_name`` for all
        unflattened parameters corresponding to the names
        ``unflat_param_names``.
    """
    non_none_non_tensors = [nt for nt in non_tensors if nt is not None]
    # Enforce that all have the same value (same type already checked)
    non_tensor_set = set(non_tensors)
    if len(non_none_non_tensors) != len(non_tensors) or len(non_tensor_set) != 1:
        raise ValueError(
            "All unflattened parameters comprising a single flattened "
            "parameter must have scalar state with the same value and dtype "
            f"but got values {non_tensor_set} for state {state_name} and  "
            f"unflattened parameter names {unflat_param_names}"
        )
    non_tensor = next(iter(non_tensor_set))
    return non_tensor


def _process_pos_dim_tensor_state(
    flat_optim_state_dict: Dict[str, Any],
    world_size: int,
) -> Dict[str, Any]:
    """
    Processes positive-dimension tensor states in ``flat_optim_state_dict`` by
    replacing them with metadata. This is done so the processed optimizer state
    dict can be broadcast from rank 0 to all ranks without copying those tensor
    states, and thus, this is meant to only be called on rank 0.

    Args:
        flat_optim_state_dict (Dict[str, Any]): Flattened optimizer state dict
            with the positive-dimension tensor states unsharded.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict with positive-
        dimension tensor states replaced by metadata.
    """
    flat_osd = flat_optim_state_dict  # alias
    no_tensor_osd: Dict[str, Any] = {"state": {}}
    for key, param_state in flat_osd["state"].items():
        no_tensor_osd["state"][key] = {}
        for state_name, value in sorted_items(param_state):
            is_pos_dim_tensor_state = torch.is_tensor(value) and value.dim() > 0
            if not is_pos_dim_tensor_state:
                no_tensor_osd["state"][key][state_name] = value
                continue
            if key.is_flat_param:  # FSDP parameter
                sharded_size = FlatParamHandle._get_sharded_size(
                    value, rank=0, world_size=world_size
                )
                assert len(sharded_size) == 1, f"{sharded_size}"
                info = _PosDimTensorInfo(sharded_size, value.dtype)
            else:  # non-FSDP parameter
                info = _PosDimTensorInfo(value.shape, value.dtype)
            no_tensor_osd["state"][key][state_name] = info
    no_tensor_osd["param_groups"] = flat_osd["param_groups"]
    return no_tensor_osd


def _broadcast_processed_optim_state_dict(
    processed_optim_state_dict: Optional[Dict[str, Any]],
    rank: int,
    group,
) -> Dict[str, Any]:
    """
    Broadcasts the processed optimizer state dict from rank 0 to all ranks.

    Args:
        processed_optim_state_dict (Optional[Dict[str, Any]]): The flattened
            optimizer state dict with positive-dimension tensor states replaced
            with metadata if on rank 0; ignored otherwise.

    Returns:
        Dict[str, Any]: The processed optimizer state dict.
    """
    # Broadcast the two data structures rank 0 to all ranks
    obj_list = [processed_optim_state_dict] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0, group=group)
    processed_optim_state_dict = obj_list[0]  # type: ignore[assignment]
    assert processed_optim_state_dict is not None
    # Keep zero-dimension tensors on CPU
    return processed_optim_state_dict


def _broadcast_pos_dim_tensor_states(
    processed_optim_state_dict: Dict[str, Any],
    flat_optim_state_dict: Optional[Dict[str, Any]],
    rank: int,
    world_size: int,
    group,
    broadcast_device: torch.device,
) -> Dict[str, Any]:
    """
    Takes ``processed_optim_state_dict``, which has metadata in place of
    positive-dimension tensor states, and broadcasts those tensor states from
    rank 0 to all ranks. For tensor states corresponding to FSDP parameters,
    rank 0 shards the tensor and broadcasts shard-by-shard, and for tensor
    states corresponding to non-FSDP parameters, rank 0 broadcasts the full
    tensor.

    Args:
        processed_optim_state_dict (Dict[str, Any]): The flattened optimizer
            state dict with positive-dimension tensor states replaced with
            metadata; this should be returned by
            :meth:`_process_pos_dim_tensor_state` and non-empty on all ranks.
        flat_optim_state_dict (Optional[Dict[str, Any]]): The flattened
            unsharded optimizer state dict with the actual positive-dimension
            tensor states if on rank 0; ignored on nonzero ranks.

    Returns:
        Dict[str, Any]: The optimizer state dict with the positive-dimension
        tensor state correctly populated via ``broadcast()`` s from rank 0.
    """
    assert (
        rank != 0 or flat_optim_state_dict is not None
    ), "Expects rank 0 to pass in the flattened optimizer state dict"
    no_tensor_osd = processed_optim_state_dict  # alias
    flat_osd = flat_optim_state_dict  # alias
    for key, param_state in no_tensor_osd["state"].items():
        for state_name, value in sorted_items(param_state):
            is_pos_dim_tensor_state = isinstance(value, _PosDimTensorInfo)
            if not is_pos_dim_tensor_state:
                continue
            if rank == 0:
                assert flat_osd is not None
                unsharded_tensor = flat_osd["state"][key][state_name]
            else:
                unsharded_tensor = None
            shape, dtype = value.shape, value.dtype
            if key.is_flat_param:  # FSDP parameter
                _broadcast_sharded_pos_dim_tensor_state(
                    unsharded_tensor,
                    param_state,
                    state_name,
                    shape,
                    dtype,
                    broadcast_device,
                    rank,
                    world_size,
                    group,
                )  # modify `param_state` destructively
            else:  # non-FSDP parameter
                _broadcast_unsharded_pos_dim_tensor_state(
                    unsharded_tensor,
                    param_state,
                    state_name,
                    shape,
                    dtype,
                    broadcast_device,
                    rank,
                    group,
                )  # modify `param_state` destructively
    return no_tensor_osd


def _broadcast_sharded_pos_dim_tensor_state(
    unsharded_tensor: Optional[torch.Tensor],
    param_state: Dict[str, Any],
    state_name: str,
    shape: torch.Size,
    dtype: torch.dtype,
    broadcast_device: torch.device,
    rank: int,
    world_size: int,
    group,
) -> None:
    """
    Broadcasts positive-dimension tensor state for the state ``state_name``
    corresponding to an FSDP parameter shard-by-shard, only to be saved on the
    relevant rank. This modifies ``param_state`` destructively.

    Args:
        unsharded_tensor (Optional[torch.Tensor]): Unsharded tensor from which
            to broadcast shards if on rank 0; ignored otherwise.
        shape (torch.Size): Shape of the sharded tensor; same on all ranks.
    """
    get_shard: Optional[functools.partial[Tuple[torch.Tensor, int]]] = None
    if rank == 0:
        assert (
            unsharded_tensor is not None
        ), "Expects rank 0 to pass in the unsharded tensor"
        get_shard = functools.partial(
            FlatParamHandle._get_shard,
            unsharded_tensor,
        )
    for target_rank in range(1, world_size):
        if rank == 0:
            assert get_shard is not None
            sharded_tensor = get_shard(target_rank, world_size)[0].to(broadcast_device)
        else:
            sharded_tensor = torch.zeros(
                shape,
                requires_grad=False,
                dtype=dtype,
                device=broadcast_device,
            )
        dist.broadcast(sharded_tensor, src=0, group=group)
        # Only keep the shard on the target rank and keep it on the broadcast
        # device, which is typically GPU
        if rank == target_rank:
            param_state[state_name] = sharded_tensor
        else:
            del sharded_tensor
    # Lastly, shard on rank 0
    if rank != 0:
        return
    param_state[state_name] = get_shard(0, world_size)[0].to(broadcast_device)  # type: ignore[misc]


def _broadcast_unsharded_pos_dim_tensor_state(
    unsharded_tensor: Optional[torch.Tensor],
    param_state: Dict[str, Any],
    state_name: str,
    shape: torch.Size,
    dtype: torch.dtype,
    broadcast_device: torch.device,
    rank: int,
    group,
) -> None:
    """
    Broadcasts positive-dimension tensor state for the state ``state_name``
    corresponding to an unsharded non-FSDP parameter from rank 0 to all ranks.
    This modifies ``param_state`` destructively.

    Args:
        unsharded_tensor (Optional[torch.Tensor]): Unsharded tensor to
            broadcast if on rank 0; ignored otherwise.
    """
    if rank == 0:
        assert (
            unsharded_tensor is not None
        ), "Expects rank 0 to pass in the unsharded tensor"
        assert (
            shape == unsharded_tensor.shape
        ), f"Shape mismatch: {shape} {unsharded_tensor.shape}"
        assert (
            dtype == unsharded_tensor.dtype
        ), f"dtype mismatch: {dtype} {unsharded_tensor.dtype}"
        unsharded_tensor = unsharded_tensor.to(broadcast_device)
    else:
        unsharded_tensor = torch.zeros(
            shape,
            requires_grad=False,
            dtype=dtype,
            device=broadcast_device,
        )
    dist.broadcast(unsharded_tensor, src=0, group=group)
    # Keep the tensor on the broadcast device, which is typically GPU
    param_state[state_name] = unsharded_tensor


def _rekey_sharded_optim_state_dict(
    sharded_osd: Dict[str, Any],
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ],
    using_optim_input: bool,
) -> Dict[str, Any]:
    """
    Rekeys the optimizer state dict from unflattened parameter names to
    flattened parameter IDs according to the calling rank's ``optim``, which
    may be different across ranks. In particular, the unflattened parameter
    names are represented as :class:`_OptimStateKey` s.
    """
    param_to_flat_param_id = (
        _get_param_to_param_id_from_optim_input(model, optim_input)
        if using_optim_input
        else _get_param_to_param_id(optim)
    )
    param_to_unflat_param_names = fsdp_file._get_param_to_unflat_param_names(model)
    # All parameter keys in `param_to_flat_param_id` should be in
    # `param_to_unflat_param_names` -- strict inequality follows when not all
    # parameters are passed to the optimizer
    assert len(param_to_flat_param_id) <= len(param_to_unflat_param_names)

    unflat_param_names_to_flat_param_id: Dict[Tuple[str, ...], int] = {}  # for "state"
    unflat_param_name_to_flat_param_id: Dict[str, int] = {}  # for "param_groups"
    for param, unflat_param_names in param_to_unflat_param_names.items():
        if param not in param_to_flat_param_id:
            # This parameter was not passed to the optimizer
            continue
        flat_param_id = param_to_flat_param_id[param]
        unflat_param_names_to_flat_param_id[tuple(unflat_param_names)] = flat_param_id
        for unflat_param_name in unflat_param_names:
            unflat_param_name_to_flat_param_id[unflat_param_name] = flat_param_id

    sharded_osd_state = sharded_osd["state"]
    rekeyed_osd_state = {}
    for key, param_state in sharded_osd_state.items():
        flat_param_id = unflat_param_names_to_flat_param_id[key.unflat_param_names]
        rekeyed_osd_state[flat_param_id] = param_state

    rekeyed_osd_param_groups: List[Dict[str, Any]] = []
    for unflat_param_group in sharded_osd["param_groups"]:
        flat_param_group = copy.deepcopy(unflat_param_group)
        flat_param_ids = sorted(
            set(
                unflat_param_name_to_flat_param_id[unflat_param_name]
                for unflat_param_name in unflat_param_group["params"]
            )
        )
        flat_param_group["params"] = flat_param_ids
        rekeyed_osd_param_groups.append(flat_param_group)

    return {"state": rekeyed_osd_state, "param_groups": rekeyed_osd_param_groups}


def _get_flat_param_to_fsdp_module(model: torch.nn.Module):
    """
    Constructs a mapping from FSDP flattened parameters to their owning FSDP
    modules and ensures that all FSDP modules are initialized.

    Args:
        model (torch.nn.model): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).

    Returns:
        Dict[FlatParameter, FullyShardedDataParallel]: Mapping from FSDP
            flattened parameters to their owning FSDP modules.
    """
    flat_param_to_fsdp_module = {}
    for module in model.modules():
        if isinstance(module, fsdp_file.FullyShardedDataParallel):
            fsdp_file._lazy_init(module, module)
            for param in module.params:  # may have none
                flat_param_to_fsdp_module[param] = module
    return flat_param_to_fsdp_module


def _get_param_id_to_param(
    optim: torch.optim.Optimizer,
):
    """
    Constructs a mapping from parameter IDs to parameters. This may be used
    both for models with ``FlatParameter`` s and without.
    """
    param_id_to_param: List[nn.Parameter] = []
    for param_group in optim.param_groups:
        for param in param_group["params"]:
            param_id_to_param.append(param)
    return param_id_to_param


def _get_param_id_to_param_from_optim_input(
    model: torch.nn.Module,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
) -> List[torch.nn.Parameter]:
    """
    Constructs a mapping from parameter IDs to parameters. This may be used
    both for models with ``FlatParameter`` s and without.

    NOTE: This method is only preserved for backward compatibility. The method
    :meth:`_get_param_id_to_param` is the preferred code path that does not
    rely on ``optim_input``.

    NOTE: We critically assume that, whether the optimizer input is a list of
    parameters or a list of parameter groups, :class:`torch.optim.Optimizer`
    enumerates the parameter IDs in order. In other words, for a parameter list
    input, the parameter IDs should be in that list order, and for a parameter
    groups input, the parameter IDs should be in order within each parameter
    group and in order across parameter groups.

    Args:
        model (torch.nn.Module): Model whose parameters are passed into the
            optimizer.
        optim_input (Optional[Union[List[Dict[str, Any]],
        Iterable[torch.nn.Parameter]]]): Input passed into the optimizer
            representing either a :class:`list` of parameter groups or an
            iterable of parameters; if ``None``, then this method assumes the
            input was ``model.parameters()``. (Default: ``None``)

    Returns:
        List[torch.nn.Parameter]: Mapping from parameter IDs to parameters,
        where the parameter ID is implicitly the index in the :class:`list`.
    """
    # Assume the standard case of passing `model.parameters()` to the optimizer
    # if `optim_input` is not specified
    if optim_input is None:
        return list(model.parameters())
    try:
        params = list(optim_input)
    except TypeError:
        raise TypeError(
            "Optimizer input should be an iterable of Tensors or dicts, "
            f"but got {optim_input}"
        )
    if len(params) == 0:
        raise ValueError("Optimizer input should not be empty")

    # Check if the optimizer input represents tensors or parameter groups
    all_tensors = True
    all_dicts = True
    for param in params:
        all_tensors &= isinstance(param, torch.Tensor)
        all_dicts &= isinstance(param, dict)
    if not all_tensors and not all_dicts:
        raise TypeError("Optimizer input should be an iterable of Tensors or dicts")
    if all_tensors:
        return params  # type: ignore[return-value]
    assert all_dicts
    param_id_to_param = []
    for param_group in params:
        has_params_key = "params" in param_group  # type: ignore[operator]
        assert has_params_key, (
            'A parameter group should map "params" to a list of the '
            "parameters in the group"
        )
        for param in param_group["params"]:  # type: ignore[index]
            # Implicitly map `flat_param_id` (current length of the list) to
            # `param`
            param_id_to_param.append(param)
    return param_id_to_param  # type: ignore[return-value]


def _get_param_to_param_id(
    optim: torch.optim.Optimizer,
) -> Dict[torch.nn.Parameter, int]:
    """Constructs the inverse mapping of :func:`_get_param_id_to_param`."""
    param_id_to_param = _get_param_id_to_param(optim)
    return {param: param_id for param_id, param in enumerate(param_id_to_param)}


def _get_param_to_param_id_from_optim_input(
    model: torch.nn.Module,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
) -> Dict[torch.nn.Parameter, int]:
    """Constructs the inverse mapping of :func:`_get_param_id_to_param`."""
    param_id_to_param = _get_param_id_to_param_from_optim_input(model, optim_input)
    return {param: param_id for param_id, param in enumerate(param_id_to_param)}


def _get_unflat_to_flat_param_ids(
    flat_to_unflat_param_ids: Dict[int, List[int]],
) -> List[int]:
    """
    Inverts the mapping ``flat_to_unflat_param_ids`` to be from unflattened
    parameter ID to flattened parameter ID, where the unflattened parameter ID
    is the index in the returned :class:`list`. There may be multiple
    unflattened parameter IDs mapping to the same flattened parameter ID.

    Args:
        flat_to_unflat_param_ids (Dict[int, List[int]]): A mapping from
            flattened parameter ID to a :class:`list` of corresponding
            unflattened parameter IDs.

    Returns:
        List[int]: A mapping from unflattened parameter ID to flattened
        parameter ID, where the unflattened parameter ID is the index in the
        :class:`list`.
    """
    # Construct as a dict and then convert to list
    unflat_to_flat_param_ids = {}
    for flat_param_id, unflat_param_ids in flat_to_unflat_param_ids.items():
        for unflat_param_id in unflat_param_ids:
            assert unflat_param_id not in unflat_to_flat_param_ids, (
                "`flat_to_unflat_param_ids` has the unflattened parameter "
                f"ID {unflat_param_id} mapped to multiple flattened "
                "parameter IDs"
            )
            unflat_to_flat_param_ids[unflat_param_id] = flat_param_id
    num_unflat_param_ids = len(unflat_to_flat_param_ids)
    unflat_param_ids_set = set(unflat_to_flat_param_ids.keys())
    assert unflat_param_ids_set == set(range(num_unflat_param_ids)), (
        "The set of unflattened parameter IDs should be {0, ..., "
        + str(num_unflat_param_ids - 1)
        + "} but got "
        + f"{unflat_param_ids_set}"
    )
    return [
        unflat_to_flat_param_ids[unflat_param_id]
        for unflat_param_id in range(num_unflat_param_ids)
    ]


def _is_zero_dim_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dim() == 0


def _optim_state_dict(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ],
    rank0_only: bool,
    shard_state: bool,
    group: Optional[dist.ProcessGroup],
    using_optim_input: bool,
) -> Dict[str, Any]:
    """
    Consolidates the optimizer state and returns it as a :class:`dict`
    following the convention of :meth:`torch.optim.Optimizer.state_dict`,
    i.e. with keys ``"state"`` and ``"param_groups"``.
    The flattened parameters in ``FSDP`` modules contained in ``model``
    are mapped back to their unflattened parameters.

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` 's
            parameters.
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        shard_state (bool): If ``True``, shard and distribute all
            non-zero-dimension states.

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state for
        ``model`` 's original unflattened parameters and including keys
        "state" and "param_groups" following the convention of
        :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=False``,
        then nonzero ranks return an empty :class:`dict`.
    """
    osd = optim.state_dict()
    osd_state, osd_param_groups = osd["state"], osd["param_groups"]
    rank = dist.get_rank(group)
    to_save = not rank0_only or (rank == 0 or shard_state)
    fsdp_osd: Dict = {"state": {}, "param_groups": []} if to_save else {}
    fsdp_osd_state = fsdp_osd["state"] if to_save else None

    # Construct the local mapping between unflattened parameter names
    # (`_OptimStateKey`s) and parameter IDs and broadcast rank 0's mapping
    param_to_unflat_param_names: Dict[
        torch.nn.Parameter, List[str]
    ] = fsdp_file._get_param_to_unflat_param_names(model)
    flat_param_id_to_param: List[torch.nn.Parameter] = (
        _get_param_id_to_param_from_optim_input(model, optim_input)
        if using_optim_input
        else _get_param_id_to_param(optim)
    )
    optim_state_key_to_flat_param_id: Dict[_OptimStateKey, int] = {}  # local
    r0_flat_param_id_to_optim_state_key: Dict[
        int, _OptimStateKey
    ] = collections.OrderedDict()  # rank 0
    for flat_param_id, param in enumerate(flat_param_id_to_param):
        # Do not include parameters without state to avoid empty mappings
        # just like in normal `torch.optim.Optimizer.state_dict()`
        if flat_param_id not in osd_state:
            continue
        optim_state_key = _OptimStateKey(
            unflat_param_names=tuple(param_to_unflat_param_names[param]),
            is_flat_param=isinstance(param, FlatParameter),
        )
        if rank == 0:
            r0_flat_param_id_to_optim_state_key[flat_param_id] = optim_state_key
        optim_state_key_to_flat_param_id[optim_state_key] = flat_param_id
    key_obj_list: List[Optional[Dict[int, _OptimStateKey]]] = (
        [r0_flat_param_id_to_optim_state_key] if rank == 0 else [None]
    )
    dist.broadcast_object_list(key_obj_list, src=0, group=group)
    assert key_obj_list[0] is not None
    r0_flat_param_id_to_optim_state_key = key_obj_list[0]

    # Ensure that all ranks have at least the optimizer states needed by
    # rank 0's optimizer
    missing_keys: List[_OptimStateKey] = []
    for r0_optim_state_key in r0_flat_param_id_to_optim_state_key.values():
        if r0_optim_state_key not in optim_state_key_to_flat_param_id:
            # A parameter from rank 0's optimizer does not exist for this
            # rank's optimizer
            missing_keys.append(r0_optim_state_key)
            continue
        flat_param_id = optim_state_key_to_flat_param_id[r0_optim_state_key]
        assert flat_param_id >= 0 and flat_param_id < len(
            flat_param_id_to_param
        ), "Check the `flat_param_id_to_param` construction"
    device = torch.device("cuda", torch.cuda.current_device())
    num_missing = torch.tensor([len(missing_keys)], dtype=torch.int32, device=device)
    dist.all_reduce(num_missing, group=group)
    if num_missing.item() > 0:
        obj_list = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(obj_list, missing_keys, group=group)
        error_msg = (
            "FSDP currently requires each rank to have at least the "
            "optimizer states needed by rank 0's optimizer but some ranks "
            "are missing some of those states"
        )
        for rank, keys in enumerate(obj_list):
            keys = cast(List[_OptimStateKey], keys)
            if len(keys) > 0:
                error_msg += (
                    f"\nRank {rank} is missing states for the parameters: "
                    f"{[key.unflat_param_names for key in keys]}"
                )
        raise RuntimeError(error_msg)

    # Iterate in rank 0's flattened parameter ID order to ensure aligned
    # all-gathers across ranks
    flat_param_to_fsdp_module = _get_flat_param_to_fsdp_module(model)
    for r0_optim_state_key in r0_flat_param_id_to_optim_state_key.values():
        flat_param_id = optim_state_key_to_flat_param_id[r0_optim_state_key]
        param = flat_param_id_to_param[flat_param_id]
        if r0_optim_state_key.is_flat_param:
            fsdp_module = flat_param_to_fsdp_module[param]
            unflat_state = _unflatten_optim_state(
                cast(FlatParameter, param),
                osd_state[flat_param_id],
                fsdp_module,
                to_save,
                shard_state,
            )
            if to_save:
                assert len(unflat_state) == len(r0_optim_state_key.unflat_param_names)
                for unflat_param_name, unflat_param_state in zip(
                    r0_optim_state_key.unflat_param_names,
                    unflat_state,
                ):
                    fsdp_osd_state[unflat_param_name] = unflat_param_state
        elif to_save:
            assert len(r0_optim_state_key.unflat_param_names) == 1
            unflat_param_name = r0_optim_state_key.unflat_param_names[0]
            fsdp_osd_state[unflat_param_name] = copy.copy(osd_state[flat_param_id])
            for state_name, value in sorted_items(fsdp_osd_state[unflat_param_name]):
                if torch.is_tensor(value):
                    fsdp_osd_state[unflat_param_name][state_name] = value.cpu()

    if not to_save:
        return {}

    # Handle the "param_groups" part of the optimizer state dict
    fsdp_osd_param_groups = fsdp_osd["param_groups"]  # alias
    for flat_param_group in osd_param_groups:
        unflat_param_group = copy.deepcopy(flat_param_group)
        param_group_params = [
            flat_param_id_to_param[flat_param_id]
            for flat_param_id in flat_param_group["params"]
        ]
        nested_unflat_param_names = [
            param_to_unflat_param_names[param] for param in param_group_params
        ]
        unflat_param_group["params"] = [
            unflat_param_name
            for unflat_param_names in nested_unflat_param_names
            for unflat_param_name in unflat_param_names
        ]  # flatten the list of lists
        fsdp_osd_param_groups.append(unflat_param_group)
    return fsdp_osd
