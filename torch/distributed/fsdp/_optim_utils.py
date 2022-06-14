import copy
import functools
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
# Import the entire FSDP file to avoid circular imports
import torch.distributed.fsdp.fully_sharded_data_parallel as FSDP
from torch.distributed.fsdp.flatten_params_wrapper import FlatParameter


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


def _unflatten_optim_state(
    fsdp_module,
    flat_param: FlatParameter,
    flat_param_state: Dict[str, Any],
    to_save: bool,
) -> List[Dict[str, Any]]:
    """
    Unflattens the optimizer state, consisting of the "state" part and the
    "param_groups" part. Unflattening the "state" part involves consolidating
    the state on the target rank and remapping from flattened to unflattened
    parameter IDs, and the "param_groups" part only involves remapping from
    flattened to unflattened parameter IDs.

    Args:
        fsdp_module (FullyShardedDataParallel): FSDP module that owns
            ``flat_param``, i.e. holds it in ``self.params``.
        flat_param (FlatParameter): The flattened parameter.
        flat_param_state (Dict[str, Any]): Entry for the flattened parameter
            in the "state" part of the optimizer state dict.
        to_save (bool): Whether to save the state on this rank.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flattened parameter
        ``flat_param`` if on the target rank or an empty :class:`list`
        otherwise. The final optimizer state dict will need to map these
        entries using the proper unflattened parameter IDs.
    """
    assert sum(p is flat_param for p in fsdp_module.params) == 1, \
        "`fsdp_module` must own `flat_param`"
    consolidated_state = _communicate_optim_state(
        fsdp_module, flat_param, flat_param_state, to_save,
    )
    unflat_param_state = _unflatten_communicated_optim_state(
        fsdp_module,
        flat_param,
        consolidated_state,
    ) if to_save else []
    return unflat_param_state


def _communicate_optim_state(
    fsdp_module,
    flat_param: FlatParameter,
    flat_param_state: Dict[str, Any],
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
        to_save (bool): Whether to save the state on this rank.

    Returns:
        ConsolidatedOptimState: Consolidated optimizer state for
        ``flat_param``; the state is not populated for non-target ranks.
    """
    param_index = -1
    for i, param in enumerate(fsdp_module.params):
        if param is flat_param:
            param_index = i
            break
    assert param_index >= 0, "`fsdp_module` must own `flat_param`"

    state = _ConsolidatedOptimState()
    tensor_state, zero_dim_tensor_state, non_tensor_state = \
        state.tensor_state, state.zero_dim_tensor_state, state.non_tensor_state
    process_group = fsdp_module.process_group

    tensor_buffer = None  # initialize lazily in case it is not needed
    for state_name, value in flat_param_state.items():
        # Positive-dimension tensor state: communicate across ranks
        if torch.is_tensor(value) and value.dim() > 0:
            # If the parameter is not sharded (e.g. world size of 1), then
            # neither is the positive-dimension tensor state, so no need to
            # communicate it -- we take the target rank's value
            if not flat_param._is_sharded:
                tensor_state[state_name] = value.cpu()
                continue
            if tensor_buffer is None:
                # Assume that positive-dimension tensor optimizer state
                # has the same shape as the sharded flattened parameter
                buffer_size = flat_param._full_param_padded.size()  # type: ignore[attr-defined]
                tensor_buffer = value.new_zeros(*buffer_size)
            dist._all_gather_base(tensor_buffer, value, group=process_group)
            if to_save:
                assert hasattr(flat_param, "_orig_size"), \
                    "Sharded flattened parameter should have `_orig_size` set"
                unpadded_numel = flat_param._orig_size.numel()  # type: ignore[attr-defined]
                tensor_state[state_name] = tensor_buffer[:unpadded_numel].cpu()
        # Zero-dimension tensor state and non-tensor state: take this rank's
        # value directly
        elif to_save:
            if _is_zero_dim_tensor(value):
                zero_dim_tensor_state[state_name] = value.cpu()
            else:
                non_tensor_state[state_name] = value
    return state


def _unflatten_communicated_optim_state(
    fsdp_module,
    flat_param: FlatParameter,
    state: _ConsolidatedOptimState,
) -> List[Dict[str, Any]]:
    """
    Unflattens the communicated optimizer state (given by ``tensor_state``,
    ``non_tensor_state``, and ``zero_dim_tensor_state``) for a single flattened
    parameter ``flat_param``. This should only be called on the target rank.

    Args:
        fsdp_module (FullyShardedDataParallel): FSDP module that owns
            ``flat_param``, i.e. holds it in ``self.params``.
        flat_param (FlatParameter): The flattened parameter.
        state (_ConsolidatedOptimState): Consolidated optimizer state.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flattened parameter
        ``flat_param``. The final optimizer state dict will need to map these
        entries using the proper unflattened parameter IDs.
    """
    assert sum(p is flat_param for p in fsdp_module.params) == 1, \
        "`fsdp_module` must own `flat_param`"
    unflat_param_state: List[Dict[str, Any]] = []
    flat_param_views: Dict[str, Iterator] = {}
    num_unflat_params = flat_param._num_unflattened_params
    tensor_state, zero_dim_tensor_state, non_tensor_state = \
        state.tensor_state, state.zero_dim_tensor_state, state.non_tensor_state

    for _ in range(num_unflat_params):
        unflat_state_param = {}
        # Add positive-dimension tensor state: unflatten with views
        for state_name, flat_tensor in tensor_state.items():
            views_generated = state_name in flat_param_views
            if not views_generated:
                param_views = flat_param.get_param_views(flat_tensor)
                flat_param_views[state_name] = param_views
            else:
                param_views = flat_param_views[state_name]
            unflat_state_param[state_name] = next(param_views)
        # Add zero-dimension tensor state: take the target rank's value
        for state_name, zero_dim_tensor in zero_dim_tensor_state.items():
            unflat_state_param[state_name] = zero_dim_tensor
        # Add non-tensor state: take the target rank's value
        for state_name, non_tensor in non_tensor_state.items():
            unflat_state_param[state_name] = non_tensor
        unflat_param_state.append(unflat_state_param)
    return unflat_param_state


def _flatten_full_optim_state_dict(
    full_optim_state_dict: Dict[str, Any],
    model: torch.nn.Module,
    shard_state: bool,
    optim_input: Optional[Union[
        List[Dict[str, Any]], Iterable[torch.nn.Parameter],
    ]] = None,
) -> Tuple[Dict[str, Any], Set[int]]:
    """
    Args:
        shard_state (bool): Whether to shard flattened positive-dimension
            tensor state; if ``False``, then the full flattened tensor is
            kept in the returned :class:`dict.

    Returns:
        Tuple[Dict[str, Any], Set[int]]: The flattened optimizer state dict
            and a set of the parameter IDs corresponding to FSDP parameters.
    """
    full_osd = full_optim_state_dict  # alias
    if "state" not in full_osd or "param_groups" not in full_osd:
        raise ValueError(
            "`full_optim_state_dict` must have the keys \"state\" and "
            "\"param_groups\" to be a valid optimizer state dict"
        )

    flat_param_id_to_param = _get_param_id_to_param(model, optim_input)
    flat_param_to_fsdp_module = _get_flat_param_to_fsdp_module(model)
    param_to_unflat_param_names = FSDP._get_param_to_unflat_param_names(model)

    # Handle the "state" part of the optimizer state dict
    flat_osd_state: Dict[int, Any] = {}
    full_osd_state = full_osd["state"]
    unflat_param_names_to_flat_param_id: Dict[str, int] = {}
    fsdp_flat_param_ids = set()  # save which IDs are for FSDP parameters
    for flat_param_id, param in enumerate(flat_param_id_to_param):  # type: ignore[assignment]
        assert param in param_to_unflat_param_names, \
            "Check the `param_to_unflat_params` construction\n" \
            f"param: {param}"
        unflat_param_names = param_to_unflat_param_names[param]
        # For FSDP parameters, we need to flatten
        if isinstance(param, FlatParameter):
            assert param in flat_param_to_fsdp_module, \
                "Check the `flat_param_to_fsdp_module` mapping " \
                f"construction\nparam={param}"
            unflat_param_names = param_to_unflat_param_names[param]
            fsdp_module = flat_param_to_fsdp_module[param]
            flat_state = _flatten_optim_state(
                full_osd_state, unflat_param_names, fsdp_module, param,
                shard_state,
            )
            flat_osd_state[flat_param_id] = flat_state
            for unflat_param_name in unflat_param_names:
                unflat_param_names_to_flat_param_id[unflat_param_name] = flat_param_id
            fsdp_flat_param_ids.add(flat_param_id)
        # For parameters from non-FSDP modules, we do not need to flatten
        else:
            assert len(unflat_param_names) == 1
            unflat_param_name = unflat_param_names[0]
            if unflat_param_name not in full_osd_state:
                # A non-FSDP module's parameter may be ignored and hence not
                # have an entry in the optimizer state
                continue
            # Remap from unflattened to flattened parameter ID -- do not
            # deepcopy to avoid unnecessarily duplicating tensor storage
            flat_osd_state[flat_param_id] = \
                copy.copy(full_osd_state[unflat_param_name])
            unflat_param_names_to_flat_param_id[unflat_param_name] = flat_param_id

    # Handle the "param_groups" part of the optimizer state dict
    sharded_osd_param_groups: List[Dict[str, Any]] = []
    for unflat_param_group in full_osd["param_groups"]:
        flat_param_group = copy.deepcopy(unflat_param_group)
        # Map from unflattened parameter names to flattened parameter IDs
        flat_param_ids = sorted(set(
            unflat_param_names_to_flat_param_id[unflat_param_name]
            for unflat_param_name in unflat_param_group["params"]
        ))
        flat_param_group["params"] = flat_param_ids
        sharded_osd_param_groups.append(flat_param_group)

    optim_state_dict = {
        "state": flat_osd_state,
        "param_groups": sharded_osd_param_groups,
    }
    return optim_state_dict, fsdp_flat_param_ids


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
        "state" part will map the flattened parameter ID to this returned
        value.
    """
    num_unflat_params = len(unflat_param_names)
    assert num_unflat_params > 0, \
        "Expects at least one unflattened parameter corresponding to the " \
        "flattened parameter"
    unflat_param_shapes = flat_param._param_shapes
    num_unflat_param_shapes = len(unflat_param_shapes)
    assert num_unflat_params == num_unflat_param_shapes, \
        f"Expects {num_unflat_params} shapes but got {num_unflat_param_shapes}"

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
        unflat_osd_state[unflat_param_name]
        if unflat_param_name in unflat_osd_state else None
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
            unflat_param_state[state_name]
            if unflat_param_state is not None else None
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
                state_name, state_values, unflat_param_names,
                unflat_param_shapes, flat_param,
            )
            if shard_state:
                # Shard the flattened tensor immediately to minimize max memory
                # usage
                sharded_flat_tensor, _ = fsdp_module._get_shard(flat_tensor)
                flat_state[state_name] = sharded_flat_tensor
            else:
                flat_state[state_name] = flat_tensor
        elif are_zero_dim_tensors:
            flat_state[state_name] = _flatten_zero_dim_tensor_optim_state(
                state_name, state_values, unflat_param_names,
            )
        else:
            assert are_non_tensors
            flat_state[state_name] = _flatten_non_tensor_optim_state(
                state_name, state_values, unflat_param_names,
            )

    return flat_state


def _flatten_tensor_optim_state(
    state_name: str,
    pos_dim_tensors: List[torch.Tensor],
    unflat_param_names: List[str],
    unflat_param_shapes: List[torch.Size],
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
    for Adam's ``exp_avg`` and ``exp_avg_sq`` but may not be true for all
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
            raise ValueError(
                "Flattening a zero-dimension parameter is not supported"
            )
        elif tensor is not None and tensor.shape != shape:
            raise ValueError(
                "Tensor optimizer state does not have same shape as its "
                f"parameter: {tensor.shape} {shape}"
            )
    # Flatten the tensor states
    cpu_device = torch.device("cpu")
    tensors = [
        torch.flatten(state_value.to(cpu_device)) if state_value is not None
        else torch.flatten(torch.zeros(
            size=shape, dtype=dtype, device=cpu_device,
        ))
        for state_value, shape
        in zip(pos_dim_tensors, unflat_param_shapes)
    ]
    padding = flat_param.num_padded
    if padding > 0:
        tensors.append(torch.zeros(padding, dtype=dtype, device=cpu_device))
    flat_tensor = torch.cat(tensors)
    # `flat_tensor`'s shape should be 1D and less than or equal to the
    # flattened parameter's shape (where the inequality is strict for positive
    # padding)
    if not flat_param._is_sharded:  # currently, only when world size is 1
        # If the parameter is not sharded, then `_full_param_padded` is not
        # used, so we skip the shape check
        return flat_tensor
    full_padded_dim = flat_param._full_param_padded.dim()  # type: ignore[attr-defined]
    full_padded_shape = flat_param._full_param_padded.shape  # type: ignore[attr-defined]
    assert flat_tensor.dim() == 1, \
        f"`flat_tensor` should be 1D but got {flat_tensor.dim()} dims"
    assert full_padded_dim == 1, \
        f"`_full_param_padded` should be 1D but got {full_padded_dim} dims"
    assert flat_tensor.shape[0] <= full_padded_shape[0], \
        f"tensor optim state: {flat_tensor.shape} " \
        f"parameter: {full_padded_shape}"
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
    if len(non_none_tensors) != len(zero_dim_tensors) or \
            len(values_set) != 1 or len(dtypes) != 1:
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
    if len(non_none_non_tensors) != len(non_tensors) or \
            len(non_tensor_set) != 1:
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
    fsdp_flat_param_ids: Set[int],
    world_size: int,
) -> Dict[str, Any]:
    """
    Processes positive-dimension tensor states in ``flat_optim_state_dict`` by
    replacing them with metadata. This is done so the processed optimizer state
    dict can be broadcast from rank 0 to all ranks without copying those tensor
    states, and thus, this is meant to only be called on rank 0.

    Args:
        flat_optim_state_dict (Dict[str, Any]): Flattened optimizer state dict
            with the positive-dimension tensor states unsharded; this should
            be returned by :meth:`_flatten_optim_state` with
            ``shard_state=False``.
        fsdp_flat_param_ids (Set[int]): Parameter IDs corresponding to FSDP
            parameters.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict with positive-
        dimension tensor states replaced by metadata.
    """
    flat_osd = flat_optim_state_dict  # alias
    no_tensor_osd: Dict[str, Any] = {"state": {}}
    cpu_device = torch.device("cpu")
    for param_id, param_state in flat_osd["state"].items():
        no_tensor_osd["state"][param_id] = {}
        for state_name, state_value in param_state.items():
            is_pos_dim_tensor_state = torch.is_tensor(state_value) and \
                state_value.dim() > 0
            if not is_pos_dim_tensor_state:
                no_tensor_osd["state"][param_id][state_name] = state_value
                continue
            if param_id in fsdp_flat_param_ids:  # FSDP parameter
                # Use `_get_chunk()` to get a view and avoid allocating any new
                # tensor storage via either `clone()` or `pad()`; each rank's
                # chunk has the same padded shape, so we can pass rank 0
                chunk, num_to_pad = FSDP.FullyShardedDataParallel._get_chunk(
                    state_value, 0, world_size,
                )
                assert len(chunk.shape) == 1, \
                    f"Chunk should be 1D but got {chunk.shape}"
                # Include the padding to get the final shard shape
                info = _PosDimTensorInfo(
                    shape=torch.Size([chunk.shape[0] + num_to_pad]),
                    dtype=chunk.dtype,
                )
            else:  # non-FSDP parameter
                info = _PosDimTensorInfo(
                    shape=state_value.shape, dtype=state_value.dtype,
                )
            no_tensor_osd["state"][param_id][state_name] = info
    no_tensor_osd["param_groups"] = copy.deepcopy(flat_osd["param_groups"])
    return no_tensor_osd


def _broadcast_processed_optim_state_dict(
    processed_optim_state_dict: Optional[Dict[str, Any]],
    fsdp_flat_param_ids: Optional[Set[int]],
    rank: int,
    group,
) -> Tuple[Dict[str, Any], Set[int]]:
    """
    Broadcasts the processed optimizer state dict and the accompanying FSDP
    parameter IDs from rank 0 to all ranks.

    Args:
        processed_optim_state_dict (Optional[Dict[str, Any]]): The full
            optimizer state dict with positive-dimension tensor states replaced
            with metadata if on rank 0; ignored otherwise.
        fsdp_flat_param_ids (Optional[Set[int]]): Parameter IDs corresponding
            to FSDP parameters if on rank 0; ignored otherwise.

    Returns:
        Tuple[Dict[str, Any], Set[int]]: The processed optimizer state dict
        and the parameter IDs corresponding to FSDP parameters.
    """
    # Broadcast the two data structures rank 0 to all ranks
    obj_list = [processed_optim_state_dict, fsdp_flat_param_ids] if rank == 0 \
        else [None, None]
    dist.broadcast_object_list(obj_list, src=0, group=group)
    processed_optim_state_dict, fsdp_flat_param_ids = obj_list  # type: ignore[assignment]
    assert processed_optim_state_dict is not None
    assert fsdp_flat_param_ids is not None
    # Keep zero-dimension tensors on CPU
    return processed_optim_state_dict, fsdp_flat_param_ids


def _broadcast_pos_dim_tensor_states(
    processed_optim_state_dict: Dict[str, Any],
    fsdp_flat_param_ids: Set[int],
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
        processed_optim_state_dict (Dict[str, Any]): The full optimizer state
            dict with positive-dimension tensor states replaced with metadata;
            should be returned by :meth:`_process_pos_dim_tensor_state` and
            non-empty on all ranks (e.g. via a ``broadcast()`` from rank 0).
        fsdp_flat_param_ids (Set[int]): Parameter IDs corresponding to FSDP
            parameters.
        flat_optim_state_dict (Optional[Dict[str, Any]]): Flattened optimizer
            state dict if on rank 0; ignored on nonzero ranks.

    Returns:
        Dict[str, Any]: The optimizer state dict with the positive-dimension
        tensor state correctly populated via ``broadcast()`` s from rank 0.
    """
    assert rank != 0 or flat_optim_state_dict is not None, \
        "Expects rank 0 to pass in the flattened optimizer state dict"
    no_tensor_osd = processed_optim_state_dict  # alias
    flat_osd = flat_optim_state_dict  # alias
    for param_id, param_state in no_tensor_osd["state"].items():
        for state_name, value in param_state.items():
            is_pos_dim_tensor_state = isinstance(value, _PosDimTensorInfo)
            if not is_pos_dim_tensor_state:
                continue
            if rank == 0:
                assert flat_osd is not None
                unsharded_tensor = flat_osd["state"][param_id][state_name]
            else:
                unsharded_tensor = None
            shape, dtype = value.shape, value.dtype
            if param_id in fsdp_flat_param_ids:  # FSDP parameter
                _broadcast_sharded_pos_dim_tensor_state(
                    unsharded_tensor, param_state, state_name, shape, dtype,
                    broadcast_device, rank, world_size, group,
                )  # modify `param_state` destructively
            else:  # non-FSDP parameter
                _broadcast_unsharded_pos_dim_tensor_state(
                    unsharded_tensor, param_state, state_name, shape, dtype,
                    broadcast_device, rank, group,
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
        assert unsharded_tensor is not None, \
            "Expects rank 0 to pass in the unsharded tensor"
        get_shard = functools.partial(
            FSDP.FullyShardedDataParallel._get_shard_functional,
            unsharded_tensor,
        )
    for target_rank in range(1, world_size):
        if rank == 0:
            assert get_shard is not None
            sharded_tensor = get_shard(target_rank, world_size)[0].to(broadcast_device)
        else:
            sharded_tensor = torch.zeros(
                shape, requires_grad=False, dtype=dtype,
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
        assert unsharded_tensor is not None, \
            "Expects rank 0 to pass in the unsharded tensor"
        assert shape == unsharded_tensor.shape, \
            f"Shape mismatch: {shape} {unsharded_tensor.shape}"
        assert dtype == unsharded_tensor.dtype, \
            f"dtype mismatch: {dtype} {unsharded_tensor.dtype}"
        unsharded_tensor = unsharded_tensor.to(broadcast_device)
    else:
        unsharded_tensor = torch.zeros(
            shape, requires_grad=False, dtype=dtype, device=broadcast_device,
        )
    dist.broadcast(unsharded_tensor, src=0, group=group)
    # Keep the tensor on the broadcast device, which is typically GPU
    param_state[state_name] = unsharded_tensor


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
        if isinstance(module, FSDP.FullyShardedDataParallel):
            module._lazy_init()
            for param in module.params:  # may have none
                flat_param_to_fsdp_module[param] = module
    return flat_param_to_fsdp_module


def _get_param_id_to_param(
    model: torch.nn.Module,
    optim_input: Optional[Union[
        List[Dict[str, Any]], Iterable[torch.nn.Parameter],
    ]] = None,
) -> List[torch.nn.Parameter]:
    """
    Constructs a mapping from parameter IDs to parameters. This may be used
    both for models with ``FlatParameter`` s and without.

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
        raise TypeError(
            "Optimizer input should be an iterable of Tensors or dicts"
        )
    if all_tensors:
        return params  # type: ignore[return-value]
    assert all_dicts
    param_id_to_param = []
    for param_group in params:
        has_params_key = "params" in param_group  # type: ignore[operator]
        assert has_params_key, \
            "A parameter group should map \"params\" to a list of the " \
            "parameters in the group"
        for param in param_group["params"]:  # type: ignore[index]
            # Implicitly map `flat_param_id` (current length of the list) to
            # `param`
            param_id_to_param.append(param)
    return param_id_to_param  # type: ignore[return-value]


def _get_param_to_param_id(
    model: torch.nn.Module,
    optim_input: Optional[Union[
        List[Dict[str, Any]], Iterable[torch.nn.Parameter],
    ]] = None,
) -> Dict[torch.nn.Parameter, int]:
    """Constructs the inverse mapping of :func:`_get_param_id_to_param`."""
    param_id_to_param = _get_param_id_to_param(model, optim_input)
    return {
        param: param_id for param_id, param in enumerate(param_id_to_param)
    }


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
            assert unflat_param_id not in unflat_to_flat_param_ids, \
                "`flat_to_unflat_param_ids` has the unflattened parameter " \
                f"ID {unflat_param_id} mapped to multiple flattened " \
                "parameter IDs"
            unflat_to_flat_param_ids[unflat_param_id] = flat_param_id
    num_unflat_param_ids = len(unflat_to_flat_param_ids)
    unflat_param_ids_set = set(unflat_to_flat_param_ids.keys())
    assert unflat_param_ids_set == set(range(num_unflat_param_ids)), \
        "The set of unflattened parameter IDs should be {0, ..., " + \
        str(num_unflat_param_ids - 1) + "} but got " + \
        f"{unflat_param_ids_set}"
    return [
        unflat_to_flat_param_ids[unflat_param_id]
        for unflat_param_id in range(num_unflat_param_ids)
    ]


def _is_zero_dim_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dim() == 0
