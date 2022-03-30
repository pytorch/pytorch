from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import torch
import torch.distributed as dist
# Import the entire FSDP file to avoid circular imports
import torch.distributed.fsdp.fully_sharded_data_parallel as FSDP
from torch.distributed.fsdp.flatten_params_wrapper import FlatParameter

OPTIM_TARGET_RANK = 0  # rank on which to save full optimizer state

class ConsolidatedOptimState:
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


def _unflatten_optim_state(
    fsdp_module,
    flat_param: FlatParameter,
    flat_param_state: Dict[str, Any],
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

    Returns:
        unflat_param_state (List[Dict[str, Any]]): A :class:`list` holding
            the entries in the "state" part of the optimizer state dict
            corresponding to the unflattened parameters comprising the
            flattened parameter ``flat_param`` if on the target rank or an
            empty :class:`list` otherwise. The final optimizer state dict will
            need to map these entries using the proper unflattened parameter
            IDs.
    """
    assert sum(p is flat_param for p in fsdp_module.params) == 1, \
        "`fsdp_module` must own `flat_param`"
    consolidated_state = _communicate_optim_state(
        fsdp_module, flat_param, flat_param_state,
    )
    to_save = fsdp_module.rank == OPTIM_TARGET_RANK
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
) -> ConsolidatedOptimState:
    """
    Communicates the optimizer state for a flattened parameter ``flat_param``
    across ranks so that the target rank holds the entire non-sharded optimizer
    state.

    If ``N`` is the number of tensor optimizer states in the optimizer state
    dict, then the communication complexity is 0 if ``N = 0`` and ``N + 1``
    otherwise (where the plus 1 comes from all-gathering the padding per rank).

    Args:
        flat_param (FlatParameter): The flattened parameter.
        flat_param_state (Dict[str, Any]): The entry in the "state" part of
        the optimizer state dict corresponding to the flattened parameter.

    Returns:
        state (ConsolidatedOptimState): Consolidated optimizer state for
            ``flat_param``; the state is not populated for non-target ranks.
    """
    param_index = -1
    for i, param in enumerate(fsdp_module.params):
        if param is flat_param:
            param_index = i
            break
    assert param_index >= 0, "`fsdp_module` must own `flat_param`"

    state = ConsolidatedOptimState()
    tensor_state, zero_dim_tensor_state, non_tensor_state = \
        state.tensor_state, state.zero_dim_tensor_state, state.non_tensor_state
    process_group = fsdp_module.process_group

    tensor_buffer = None  # initialize lazily in case it is not needed
    to_save = fsdp_module.rank == OPTIM_TARGET_RANK
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
        # value directly (`deepcopy()`ing to avoid aliasing surprises)
        elif to_save:
            if _is_zero_dim_tensor(value):
                zero_dim_tensor_state[state_name] = value
            else:
                non_tensor_state[state_name] = value
    return state


def _unflatten_communicated_optim_state(
    fsdp_module,
    flat_param: FlatParameter,
    state: ConsolidatedOptimState,
) -> List[Dict[str, Any]]:
    """
    Unflattens the communicated optimizer state (given by ``tensor_state``,
    ``non_tensor_state``, and ``zero_dim_tensor_state``) for a single flattened
    parameter ``flat_param``. This should only be called on the target rank.

    Args:
        fsdp_module (FullyShardedDataParallel): FSDP module that owns
            ``flat_param``, i.e. holds it in ``self.params``.
        flat_param (FlatParameter): The flattened parameter.
        state (ConsolidatedOptimState): Consolidated optimizer state.

    Returns:
        unflat_param_state (List[Dict[str, Any]]): A :class:`list` holding
            the entries in the "state" part of the optimizer state dict
            corresponding to the unflattened parameters comprising the
            flattened parameter ``flat_param``. The final optimizer state dict
            will need to map these entries using the proper unflattened
            parameter IDs.
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


def _flatten_optim_state(
    unflat_osd_state: Dict[str, Dict[str, Any]],
    unflat_param_names: List[str],
    fsdp_module,
    flat_param: FlatParameter,
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

    Returns:
        flat_state (Dict[str, Any]): A :class:`dict` mapping state names to
            their values for a particular flattened parameter. The sharded
            optimizer state dict's "state" part will map the flattened
            parameter ID to this returned value.
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
            # Shard the flattened tensor immediately to minimize the max memory
            # usage
            sharded_flat_tensor, _ = fsdp_module._get_shard(flat_tensor)
            flat_state[state_name] = sharded_flat_tensor
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
        flat_tensor (torch.Tensor): A flattened tensor containing the optimizer
            state corresponding to ``state_name`` constructed by concatenating
            the unflattened parameter tensor states in ``pos_dim_tensors``
            (using zero tensors for any unflattened parameters without the
            state).
    """
    non_none_tensors = [t for t in pos_dim_tensors if t is not None]
    # Check that all are tensors on CPU with the same dtype
    cpu_device = torch.device("cpu")
    if not all(t.device == cpu_device for t in non_none_tensors):
        raise ValueError("All tensor optimizer state should be on CPU")
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
    tensors = [
        torch.flatten(state_value) if state_value is not None
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
        zero_dim_tensor (torch.Tensor): A zero-dimensional tensor giving the
            value of the state ``state_name`` for all unflattened parameters
            corresponding to the names ``unflat_param_names``.
    """
    non_none_tensors = [t for t in zero_dim_tensors if t is not None]
    # Enforce that all have the same value and dtype
    values_set = set(t.item() for t in zero_dim_tensors)
    dtypes = set(t.dtype for t in zero_dim_tensors)
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
        non_tensor (Any): A non-tensor giving the value of the state
            ``state_name`` for all unflattened parameters corresponding to the
            names ``unflat_param_names``.
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


def _get_flat_param_to_fsdp_module(
    model: torch.nn.Module,
):
    """
    Constructs a mapping from FSDP flattened parameters to their owning FSDP
    modules and ensures that all FSDP modules are initialized.

    Args:
        model (torch.nn.model): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance).
    """
    flat_param_to_fsdp_module = {}
    for module in model.modules():
        if isinstance(module, FSDP.FullyShardedDataParallel):
            module._lazy_init()
            for param in module.params:  # may have none
                flat_param_to_fsdp_module[param] = module
    return flat_param_to_fsdp_module


def _get_flat_param_id_to_param(
    model: torch.nn.Module,
    optim_input: Optional[Union[
        List[Dict[str, Any]], Iterable[torch.nn.Parameter],
    ]] = None,
) -> List[torch.nn.Parameter]:
    """
    Constructs a mapping from flattened parameter IDs to flattened parameters.

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
        flat_param_id_to_param (List[torch.nn.Parameter]): Mapping from
            flattened parameter IDs to flattened parameters, where the
            parameter ID is implicitly the index in the :class:`list`.
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
    flat_param_id_to_param = []
    for param_group in params:
        has_params_key = "params" in param_group  # type: ignore[operator]
        assert has_params_key, \
            "A parameter group should map \"params\" to a list of the " \
            "parameters in the group"
        for param in param_group["params"]:  # type: ignore[index]
            # Implicitly map `flat_param_id` (current length of the list) to
            # `param`
            flat_param_id_to_param.append(param)
    return flat_param_id_to_param  # type: ignore[return-value]


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
        unflat_to_flat_param_ids (List[int]): A mapping from unflattened
            parameter ID to flattened parameter ID, where the unflattened
            parameter ID is the index in the :class:`list`.
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
