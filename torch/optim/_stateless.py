# mypy: allow-untyped-defs
import contextlib
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor


def _prepare_optimizer_reparametrization(
    optimizer: "torch.optim.Optimizer",
    parameters_and_buffers: dict[str, Tensor],
    optimizer_state_dict: dict[str, Any],
):
    """
    Validate and normalize optimizer state for ``reparametrize_optimizer_for_tracing``.

    This follows the same structural assumptions as DCP-compatible optimizers,
    but consumes the raw ``optimizer.state_dict()`` format:
    ``state`` is keyed by packed parameter ids and each param group contains
    the live optimizer group fields plus a packed ``params`` list whose order
    matches ``optimizer.param_groups``.
    """
    if not optimizer.state:
        raise RuntimeError(
            "reparametrize_optimizer_for_tracing requires initialized optimizer state."
        )
    if not isinstance(optimizer_state_dict, dict):
        raise RuntimeError(
            "reparametrize_optimizer_for_tracing requires a DCP-style optimizer state_dict."
        )

    state = optimizer_state_dict.get("state")
    if not isinstance(state, dict):
        raise RuntimeError(
            "reparametrize_optimizer_for_tracing requires optimizer_state_dict['state'] to "
            "be a dict mapping packed parameter ids to per-param state dicts, "
            f"got {type(state).__name__}."
        )
    param_groups = optimizer_state_dict.get("param_groups")
    if not isinstance(param_groups, list):
        raise RuntimeError(
            "reparametrize_optimizer_for_tracing requires optimizer_state_dict['param_groups'] "
            f"to be a list of param-group dicts, got {type(param_groups).__name__}."
        )
    if any(isinstance(name, torch.Tensor) for name in state):
        raise RuntimeError(
            "reparametrize_optimizer_for_tracing requires optimizer.state_dict()-style "
            "state keyed by packed parameter ids."
        )
    if len(optimizer.param_groups) != len(param_groups):
        raise RuntimeError(
            "optimizer_state_dict has a different number of parameter groups than "
            "the live optimizer."
        )

    group_rebind_infos = []
    # Raw optimizer state_dicts address parameters by packed integer ids, so we
    # align explicit parameter tensors with optimizer.param_groups by order.
    # Example: if param_groups[*]["params"] is [[0, 1], [2]] and
    # parameters_and_buffers.values() is [fake_p0, fake_p1, fake_p2], then the
    # first optimizer group is rebound to [fake_p0, fake_p1] and the second to
    # [fake_p2].
    flat_parameters = list(parameters_and_buffers.values())
    flat_param_offset = 0
    packed_param_ids: set[int] = set()
    for idx, (group, saved_group) in enumerate(
        zip(optimizer.param_groups, param_groups, strict=True)
    ):
        if not isinstance(saved_group, dict):
            raise RuntimeError(
                "reparametrize_optimizer_for_tracing requires each optimizer param group "
                "to be a dictionary."
            )
        names = saved_group.get("params")
        if not isinstance(names, list) or not all(
            isinstance(param_id, int) for param_id in names
        ):
            raise RuntimeError(
                "reparametrize_optimizer_for_tracing requires optimizer.state_dict()-style "
                "param_groups[*]['params'] entries keyed by packed parameter ids."
            )
        if len(group["params"]) != len(names):
            raise RuntimeError(
                "optimizer_state_dict param group does not match the size of "
                f"live optimizer param group {idx}."
            )
        next_offset = flat_param_offset + len(names)
        if next_offset > len(flat_parameters):
            raise RuntimeError(
                "reparametrize_optimizer_for_tracing requires the explicit parameter state to "
                "match optimizer.param_groups ordering."
            )
        # Slice out the explicit tensors that should back this optimizer group.
        rebind_params = flat_parameters[flat_param_offset:next_offset]
        flat_param_offset = next_offset

        for param_id in names:
            packed_param_ids.add(param_id)
            param_state = state.get(param_id, {})
            if not isinstance(param_state, dict):
                raise RuntimeError(
                    "reparametrize_optimizer_for_tracing requires per-parameter optimizer "
                    "state entries to be dictionaries."
                )

        missing_group_keys = [
            key for key in saved_group if key != "params" and key not in group
        ]
        if missing_group_keys:
            raise RuntimeError(
                "reparametrize_optimizer_for_tracing requires optimizer.state_dict()-style "
                "param group keys to match the live optimizer group keys. "
                f"Missing live keys for group {idx}: {missing_group_keys}"
            )

        group_rebind_infos.append(
            (
                group,  # live optimizer group to mutate
                saved_group,  # serialized group values to install temporarily
                rebind_params,  # explicit tensors that replace group["params"]
                {
                    key: group[key] for key in saved_group if key != "params"
                },  # restore data
            )
        )

    if flat_param_offset != len(flat_parameters):
        raise RuntimeError(
            "reparametrize_optimizer_for_tracing requires the explicit parameter state to "
            "match optimizer.param_groups ordering."
        )

    extra_keys = [key for key in state if key not in packed_param_ids]
    if extra_keys:
        raise RuntimeError(
            "reparametrize_optimizer_for_tracing requires optimizer_state_dict['state'] to "
            "be keyed only by packed parameter ids from "
            f"param_groups[*]['params']; got extra keys {extra_keys!r}."
        )
    return state, group_rebind_infos


@contextlib.contextmanager
def reparametrize_optimizer_for_tracing(
    optimizer: "torch.optim.Optimizer",
    parameters_and_buffers: dict[str, Tensor],
    optimizer_state_dict: dict[str, Any],
):
    """Temporarily rebind ``optimizer`` to explicit tensors for tracing.

    Args:
        optimizer: Live ``torch.optim.Optimizer``. Its ``state`` and each
            ``param_groups[i]["params"]`` are swapped for the duration of the
            context and restored on exit. Must already have initialized state
            (i.e. have been stepped at least once).
        parameters_and_buffers: Parameters and buffers of the module
            associated with ``optimizer``. Order must match the live
            ``optimizer.param_groups[*]["params"]``.
        optimizer_state_dict: Raw ``optimizer.state_dict()``-format dict, i.e.
            ``{"state": {<packed_id>: {...}, ...}, "param_groups": [...]}``.
            Per-param state dicts are shallow-copied (tensor values shared
            with the input — in-place ops on existing entries propagate back;
            structural changes stay local).
    """
    state, group_rebind_infos = _prepare_optimizer_reparametrization(
        optimizer, parameters_and_buffers, optimizer_state_dict
    )

    original_state = optimizer.state
    original_group_params = [group["params"] for group in optimizer.param_groups]

    try:
        rebind_state: defaultdict[Tensor, Any] = defaultdict(dict)

        for group, saved_group, rebind_params, _ in group_rebind_infos:
            # Rebind the live optimizer group to the explicit tensors and saved
            # group metadata for the trace region.
            group["params"] = rebind_params
            for key, value in saved_group.items():
                if key == "params":
                    continue
                group[key] = value

            for rebind_param, param_id in zip(
                group["params"], saved_group["params"], strict=True
            ):
                # Re-key per-parameter optimizer state from packed ids to the
                # rebound parameter tensors. Shallow-copy the per-param dict so
                # tensor values are shared (in-place ops propagate) but
                # structural mutations during the trace stay local.
                rebind_state[rebind_param] = dict(state.get(param_id, {}))

        optimizer.state = rebind_state
        yield
    finally:
        # Restore the original live optimizer object exactly.
        for group, params in zip(
            optimizer.param_groups, original_group_params, strict=True
        ):
            group["params"] = params
        for group, _, _, saved_values in group_rebind_infos:
            for key, value in saved_values.items():
                group[key] = value
        optimizer.state = original_state
