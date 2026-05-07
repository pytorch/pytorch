# mypy: allow-untyped-defs
import contextlib
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor


def _validate_state_field(state: Any) -> dict[Any, Any]:
    """``state`` must be a dict keyed by packed parameter ids (not Tensors);
    per-id values must be dicts. Non-int keys are tolerated here and rejected
    later by the "extra keys" check.
    """
    if not isinstance(state, dict):
        raise RuntimeError(
            "swap_in_optimizer_state requires swapin_optimizer_state_dict['state'] to "
            "be a dict mapping packed parameter ids to per-param state dicts, "
            f"got {type(state).__name__}."
        )
    if any(isinstance(k, torch.Tensor) for k in state):
        raise RuntimeError(
            "swap_in_optimizer_state requires optimizer.state_dict()-style "
            "state keyed by packed parameter ids."
        )
    if any(isinstance(k, int) and not isinstance(v, dict) for k, v in state.items()):
        raise RuntimeError(
            "swap_in_optimizer_state requires per-parameter optimizer "
            "state entries to be dictionaries."
        )
    return state


def _validate_param_groups_field(
    optimizer: "torch.optim.Optimizer", param_groups: Any
) -> list[dict[str, Any]]:
    """``param_groups`` must be a list whose length matches the live optimizer."""
    if not isinstance(param_groups, list):
        raise RuntimeError(
            "swap_in_optimizer_state requires swapin_optimizer_state_dict['param_groups'] "
            f"to be a list of param-group dicts, got {type(param_groups).__name__}."
        )
    if len(optimizer.param_groups) != len(param_groups):
        raise RuntimeError(
            "swapin_optimizer_state_dict has a different number of parameter groups than "
            "the live optimizer."
        )
    return param_groups


def _validate_group_against_live(
    idx: int,
    group: dict[str, Any],
    swapin_group: Any,
) -> list[int]:
    """Validate a single swap-in param group against its live counterpart and
    return its packed parameter ids.

    Asserts: swap-in group is a dict, ``params`` is a list of ints, length
    matches the live group, no keys missing from the live group.
    """
    if not isinstance(swapin_group, dict):
        raise RuntimeError(
            "swap_in_optimizer_state requires each optimizer param group "
            "to be a dictionary."
        )
    swapin_param_ids = swapin_group.get("params")
    if not isinstance(swapin_param_ids, list) or not all(
        isinstance(pid, int) for pid in swapin_param_ids
    ):
        raise RuntimeError(
            "swap_in_optimizer_state requires optimizer.state_dict()-style "
            "param_groups[*]['params'] entries keyed by packed parameter ids."
        )
    if len(group["params"]) != len(swapin_param_ids):
        raise RuntimeError(
            "swapin_optimizer_state_dict param group does not match the size of "
            f"live optimizer param group {idx}."
        )
    missing_group_keys = [k for k in swapin_group if k != "params" and k not in group]
    if missing_group_keys:
        raise RuntimeError(
            "swap_in_optimizer_state requires optimizer.state_dict()-style "
            "param group keys to match the live optimizer group keys. "
            f"Missing live keys for group {idx}: {missing_group_keys}"
        )
    return swapin_param_ids


def _prepare_swap_in(
    optimizer: "torch.optim.Optimizer",
    parameters: dict[str, Tensor],
    swapin_optimizer_state_dict: dict[str, Any],
):
    """
    Validate and normalize optimizer state for ``swap_in_optimizer_state``.

    This follows the same structural assumptions as DCP-compatible optimizers,
    but consumes the raw ``optimizer.state_dict()`` format:
    ``state`` is keyed by packed parameter ids and each param group contains
    the live optimizer group fields plus a packed ``params`` list whose order
    matches ``optimizer.param_groups``.
    """
    if not optimizer.state:
        raise RuntimeError(
            "swap_in_optimizer_state requires initialized optimizer state."
        )
    if not isinstance(swapin_optimizer_state_dict, dict):
        raise RuntimeError(
            "swap_in_optimizer_state requires a DCP-style optimizer state_dict."
        )
    swapin_state = _validate_state_field(swapin_optimizer_state_dict.get("state"))
    swapin_param_groups = _validate_param_groups_field(
        optimizer, swapin_optimizer_state_dict.get("param_groups")
    )

    # Raw optimizer state_dicts address parameters by packed integer ids, so we
    # align explicit parameter tensors with optimizer.param_groups by order.
    # Example: if param_groups[*]["params"] is [[0, 1], [2]] and
    # parameters.values() is [fake_p0, fake_p1, fake_p2], then the first
    # optimizer group is swapped onto [fake_p0, fake_p1] and the second
    # onto [fake_p2].
    flat_parameters = list(parameters.values())
    flat_param_offset = 0
    seen_param_ids: set[int] = set()
    group_swapin_infos = []
    for idx, (group, swapin_group) in enumerate(
        zip(optimizer.param_groups, swapin_param_groups, strict=True)
    ):
        swapin_param_ids = _validate_group_against_live(idx, group, swapin_group)
        seen_param_ids.update(swapin_param_ids)

        next_offset = flat_param_offset + len(swapin_param_ids)
        if next_offset > len(flat_parameters):
            raise RuntimeError(
                "swap_in_optimizer_state requires the explicit parameter state to "
                "match optimizer.param_groups ordering."
            )
        swapin_params = flat_parameters[flat_param_offset:next_offset]
        flat_param_offset = next_offset

        group_swapin_infos.append(
            (
                group,  # live optimizer group to mutate
                # per-group hyperparameters from the input state_dict (the
                # exact keys depend on the optimizer class) plus the packed
                # ``params`` ids; installed onto the live group for the
                # duration of the context. Example for AdamW:
                #   {
                #     'lr': 0.1, 'weight_decay': 0.01,
                #     'betas': (0.9, 0.999), 'eps': 1e-08,
                #     'amsgrad': False, 'maximize': False,
                #     'foreach': None, 'capturable': False,
                #     'differentiable': False, 'fused': None,
                #     'decoupled_weight_decay': True,
                #     'params': [0, 1],
                #   }
                swapin_group,
                swapin_params,  # explicit tensors that replace group["params"]
                # snapshot of the live group's pre-swap values for the keys
                # we're about to overwrite, used to restore on context exit
                {k: group[k] for k in swapin_group if k != "params"},
            )
        )

    if flat_param_offset != len(flat_parameters):
        raise RuntimeError(
            "swap_in_optimizer_state requires the explicit parameter state to "
            "match optimizer.param_groups ordering."
        )

    extra_keys = [k for k in swapin_state if k not in seen_param_ids]
    if extra_keys:
        raise RuntimeError(
            "swap_in_optimizer_state requires swapin_optimizer_state_dict['state'] to "
            "be keyed only by packed parameter ids from "
            f"param_groups[*]['params']; got extra keys {extra_keys!r}."
        )
    return swapin_state, group_swapin_infos


@contextlib.contextmanager
def swap_in_optimizer_state(
    optimizer: "torch.optim.Optimizer",
    parameters: dict[str, Tensor],
    swapin_optimizer_state_dict: dict[str, Any],
):
    """A context manager that temporarily swaps an optimizer onto
    user-supplied parameter and state tensors, so ``optimizer.step()`` runs
    against those stand-ins during the context manager.

    ``optimizer.load_state_dict`` cannot serve this purpose: it only restores
    state values into the existing live params (its ``update_group`` writes
    ``new_group["params"] = group["params"]``, always discarding the input's
    ``params``). For tracing we need ``optimizer.step()`` to operate on
    stand-in tensors (e.g. FakeTensors / functional copies) without
    permanently mutating the live optimizer — so we must swap the params
    themselves, not just their state values, and put both back on exit.

    Args:
        optimizer: Live ``torch.optim.Optimizer``. Its ``state`` and each
            ``param_groups[i]["params"]`` are swapped for the duration of the
            context and restored on exit. Must already have initialized state.
        parameters: Replacement parameter tensors to swap into
            ``optimizer.param_groups[*]["params"]``, in
            ``model.named_parameters()`` order.
        swapin_optimizer_state_dict: Raw ``optimizer.state_dict()``-format dict, i.e.
            ``{"state": {<packed_id>: {...}, ...}, "param_groups": [...]}``.
            Swapped into the live optimizer for the duration of the context.
            Per-param state dicts are shallow-copied (tensor values shared
            with the input — in-place ops on existing entries propagate back;
            structural changes stay local).

    Example::

        # Trace optimizer.step() against fake parameters and fake optimizer
        # state, leaving the live optimizer untouched.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._subclasses import FakeTensorMode

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            fake_params = {
                n: fake_mode.from_tensor(p) for n, p in model.named_parameters()
            }
            fake_osd = pytree.tree_map_only(
                torch.Tensor, fake_mode.from_tensor, optimizer.state_dict()
            )


        def step_fn(params, osd):
            with swap_in_optimizer_state(optimizer, params, osd):
                optimizer.step()
            return params, osd


        gm = make_fx(step_fn)(fake_params, fake_osd)
    """
    state, group_swapin_infos = _prepare_swap_in(
        optimizer, parameters, swapin_optimizer_state_dict
    )

    original_state = optimizer.state
    original_group_params = [group["params"] for group in optimizer.param_groups]

    try:
        swapin_state: defaultdict[Tensor, Any] = defaultdict(dict)

        for group, swapin_group, swapin_params, _ in group_swapin_infos:
            # Swap the explicit tensors and saved group metadata onto the
            # live optimizer group.
            group["params"] = swapin_params
            for key, value in swapin_group.items():
                if key == "params":
                    continue
                group[key] = value

            for swapin_param, param_id in zip(
                swapin_params, swapin_group["params"], strict=True
            ):
                # Re-key per-parameter optimizer state from packed ids to the
                # swapped-in parameter tensors. Shallow-copy the per-param dict so
                # tensor values are shared (in-place ops propagate) but
                # structural mutations during the trace stay local.
                swapin_state[swapin_param] = dict(state.get(param_id, {}))

        optimizer.state = swapin_state
        yield
    finally:
        # Restore the original live optimizer object exactly.
        for group, params in zip(
            optimizer.param_groups, original_group_params, strict=True
        ):
            group["params"] = params
        for group, _, _, saved_values in group_swapin_infos:
            for key, value in saved_values.items():
                group[key] = value
        optimizer.state = original_state
