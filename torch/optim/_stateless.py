# mypy: allow-untyped-defs
import contextlib
from collections import defaultdict
from typing import Any, NamedTuple

import torch
from torch import Tensor


class _GroupSwapinInfo(NamedTuple):
    """Per-group bookkeeping for ``swap_in_optimizer_params_and_state``.

    Fields:
        live_group: live ``optimizer.param_groups[i]`` dict, mutated in place.
        swapin_group: per-group hyperparameter dict from the input state_dict
            (e.g. lr, betas, ...) plus packed ``params`` ids.
        swapin_params: replacement parameter tensors for ``live_group["params"]``.
            Same order as ``swapin_group["params"]`` (i.e. the i-th tensor
            corresponds to the i-th packed id), so the two are zippable.
    """

    live_group: dict[str, Any]
    swapin_group: dict[str, Any]
    swapin_params: list[Tensor]


def _validate_state_field(state: Any) -> dict[Any, Any]:
    """Check ``state`` is a dict mapping packed parameter ids to per-param state dicts."""
    if not isinstance(state, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires swapin_optim_state['state'] to "
            "be a dict mapping packed parameter ids to per-param state dicts, "
            f"got {type(state).__name__}."
        )
    if any(isinstance(k, torch.Tensor) for k in state):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "state keyed by packed parameter ids."
        )
    if any(isinstance(k, int) and not isinstance(v, dict) for k, v in state.items()):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires per-parameter optimizer "
            "state entries to be dictionaries."
        )
    return state


def _validate_param_groups_field(
    optimizer: "torch.optim.Optimizer", param_groups: Any
) -> list[dict[str, Any]]:
    """``param_groups`` must be a list whose length matches the live optimizer's param groups."""
    if not isinstance(param_groups, list):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires swapin_optim_state['param_groups'] "
            f"to be a list of param-group dicts, got {type(param_groups).__name__}."
        )
    if len(optimizer.param_groups) != len(param_groups):
        raise RuntimeError(
            "swapin_optim_state has a different number of parameter groups than "
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

    Asserts:
    1. swap-in group is a dict.
    2. ``group['params']`` is a list of ints, and its length matches the params of the live optim group.
    3. Keys inside swapin group should match the keys in the live optim group.
    """
    if not isinstance(swapin_group, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires each optimizer param group "
            "to be a dictionary."
        )
    swapin_param_ids = swapin_group.get("params")
    if not isinstance(swapin_param_ids, list) or not all(
        isinstance(pid, int) for pid in swapin_param_ids
    ):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "param_groups[*]['params'] entries keyed by packed parameter ids."
        )
    if len(group["params"]) != len(swapin_param_ids):
        raise RuntimeError(
            f"swapin_optim_state param group {idx} has a different number of "
            "params than the live optimizer param group."
        )
    swapin_only = [k for k in swapin_group if k not in group]
    live_only = [k for k in group if k not in swapin_group]
    if swapin_only or live_only:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires optimizer.state_dict()-style "
            "param group keys to exactly match the live optimizer group keys for "
            f"group {idx}. "
            f"Keys only in swap-in: {swapin_only}. Keys only in live: {live_only}."
        )
    return swapin_param_ids


def _prepare_swap_in(
    optimizer: "torch.optim.Optimizer",
    swapin_parameters: dict[str, Tensor],
    swapin_optim_state: dict[str, Any],
) -> tuple[dict[Any, Any], list[_GroupSwapinInfo]]:
    """
    Validate and normalize optimizer state for ``swap_in_optimizer_params_and_state``.

    This follows the same structural assumptions as DCP-compatible optimizers,
    but consumes the raw ``optimizer.state_dict()`` format:
    ``state`` is keyed by packed parameter ids and each param group contains
    the live optimizer group fields plus a packed ``params`` list whose order
    matches ``optimizer.param_groups``.
    """
    if not optimizer.state:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires initialized optimizer state."
        )
    if not isinstance(swapin_optim_state, dict):
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires a DCP-style optimizer state_dict."
        )
    swapin_state = _validate_state_field(swapin_optim_state.get("state"))
    swapin_param_groups = _validate_param_groups_field(
        optimizer, swapin_optim_state.get("param_groups")
    )

    # Raw optimizer state_dicts address parameters by packed integer ids, so we
    # align explicit parameter tensors with optimizer.param_groups by order.
    # Example: if param_groups[*]["params"] is [[0, 1], [2]] and
    # swapin_parameters.values() is [fake_p0, fake_p1, fake_p2], then the
    # first optimizer group is swapped onto [fake_p0, fake_p1] and the
    # second onto [fake_p2].
    flat_parameters = list(swapin_parameters.values())
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
                "swap_in_optimizer_params_and_state requires the explicit parameter state to "
                "match optimizer.param_groups ordering."
            )
        swapin_params = flat_parameters[flat_param_offset:next_offset]
        flat_param_offset = next_offset

        # ``swapin_group`` is the per-group hyperparameter dict from the
        # input state_dict (e.g. lr, betas, eps, weight_decay, amsgrad,
        # maximize, foreach, capturable, differentiable, fused, ...) plus
        # the packed ``params`` ids; installed onto the live group for the
        # duration of the context. Example for a param-group for AdamW:
        #   {
        #     'lr': 0.1, 'weight_decay': 0.01,
        #     'betas': (0.9, 0.999), 'eps': 1e-08,
        #     'amsgrad': False, 'maximize': False,
        #     'foreach': None, 'capturable': False,
        #     'differentiable': False, 'fused': None,
        #     'decoupled_weight_decay': True,
        #     'params': [0, 1],
        #   }
        group_swapin_infos.append(
            _GroupSwapinInfo(
                live_group=group,
                swapin_group=swapin_group,
                swapin_params=swapin_params,
            )
        )

    extra_keys = [k for k in swapin_state if k not in seen_param_ids]
    if extra_keys:
        raise RuntimeError(
            "swap_in_optimizer_params_and_state requires swapin_optim_state['state'] to "
            "be keyed only by packed parameter ids from "
            f"param_groups[*]['params']; got extra keys {extra_keys!r}."
        )
    return swapin_state, group_swapin_infos


@contextlib.contextmanager
def swap_in_optimizer_params_and_state(
    optimizer: "torch.optim.Optimizer",
    swapin_parameters: dict[str, Tensor],
    swapin_optim_state: dict[str, Any],
):
    """Temporarily replace an optimizer's parameters and state with the
    supplied params and optim states, then restore them on exit.

    For the duration of the context, all optimizer APIs (including
    user hooks) see the swap-in values; the live optimizer is restored on
    exit.

    The difference between this API and ``optimizer.load_state_dict`` is
    that ``optimizer.load_state_dict`` only updates the optimizer's state
    and leaves the parameters in ``param_groups`` untouched. This API also
    swaps in the parameters, so that ``optimizer.step()`` acts on the
    swap-in parameter tensors you supply.

    Args:
        optimizer: the live optimizer; its state must already be
            initialized.
        swapin_parameters: tensors to use as parameters during the context,
            provided in the same order as the existing input parameters to
            the optimizer (most commonly in ``model.named_parameters()``
            order).
        swapin_optim_state: an ``optimizer.state_dict()``-shaped dict
            (``{"state": ..., "param_groups": ...}``) holding the state to
            install. ``"state"`` is keyed by packed integer parameter ids
            and ``"param_groups"`` mirrors ``optimizer.param_groups``,
            with each ``"params"`` entry as a list of those packed ids
            and the remaining keys carrying per-group hyperparameters
            (``lr``, ``betas``, ``foreach``, ``capturable``, ...).
            Only in-place tensor edits propagate back to the user supplied
            ``swapin_optim_state``; all other side-effects (e.g.,
            assigning a new tensor to the optim state)
            are ignored.

    Example:

        One use of this API is to run ``optimizer.step()`` against
        ``FakeTensor`` versions of the parameters and state for nonstrict
        tracing — capturing an FX graph of the step without touching the
        live optimizer::

            from torch.fx.experimental.proxy_tensor import make_fx
            from torch._subclasses import FakeTensorMode
            from torch.utils import _pytree as pytree

            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            with fake_mode:
                fake_params = {
                    n: fake_mode.from_tensor(p) for n, p in model.named_parameters()
                }
                fake_osd = pytree.tree_map_only(
                    torch.Tensor, fake_mode.from_tensor, optimizer.state_dict()
                )


            def step_fn(params, osd):
                with swap_in_optimizer_params_and_state(optimizer, params, osd):
                    optimizer.step()
                return params, osd


            gm = make_fx(step_fn)(fake_params, fake_osd)
    """
    state, group_swapin_infos = _prepare_swap_in(
        optimizer, swapin_parameters, swapin_optim_state
    )

    original_state = optimizer.state
    # Shallow-copy each group dict so we can restore the full pre-swap
    # contents (params list, lr, betas, ...) on exit. Shallow
    # copy preserves the identity of the original ``params`` list, which
    # callers may compare with ``is``.
    original_group_snapshots = [dict(g) for g in optimizer.param_groups]

    try:
        swapin_state: defaultdict[Tensor, Any] = defaultdict(dict)

        for info in group_swapin_infos:
            # Swap the explicit tensors and saved group metadata onto the
            # live optimizer group.
            info.live_group["params"] = info.swapin_params
            for key, value in info.swapin_group.items():
                if key == "params":
                    continue
                info.live_group[key] = value

            for swapin_param, param_id in zip(
                info.swapin_params, info.swapin_group["params"], strict=True
            ):
                # Re-key per-parameter optimizer state from packed ids to the
                # swapped-in parameter tensors. Shallow-copy the per-param dict so
                # inplace tensor updates are propagated but structural mutations
                # to the state_dict during the context stay local.
                swapin_state[swapin_param] = dict(state.get(param_id, {}))

        optimizer.state = swapin_state
        yield
    finally:
        # Restore each live group dict to its pre-swap contents.
        for group, snapshot in zip(
            optimizer.param_groups, original_group_snapshots, strict=True
        ):
            group.clear()
            group.update(snapshot)
        optimizer.state = original_state
