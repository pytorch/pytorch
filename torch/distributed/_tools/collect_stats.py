import copy
from typing import Callable, Dict, List, OrderedDict, Tuple, TypedDict

import torch
from torch import nn, optim
from torch.distributed._tools.mem_tracker import (
    _MemRefType,
    _ModMemStats,
    _ModState,
    MemTracker,
)
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator, SACTradeOffStats


class ModOrder(TypedDict):
    fw_pre_order: List[str]
    bw_pre_order: List[str]
    fw_post_order: List[str]
    bw_post_order: List[str]


class ModRuntime(TypedDict):
    fw: float
    bw: float


def collect_runtime_stats(
    model: nn.Module,
    optimizer: optim.Optimizer,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = lambda x, y: sum(x, y),
    estimate_mode: str = "operator-level-cost-model",
    display_stats: bool = False,
    display_depth: int = 4,
) -> Tuple[Dict[str, ModRuntime], ModOrder]:
    """
    Collect modulewise runtime stats for a given model, as well as
    the execution order of the modules.

    Returns:
        Dict[str, ModRuntime]: A dictionary of module runtimes.
            Dictionary key is the fully qualified name (FQN) of the module.
        Dict[str, ModOrder]: List of module FQNs in their execution order.
            Dictionary key is the fully qualified name (FQN) of the module.
    """
    # We just need one actual iteration for estimation
    inp, target = inp_and_target

    def inner() -> None:
        loss = loss_fn(model(inp), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Initializing optimizer states and warm-up
    inner()

    rte = RuntimeEstimator()
    with rte(estimate_mode_type=estimate_mode):
        inner()  # We use only one iteration for estimation

    if display_stats:
        rte.display_modulewise_stats(depth=display_depth)

    return (
        {fqn: {"fw": v["fw"], "bw": v["bw"]} for fqn, v in rte.mod_runtimes.items()},
        {
            "fw_pre_order": list(rte.mod_fw_pre_order),
            "bw_pre_order": list(rte.mod_bw_pre_order),
            "fw_post_order": list(rte.mod_fw_post_order),
            "bw_post_order": list(rte.mod_bw_post_order),
        },
    )


def collect_mem_stats(
    model: nn.Module,
    optimizer: optim.Optimizer,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = lambda x, y: sum(x, y),
    display_stats: bool = False,
    display_depth: int = 4,
    display_units: str = "MiB",
    display_tabulate: bool = True,
) -> Dict[nn.Module, _ModMemStats]:
    """
    Collect modulewise memory stats for a given model.

    Returns:
        Dict[nn.Module, _ModMemStats]: A dictionary of module memory statistics.
            Dictionary key is the nn.module, and value is a ``_ModMemStats`` object.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    inp, target = inp_and_target
    mem_tracker = MemTracker()
    mem_tracker.track_external(model, optimizer)
    last_snapshot = None
    with mem_tracker as mt:
        for iter_idx in range(2):
            loss = loss_fn(model(inp), target)
            loss.backward()
            if iter_idx == 1:
                last_snapshot = mt.get_tracker_snapshot("current")
            optimizer.step()
            optimizer.zero_grad()
            if iter_idx == 0:
                mt.reset_mod_stats()
    assert last_snapshot is not None
    for mod_stats in mt.memory_tracking.values():
        if _ModState.POST_BW not in mod_stats.snapshots.keys():
            mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
                copy.deepcopy(last_snapshot)
            )

    if display_stats:
        mt.display_modulewise_snapshots(
            depth=display_depth, units=display_units, tabulate=display_tabulate
        )
        mt.display_snapshot("peak", units="MiB", tabulate=display_tabulate)

    return dict(copy.deepcopy(mt.memory_tracking))


def collect_sac_tradeoff_stats(
    model: nn.Module,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = lambda x, y: sum(x, y),
    save_tradeoff_graphs: bool = False,
    n_segments: int = 2,
    display_stats: bool = False,
    display_depth: int = 4,
    display_tabulate: bool = True,
) -> Dict[str, SACTradeOffStats]:
    """
    Collect modulewise AC tradeoff stats for a given model.

    Returns:
        Dict[str, SACTradeOffStats]: A dictionary of module memory statistics.
            Dictionary key is the fully qualified name (FQN) of the module.
            Value is a ``SACTradeOffStats`` object.
    """
    inp, target = inp_and_target
    with SACEstimator() as sace:
        _ = loss_fn(model(inp), target)
    sace.pwlf_sac_tradeoff_stats(
        n_segments=n_segments, save_tradeoff_graphs=save_tradeoff_graphs
    )
    if display_stats:
        sace.display_modulewise_sac_stats(
            depth=display_depth, print_tabular=display_tabulate
        )
    return copy.deepcopy(sace.sac_mod_tradeoff_stats)


class ModStats(TypedDict):
    fqn: str
    # per-module params
    param_per_module: int
    # per-module grads
    grad_per_module: int
    # total accumulated gradients up to and including this module
    grad_total: int
    # per module fw activation size (excluding input and output)
    act_fw_per_module: int
    # per module bw activation size during peak_bw
    act_bw_per_module: int
    # per module activation grad size during peak_bw
    act_grad_per_module: int
    # total activation size up to but excluding the current module
    # includes input of the current module (i.e., output of previous module)
    act_total: int
    # Inputs to the module
    input_per_module: int
    # Outputs of the module
    output_per_module: int
    # Total fw run-time of the module
    fw_runtime_per_module: float
    # Total bw run-time of the module
    bw_runtime_per_module: float
    # Is this module a leaf module
    is_leaf: bool
    # Total ac run-time of the module
    sac_runtime: float
    # Total ac_memory for the module
    sac_memory: int
    # Number of piecewise-linear functions used for approximating ac tradeoff curve
    n_segments: int
    # Slopes of the of piecewise-linear functions
    slopes: List[float]
    # Intercepts of the of piecewise-linear functions
    intercepts: List[float]
    # X breakpoints of the of piecewise-linear functions
    breakpoints: List[float]
    # Original trade-off curve
    tradeoff_curve: OrderedDict[float, float]


class ModuleInfo(TypedDict):
    mod_order: ModOrder
    mod_stats: List[ModStats]


def aggregate_stats(
    model: nn.Module,
    optimizer: optim.Optimizer,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable,
    dev: torch.device,
) -> ModuleInfo:
    """
    Collect modulewise stats for a given model, including memory, runtime, and AC tradeoff stats.

    Returns:
        ModuleInfo: A dictionary with module order and module stats.
    """
    mod_mem_stats = collect_mem_stats(model, optimizer, inp_and_target, loss_fn)
    (mod_runtime_stats, mod_order) = collect_runtime_stats(
        model, optimizer, inp_and_target, loss_fn
    )
    mod_sac_tradeoff_stats = collect_sac_tradeoff_stats(model, inp_and_target, loss_fn)
    module_info: ModuleInfo = {
        "mod_order": mod_order,
        "mod_stats": [],
    }

    for mod in model.modules():
        if mod_mem_stat := mod_mem_stats.get(mod, None):
            if tradeoff_stats := mod_sac_tradeoff_stats.get(mod_mem_stat.mod_fqn, None):
                sac_runtime = tradeoff_stats.sac_runtime
                sac_memory = tradeoff_stats.sac_memory
                n_segments = tradeoff_stats.n_segments
                slopes = tradeoff_stats.slopes
                intercepts = tradeoff_stats.intercepts
                breakpoints = tradeoff_stats.fit_breaks
                tradeoff_curve = tradeoff_stats.tradeoff_curve
                is_leaf = False
            else:
                sac_runtime = sac_memory = n_segments = 0
                slopes = intercepts = breakpoints = []
                tradeoff_curve: OrderedDict[float, float] = OrderedDict()  # type: ignore[no-redef]
                is_leaf = True

            mod_stat: ModStats = {
                "fqn": mod_mem_stat.mod_fqn,
                "param_per_module": mod_mem_stat.parameter_mem,
                "grad_per_module": mod_mem_stat.parameter_mem,
                "grad_total": mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                    _MemRefType.GRAD
                ],
                "act_fw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.snapshots[_ModState.PRE_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.output_mem,
                ),
                "act_bw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.ACT],
                ),
                "act_grad_per_module": (
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.TEMP]
                    - mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                        _MemRefType.TEMP
                    ]
                ),
                "act_total": mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][
                    _MemRefType.ACT
                ],
                "input_per_module": mod_mem_stat.input_mem,
                "output_per_module": mod_mem_stat.output_mem,
                "fw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["fw"],
                "bw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["bw"],
                "is_leaf": is_leaf,
                "sac_runtime": sac_runtime,
                "sac_memory": sac_memory,
                "n_segments": n_segments,
                "slopes": slopes,
                "intercepts": intercepts,
                "breakpoints": breakpoints,
                "tradeoff_curve": tradeoff_curve,
            }
            module_info["mod_stats"].append(mod_stat)

    return module_info
