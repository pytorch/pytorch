import copy
import logging
import math
import os
from collections import defaultdict
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    TypedDict,
)

import numpy as np

import torch
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
)
from torch.distributed._tools.mem_tracker import (
    _MemRefType,
    _ModMemStats,
    _ModState,
    _PYTORCH_MIN_ALLOCATE,
    MemTracker,
)
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import (
    SACEstimator,
    SACGreedyOrderMeta,
    SACStats,
    SACTradeOffStats,
)


_DEBUG_ILP_UTILS = int(os.environ.get("DEBUG_AUTO_SAC", 0))
# Create a logger object
logger = logging.getLogger()

# Set the logging level to according to env variable
if _DEBUG_ILP_UTILS == 1:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def collect_stats(
    train_step: Callable,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    inputs: Any,
    runtime_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[MemTracker, RuntimeEstimator, SACEstimator]:
    """
    Collects memory, runtime, and activation checkpointing statistics for PyTorch models during a training step.

    This function runs a training step while gathering statistics using `MemTracker`, `RuntimeEstimator`,
    and `SACEstimator`. It tracks memory usage, runtime estimation, and selective activation checkpointing
    (SAC) trade-offs for the provided models, optimizers, and inputs.

    Args:
        train_step (Callable):
            A function that executes a single training step for the given models, optimizers, and inputs.
            It should have the signature `train_step(models, optimizers, inputs)`.
        models (List[torch.nn.Module]):
            A list of PyTorch models whose statistics are to be collected.
        optimizers (List[torch.optim.Optimizer]):
            A list of optimizers corresponding to the models.
        inputs (Any):
            The inputs required for the training step. This can be any data structure suitable for the model's `forward` method.
        runtime_kwargs (Optional[Dict[str, Any]], optional):
            A dictionary of runtime-related configuration parameters. Supported keys:
                - `"estimate_mode"` (str): The runtime estimation mode to use. Supported modes:
                    - `"operator-level-benchmark"`: Estimates runtime using operator benchmarking.
                    - `"operator-level-cost-model"`: Estimates runtime using a roofline cost model.
                    Defaults to `"operator-level-cost-model"`.
                - `"gpu_type"` (str): The GPU type to configure specific settings (e.g., `"H100_SXM_80GB"`).
                - `"custom_config"` (Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]):
                  A tuple containing:
                    - A dictionary mapping `torch.dtype` to peak FLOPS (in GFLOPS/s).
                    - A dictionary mapping `torch.dtype` to peak FLOPS factors.
                    - The peak bandwidth (in GB/s).

    Returns:
        Tuple[MemTracker, RuntimeEstimator, SACEstimator]:
            - `MemTracker`: Tracks and categorizes tensor memory during the training step.
            - `RuntimeEstimator`: Estimates the runtime for forward and backward operations under the specified mode.
            - `SACEstimator`: Provides memory and recomputation trade-off statistics for selective activation checkpointing.

    Example usage:

        .. code-block:: python

            def train_step(models, optimizers, inputs):
                # Abstract training step implementation
                ...
            with FakeTensorMode():
                models = [...]  # List of PyTorch models
                optimizers = [...]  # List of optimizers
                inputs = [...]  # Inputs for the training step

                # Collect statistics
                mem_tracker, runtime_estimator, sac_estimator = collect_mod_stats(
                    train_step=train_step,
                    models=models,
                    optimizers=optimizers,
                    inputs=inputs
                )
    """
    runtime_kwargs = runtime_kwargs or {}

    # Extract runtime_kwargs with defaults
    estimate_mode = runtime_kwargs.get("estimate_mode", "operator-level-cost-model")
    gpu_type = runtime_kwargs.get("gpu_type", "")
    custom_config = runtime_kwargs.get("custom_config", None)

    # Initialize the optimizer states
    train_step(models, optimizers, inputs)

    mem_tracker = mem_tracker = MemTracker()
    mem_tracker.track_external(*models, *optimizers, inputs)

    with mem_tracker:
        train_step(models, optimizers, inputs)

    runtime_estimator = RuntimeEstimator()
    with runtime_estimator(
        estimate_mode_type=estimate_mode, gpu_type=gpu_type, custom_config=custom_config
    ):
        train_step(models, optimizers, inputs)

    sac_estimator = SACEstimator()
    with sac_estimator(
        estimate_mode_type=estimate_mode, gpu_type=gpu_type, custom_config=custom_config
    ):
        train_step(models, optimizers, inputs)

    if _DEBUG_ILP_UTILS:
        runtime_estimator.display_modulewise_stats(depth=3)
        mem_tracker.display_modulewise_snapshots(depth=3, units="GiB", tabulate=True)
        sac_estimator.display_modulewise_sac_stats(depth=4, print_tabular=True)

    return mem_tracker, runtime_estimator, sac_estimator


class ModOrder(TypedDict):
    fw_pre_order: List[str]
    bw_pre_order: List[str]
    fw_post_order: List[str]
    bw_post_order: List[str]


class ModRuntime(TypedDict):
    fw: float
    bw: float


class ModStats(TypedDict):
    fqn: str
    # per-module params
    param_per_module: int
    # per-module grads
    grad_per_module: int
    # total accumulated gradients up to this module
    grad_total: int
    # per module fw activation size (excluding input and output)
    act_fw_per_module: int
    # per module bw activation size during peak_bw
    act_bw_per_module: int
    # per module activation grad size during peak_bw
    act_grad_per_module: int
    # total activation size accumulated including the current module
    act_total: int
    # Inputs to the module
    input_per_module: int
    # Outputs of the module
    output_per_module: int
    # Total fw run-time of the module
    fw_runtime_per_module: float
    # Total bw run-time of the module
    bw_runtime_per_module: float
    # Is the backward of this module called. If not, then it does not require grads
    requires_grad: bool
    # Is this module a leaf module
    is_leaf: bool
    # Total ac run-time of the module
    sac_runtime: float
    # Total ac_memory for the module
    sac_memory: int
    # The mandatory saved activation memory for a module
    saved_memory: int
    # Number of piecewise-linear functions used for approximating ac tradeoff curve
    n_segments: int
    # Slopes of the of piecewise-linear functions
    slopes: List[float]
    # Intercepts of the of piecewise-linear functions
    intercepts: List[float]
    # X breakpoints of the of piecewise-linear functions
    breakpoints: List[float]
    # Original trade-off curves
    tradeoff_curve: OrderedDict[float, float]


class ModuleInfo(TypedDict):
    mod_order: ModOrder
    mod_stats: List[ModStats]
    root_fqns: Set[str]
    root_dtype_factors: Dict[str, Dict[str, float]]
    root_opt_mem: Dict[str, int]


def aggregate_stats(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    mem_tracker: MemTracker,
    runtime_estimator: RuntimeEstimator,
    sac_estimator: SACEstimator,
    dev: torch.device,
    mp_policies: Optional[Dict[str, MixedPrecisionPolicy]]=None,
) -> ModuleInfo:
    """
    Collect modulewise stats for a given model, including memory, runtime, and AC tradeoff stats.

    Args:
        models (List[torch.nn.Module]):
            A list of PyTorch root modules whose statistics are collected
        optimizers (List[torch.optim.Optimizer]):
            A list of optimizers corresponding to the models
        runtime_estimator: `RuntimeEstimator` object with runtime stats
        mem_tracker: `MemTracker` object with memory stats
        sac_estimator: `SACEstimator` object with AC tradeoff stats
        dev: device the model was run on (used to extract memory stats from `MemTracker`)

    Returns:
        ModuleInfo: A dictionary with module order and module stats.
    """
    # Display Runtime and Memory Estimations
    peak_memory = mem_tracker.get_tracker_snapshot("peak")[dev]["Total"]
    logger.info(
        "Estimated Peak Memory (Before Auto-SAC): %.2f GiB", peak_memory / 2**30
    )
    logger.info(
        "Estimated Total Compute Time (Before Auto-SAC): %.2f ms",
        runtime_estimator.total_compute_time,
    )

    # Memory stats
    mod_mem_stats: Dict[torch.nn.Module, _ModMemStats] = dict(
        copy.deepcopy(mem_tracker.memory_tracking)
    )
    root_mods = {
        mod for mod, m_stat in mod_mem_stats.items() if m_stat.mod_fqn.count(".") == 0
    }
    mp_policies = mp_policies if mp_policies else {}
    root_opt_mem:Dict[str, int] = {}
    root_dtype_factors: Dict[str, Dict[str, int]] = {}
    root_fqns: Set[str] = set()
    for root_mod in root_mods:
        root_mem_stat = mod_mem_stats[root_mod]
        root_fqn = root_mem_stat.mod_fqn
        opt_state_bytes = 0
        dtypes: Set[torch.dtype] = set()
        for param in root_mod.parameters():
            dtypes.add(param.dtype)
            for opt in optimizers:
                if state := opt.state.get(param, None):
                    for v in state.values():
                        if isinstance(v, torch.Tensor) and v.device == dev:
                            opt_state_bytes += (
                                math.ceil(
                                    v.untyped_storage().nbytes() / _PYTORCH_MIN_ALLOCATE
                                )
                                * _PYTORCH_MIN_ALLOCATE
                            )

        root_opt_mem[root_fqn] = opt_state_bytes
        if root_mod not in models:
            logger.warning(
                "Found Root Module %s but was not provided to Auto-SAC as an input.\n"
                "The Module FQNs in AC decisions will include %s as one of the root modules. "
                "If %s is a sub-module, then its ancestor modules were not called during the forward pass. "
                "This may result in a sub-optimal solution.",
                root_fqn,
                root_fqn,
                root_fqn,
            )
        assert len(dtypes) == 1, f"Only uniform dtypes for parameters are supported, found {dtypes} for {root_fqn}."
        orig_dtype = dtypes.pop()
        mp_policy = mp_policies.get(root_fqn, MixedPrecisionPolicy())
        mp_param_dtype = mp_policy.param_dtype or orig_dtype
        mp_reduce_dtype = mp_policy.reduce_dtype or mp_param_dtype
        root_dtype_factors[root_fqn] = {
            "param": mp_param_dtype.itemsize / orig_dtype.itemsize,
            "reduce": mp_reduce_dtype.itemsize / orig_dtype.itemsize,
            "act": mp_param_dtype.itemsize / orig_dtype.itemsize
        }
        root_fqns.add(root_fqn)
    total_opt_memory = mem_tracker.get_tracker_snapshot("peak")[dev][_MemRefType.OPT]
    assert (
        sum(root_opt_mem.values()) == total_opt_memory
    ), "Mismatch in Optimizer State Memory"
    # Runtime stats
    mod_runtime_stats: Dict[str, ModRuntime] = {
        fqn: {"fw": v["fw"], "bw": v["bw"]}
        for fqn, v in runtime_estimator.mod_runtimes.items()
    }

    # Module order
    mod_order: ModOrder = {
        "fw_pre_order": list(runtime_estimator.mod_fw_pre_order),
        "bw_pre_order": list(runtime_estimator.mod_bw_pre_order),
        "fw_post_order": list(runtime_estimator.mod_fw_post_order),
        "bw_post_order": list(runtime_estimator.mod_bw_post_order),
    }
    # Selective Activation Checkpointing stats
    sac_estimator.pwlf_sac_tradeoff_curve(save_tradeoff_graphs=False)
    mod_sac_tradeoff_stats: Dict[str, SACTradeOffStats] = copy.deepcopy(
        sac_estimator.sac_mod_tradeoff_stats
    )
    mod_sac_greedy_stats: Dict[str, SACGreedyOrderMeta] = copy.deepcopy(
        sac_estimator.sac_mod_greedy_order_meta
    )

    mod_sac_stats: Dict[str, SACStats] = copy.deepcopy(sac_estimator.sac_mod_stats)

    module_info: ModuleInfo = {
        "mod_order": mod_order,
        "mod_stats": [],
        "root_fqns": root_fqns,
        "root_dtype_factors": root_dtype_factors,
        "root_opt_mem": root_opt_mem,
    }

    for mod_mem_stat in mod_mem_stats.values():
        if tradeoff_stats := mod_sac_tradeoff_stats.get(mod_mem_stat.mod_fqn, None):
            sac_runtime = tradeoff_stats.sac_runtime
            sac_memory = tradeoff_stats.sac_memory
            n_segments = tradeoff_stats.n_segments
            slopes = tradeoff_stats.slopes
            intercepts = tradeoff_stats.intercepts
            breakpoints = tradeoff_stats.fit_breaks
            tradeoff_curve = tradeoff_stats.tradeoff_curve
            is_leaf = False

            sac_stats_memory = mod_sac_stats[mod_mem_stat.mod_fqn].memory
            greedy_meta = mod_sac_greedy_stats[mod_mem_stat.mod_fqn]
            stored_ops, inplace_op_groups, random_inplace_ops = (
                greedy_meta.stored_ops,
                greedy_meta.inplace_op_groups,
                greedy_meta.random_inplace_ops,
            )
            stored_indices: Set[int] = set()
            for s_idx in stored_ops:
                stored_indices.add(s_idx)
                if s_idx in inplace_op_groups:
                    stored_indices.update(inplace_op_groups[s_idx])
                if s_idx in random_inplace_ops:
                    stored_indices.update(random_inplace_ops)
            saved_memory = sum(sac_stats_memory[op_idx] for op_idx in stored_indices)
        else:
            sac_runtime = sac_memory = n_segments = saved_memory = 0
            slopes = intercepts = breakpoints = []
            tradeoff_curve: OrderedDict[float, float] = OrderedDict()  # type: ignore[no-redef]
            is_leaf = True
        has_bw = _ModState.PRE_BW in mod_mem_stat.snapshots
        mod_stat: ModStats = {
            "fqn": mod_mem_stat.mod_fqn,
            "param_per_module": mod_mem_stat.parameter_mem,
            "requires_grad": has_bw,
            "grad_per_module": mod_mem_stat.parameter_mem if has_bw else 0,
            "grad_total": mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                _MemRefType.GRAD
            ]
            if has_bw
            else 0,
            "act_fw_per_module": max(
                0,
                mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][_MemRefType.ACT]
                - mod_mem_stat.snapshots[_ModState.PRE_FW][-1][dev][_MemRefType.ACT]
                - mod_mem_stat.output_mem,
            ),
            "act_bw_per_module": mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][
                _MemRefType.ACT
            ]
            if has_bw
            else 0,
            "act_grad_per_module": max(
                mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.TEMP],
                mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][_MemRefType.TEMP]
            )
            if has_bw
            else 0,
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
            "saved_memory": saved_memory,
            "n_segments": n_segments,
            "slopes": slopes,
            "intercepts": intercepts,
            "breakpoints": breakpoints,
            "tradeoff_curve": tradeoff_curve,
        }
        module_info["mod_stats"].append(mod_stat)

    return module_info


class Node(ModStats):
    index: int  # index according to forward pre-order
    pos_fw_post_order: int  # index according to forward post-order


class Graph:
    def __init__(self, 
            n: int,
            fw_post_order: List[str] = [],
            root_fqns: Set[str] = set(),
            root_dtype_factors: Dict[str, Dict[str, float]] = {},
            root_opt_mem: Dict[str, int] = {},
        ) -> None:
        self.nodes: List[Node] = []
        self.name2node: Dict[str, Node] = {}
        self.ad_matrix = np.zeros((n, n))
        self.fw_post_order = fw_post_order
        self.root_fqns = root_fqns
        self.root_dtype_factors = root_dtype_factors
        self.root_opt_mem = root_opt_mem

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.name2node[node["fqn"]] = node

    def get_root_idx(self, idx: int) -> int:
        ancestor_idxs = self.ad_matrix[:, idx].nonzero()[0]
        return ancestor_idxs.min()


def parse_module_info(module_info: ModuleInfo) -> Graph:
    """
    Parse module info and create a graph (tree) of modules. The graph will be
    used by MILP solver to find optimal SAC and/or FSDP configurations.
    """
    # sort the modules by pre-order and add them to the graph
    fw_pre_order = module_info["mod_order"]["fw_pre_order"]
    mod_stats = sorted(
        module_info["mod_stats"], key=lambda x: fw_pre_order.index(x["fqn"])
    )
    pre_order_fqns = set(fw_pre_order)
    mod_stats_fqns = {m["fqn"] for m in mod_stats}
    missing_fqns = pre_order_fqns - mod_stats_fqns
    assert len(missing_fqns) == 0, (
            f"FQNs {missing_fqns} are present in pre-order but missing in"
            " memory staistics. A module maybe reused across multiple root modules."
            " This may result in incorrectness."
        )
    n_nodes = len(mod_stats)

    # create graph
    g = Graph(
        n_nodes,
        module_info["mod_order"]["fw_post_order"],
        module_info["root_fqns"],
        module_info["root_dtype_factors"],
        module_info["root_opt_mem"]
    )

    for i, one_mod_stats in enumerate(mod_stats):
        node: Node = cast(Node, one_mod_stats)
        node["index"] = i
        node["pos_fw_post_order"] = g.fw_post_order.index(node["fqn"])
        g.add_node(node)

    # set up ancestor-descendant matrix
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if is_self_or_submodule(g.nodes[j]["fqn"], g.nodes[i]["fqn"]):
                g.ad_matrix[i][j] = 1
            else:
                break
    # Check if the list modules provided as input are root modules
    for root_fqn in g.root_fqns:
        root_idx = g.name2node[root_fqn]["index"]
        num_ancestors = g.ad_matrix[:, root_idx].sum()
        assert num_ancestors == 1, f"Expected {root_fqn} to be a root module."

    mod_call_cnt: Dict[str, int] = defaultdict(int)
    for fqn in fw_pre_order:
        mod_call_cnt[fqn] += 1
    for (
        fqn,
        count,
    ) in mod_call_cnt.items():
        assert count == 1, f"Module {fqn} is called {count} times." f" This is not supported yet."

    return g


def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    check if name_descendant is a submodule of name_ancestor, or if they are the same
    """
    return name_descendant == name_ancestor or name_ancestor + "." in name_descendant


def is_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    if name_descendant is a submodule of name_ancestor, but not the same
    """
    return name_ancestor + "." in name_descendant


def display_bytes(b: int, unit: str = "MiB") -> str:
    """
    return a string that represent the number of bytes in a desired unit
    """
    if unit == "KiB":
        return f"{b/2**10:.2f} KiB"
    if unit == "MiB":
        return f"{b/2**20:.2f} MiB"
    if unit == "GiB":
        return f"{b/2**30:.2f} GiB"
    return f"{b:.2f} bytes"
