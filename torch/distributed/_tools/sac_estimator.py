import math
import os
import sys
import warnings
from collections import OrderedDict
from dataclasses import astuple, dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from typing_extensions import Self

import torch
from torch import nan, nn, UntypedStorage
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.testing._internal.composite_compliance import (
    is_inplace,
    is_inplace_view_fn,
    is_view_fn,
)
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import SAC_IGNORED_OPS


__all__ = ["SACEstimator", "SACStats", "MSPS", "SACTradeOffStats", "SACGreedyOrderMeta"]
aten = torch.ops.aten

_ADDITIONAL_IGNORED_OPS = {
    aten.lift_fresh.default,  # type: ignore[attr-defined]
    torch.ops.profiler._record_function_exit._RecordFunction,  # type: ignore[attr-defined]
    aten.clone.default,  # type: ignore[attr-defined] # seems needed for torch.compile
}
OPS_TO_ALWAYS_SKIP = SAC_IGNORED_OPS | _ADDITIONAL_IGNORED_OPS
# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)


def _get_untyped_storages(t: torch.Tensor) -> Set[torch.UntypedStorage]:
    """
    Retrieves untyped storages from a `torch.Tensor` or one of its traceable wrapper-subclass.

    Args:
       t (torch.Tensor): Input `torch.Tensor` or traceable wrapper-subclass of `torch.Tensor`.

    Returns:
        Set[torch.UntypedStorage]: Set of untyped storages.

    Warns:
        UserWarning: If the flattened input is not a tensor or traceable wrapper-subclass.
    """
    unflattened_tensors = [t]
    flattened_tensor_storages = set()
    while len(unflattened_tensors) > 0:
        obj = unflattened_tensors.pop()
        if is_traceable_wrapper_subclass(obj):
            attrs, _ = obj.__tensor_flatten__()  # type: ignore[attr-defined]
            unflattened_tensors.extend([getattr(obj, attr) for attr in attrs])
        else:
            if not hasattr(obj, "untyped_storage"):
                warnings.warn(
                    f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                flattened_tensor_storages.add(obj.untyped_storage())
    return flattened_tensor_storages


def _display_stats_tabular(headers: List[str], table_data: List[List[Any]]) -> None:
    try:
        from tabulate import tabulate
    except ImportError as err:
        raise ImportError("Please install tabulate.") from err

    # Use tabulate to print the table
    print(tabulate(table_data, headers=headers, tablefmt="rst"))


# Based on:
# https://github.com/fairinternal/xformers/blob/0ded5697a2ea15711ce45131002d04e72053cc6d/xformers/checkpoint.py#L62
@dataclass
class _SACMetadata:
    """
    Stores metadata for a single operator for SAC.

    Attributes:
        func (Any): The operator function.
        time_taken (float): The time taken by the operator.
        memory_used (float): The memory used by the operator.
        curr_idx (int): The current operator index.
        output_ids (Tuple[int, ...]): The storage IDs of the operator's outputs.
        inplace_info (Tuple[int, ...]): Tuple of self and parent operator for in-place operator.
        is_view_like (bool): Whether the operator is view-like.
        is_rand_op (bool): Whether the operator is a random operator.
    """

    func: Any
    time_taken: float
    memory_used: float
    curr_idx: int
    output_ids: Tuple[int, ...]
    inplace_info: Tuple[int, ...]
    is_view_like: bool
    is_rand_op: bool


@dataclass
class _SACModMetadata:
    """
    Stores metadata for a module for SAC.

    Attributes:
        start_idx (int): The starting index of the module's operators.
        force_store_random (bool): Whether to force store random operators in the module.
        sac_metadata (List[_SACMetadata]): List of metadata for each operator in the module.
    """

    start_idx: int
    force_store_random: bool
    sac_metadata: List[_SACMetadata]


@dataclass
class SACStats:
    """
    A class for storing Activation Checkpointing statistics corresponding to a module.

    Attributes:
        func_names (List[str]): List of operator names.
        runtimes (List[float]): List of operator runtimes in millliseconds.
        memory (List[int]): List of operator memory usage in bytes.
        view_like_ops (List[int]): Indices of view-like operators.
        rand_ops (List[int]): Indices of random operators.
        saved_autograd_ops (List[int]): Indices of operator results saved by autograd engine.
        inplace_ops (List[Tuple[int, int]]): Tuple of indices of op and its first parent for Inplace operators.
        force_store_random (bool): Whether to force store random operator results.
    """

    func_names: List[str]
    runtimes: List[float]
    memory: List[int]
    view_like_ops: List[int]
    rand_ops: List[int]
    saved_autograd_ops: List[int]
    inplace_ops: List[Tuple[int, int]]
    force_store_random: bool


class MSPS(NamedTuple):
    """
    Represents Memory and Runtime Statistics for an operator/operator group.

    Attributes:
        func_names (Set[str]): Set of operator/operator group names.
        op_idx (int): Operator index (group head index incase of operator groups).
        memory (int): Memory usage in bytes.
        runtime (float): Runtime in milliseconds.
        msps (float): Memory per second calculated as memory/runtime.
    """

    func_names: Set[str]
    op_idx: int
    memory: int
    runtime: float
    msps: float


@dataclass
class SACTradeOffStats:
    """
    Stores statistics for activation-checkpointing trade-off.

    Attributes:
        n_segments (int): Number of piecewise linear segments fitted to the trade-off curve.
        slopes (List[float]): Slopes of the pieces of linear segments fitted to the trade-off curve.
        intercepts (List[float]): Intercepts of the of the pieces of linear segments fitted to the trade-off curve.
        fit_breaks (List[float]): Breakpoints of the of the pieces of linear segments fitted to the trade-off curve.
        tradeoff_curve (OrderedDict[float, float]): Trade-off curve data of memory discarded vs recomputation time.
        sac_memory (int): Total memory of operations available for activation checkpointing in bytes.
        sac_runtime (float): Total runtime of operations available for activation checkpointing in milliseconds.
    """

    n_segments: int
    slopes: List[float]
    intercepts: List[float]
    fit_breaks: List[float]
    tradeoff_curve: OrderedDict[float, float]
    sac_memory: int
    sac_runtime: float


@dataclass
class SACGreedyOrderMeta:
    """
    Stores metadata for Greedy-order SAC.

    Attributes:
        recomputed_ops (Set[int]): Set of operator indices to be recomputed.
        stored_ops (Set[int]): Set of operator indices to be stored.
        inplace_op_groups (Dict[int, Set[int]]): Dictionary of inplace operator groups from group-head to operators.
        random_ops_group (Dict[int, Set[int]]): Dictionary of random op group head to random ops.
        msps_meta (List[MSPS]): List of Memory and Runtime Statistics for operators.
    """

    recomputed_ops: Set[int]
    stored_ops: Set[int]
    inplace_op_groups: Dict[int, Set[int]]
    random_ops_group: Dict[int, Set[int]]
    msps_meta: List[MSPS]


class SACEstimator(TorchDispatchMode):
    """
    Estimates the memory and recomputation time trade-offs for applying Selective Activation Checkpointing (SAC).

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the memory and
    runtime trade-offs of functions or ``torch.nn.Module``s for Selective Activation Checkpointing (SAC). It provides
    detailed statistics and metadata information for operators of each module and provides a greedy order for selecting
    the operators to be recomputed/checkpointed.  It also constructs the per-module trade-off graph of discarded memory
    vs recomputation time for the obtained greedy order. Using ``RuntimeEstimator`` under the hood, it supports two
    estimation modes, `operator-level-benchmark` and (`operator-level-cost-model` (roofline model).

    Attributes:
        sac_mod_stats (Dict[str, SACStats]): Dictionary from module FQN (fuly qualified name) to ``SACStats``.
        sac_mod_tradeoff_stats (Dict[str, SACTradeOffStats]): Dictionary from module FQN to ``SACTradeOffStats``.
        sac_mod_greedy_order_meta (Dict[str, SACGreedyOrderMeta]): Dictionary from module FQN to ``SACGreedyOrderMeta``.

    Note:
        1) This class is designed to be used under ``FakeTensorMode``.
        2) Currently, it only supports estimation of compute time and memory usage, and does not consider communication.

    Example usage:

        .. code-block:: python

            sac_estimator = SACEstimator()
            with FakeTensorMode():
                module = ...
                inp = ...
                with sac_estimator('operator-level-cost-model'):
                    output = module(inp)
                sac_estimator.display_modulewise_sac_stats(depth=4, print_tabular=True)
    """

    def __init__(self) -> None:
        self.sac_mod_stats: Dict[str, SACStats] = {}
        self.sac_mod_tradeoff_stats: Dict[str, SACTradeOffStats] = {}
        self.sac_mod_greedy_order_meta: Dict[str, SACGreedyOrderMeta] = {}
        self._mod_tracker = ModTracker()
        self._sac_metadata: List[_SACMetadata] = []
        self._sac_mod_metadata: Dict[str, _SACModMetadata] = {}
        self._leaf_modules: Set[str] = set()
        self._saved_tensor_hook_ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack_hook, lambda x: x
        )
        self._saved_tensor_ids: Set[int] = set()
        self._estimate_runtime = RuntimeEstimator._roofline_estimate

    def _pack_hook(self, x: torch.Tensor) -> torch.Tensor:
        # Hook function to track underlying storage IDs of tensors
        # Updates the _saved_tensor_ids set with the IDs of the tensor's storages
        # Used in conjunction with torch.autograd.graph.saved_tensors_hooks
        untyped_storages = _get_untyped_storages(x)
        storage_ids = (hash(st) for st in untyped_storages)
        self._saved_tensor_ids.update(storage_ids)
        return x

    def _pre_fw_hook(self, mod: nn.Module, inputs: Any) -> None:
        # Pre-forward hook function to prepare module metadata
        # Tracks module FQN, force store random flag, and ``SACModMetadata``
        # Initializes metadata for non-leaf modules, marks leaf modules
        mod_fqn = self._mod_tracker.get_known_fqn(mod)
        assert mod_fqn is not None
        num_children = sum(1 for _ in mod.children())
        if num_children > 0:
            force_store_random = self._get_force_store_random(inputs)
            self._sac_mod_metadata[mod_fqn] = _SACModMetadata(
                start_idx=len(self._sac_metadata),
                force_store_random=force_store_random,
                sac_metadata=[],
            )
        else:
            self._leaf_modules.add(mod_fqn)

    def _post_fw_hook(self, mod: nn.Module, inputs: Any, outputs: Any) -> None:
        # 1. Retrieves the module's FQN and checks if it's a leaf module
        # 2. If not a leaf module, computes:
        #    - ``SACStats`` using the module's metadata and force store random flag
        #    - ``SACGreedyOrderMeta`` using the computed SAC statistics
        mod_fqn = self._mod_tracker.get_known_fqn(mod)
        assert mod_fqn is not None
        if mod_fqn in self._leaf_modules:
            return
        else:
            self.sac_mod_stats[mod_fqn] = self._get_sac_stats(
                data=self._sac_mod_metadata[mod_fqn].sac_metadata,
                force_store_random=self._sac_mod_metadata[mod_fqn].force_store_random,
            )
            self.sac_mod_greedy_order_meta[mod_fqn] = self._get_greedy_order_meta(
                self.sac_mod_stats[mod_fqn]
            )

    def _get_force_store_random(self, inputs: Any) -> bool:
        flat_inputs, _ = tree_flatten(inputs)
        return all(not isinstance(x, torch.Tensor) for x in flat_inputs)

    def _get_sac_stats(
        self, data: List[_SACMetadata], force_store_random: bool
    ) -> SACStats:
        # 1. Ignore the operations that should be skipped by SAC such as aten.detach.default because autograd
        # inserts those during backward and it breaks the fwd-bwd alignment
        filtered_data = [x for x in data if x.func not in OPS_TO_ALWAYS_SKIP]

        (
            ops,
            runtimes_,
            memory_,
            new_ids,
            output_ids,
            inplace_ops_,
            view_like_ops_,
            rand_ops_,
        ) = zip(*[astuple(x) for x in filtered_data], strict=True)

        # 2. Extract the metadata information
        runtimes = list(runtimes_)
        memory = list(memory_)
        func_names = [op._overloadpacket.__name__ for op in ops]
        view_like_ops = [i for i, x in enumerate(view_like_ops_) if x]
        rand_ops = [i for i, x in enumerate(rand_ops_) if x]
        saved_autograd_ops = [
            i
            for i, out_ids in enumerate(output_ids)
            if set(out_ids).issubset(self._saved_tensor_ids)
        ]

        # 3. Remap the inplace indices as we have removed OPS_TO_ALWAYS_SKIP
        # FIXME @sanketpurandare: Fix this by changing the parent of the inplace-op
        # to itself if the original parent is in OPS_TO_ALWAYS_SKIP.
        try:
            inplace_ops = [tuple(map(new_ids.index, x)) for x in inplace_ops_ if x]
        except ValueError as err:
            raise ValueError(
                f"The remapping of inplace ops failed since one of the inplace op parents"
                f" must have been present in {OPS_TO_ALWAYS_SKIP}"
            ) from err

        # 4. The last operation is always stored as the output of the checkpoint
        # block, so we can avoid recomputing it. We set the memory to zero
        # instead of adding a new constraint because we want both the 0 and 1
        # endpoints for memory_budget to be valid
        # FIXME @sanketpurandare: this heuristic for finding the last non-view non-inplace op
        # might not always be correct, which would yield suboptimal policies
        last_op = len(ops) - 1
        skip_ops_ = set(view_like_ops) | set({x[0] for x in inplace_ops})
        reversed_skip_ops = sorted(skip_ops_, reverse=True)
        for op in reversed_skip_ops:
            if op == last_op:
                last_op -= 1

        memory[last_op] = 0

        # 5. Create a single ``SACStats`` object for the entire block of ``_SACMetadata``.
        return SACStats(
            func_names=func_names,
            runtimes=runtimes,
            memory=memory,
            view_like_ops=view_like_ops,
            rand_ops=rand_ops,
            saved_autograd_ops=saved_autograd_ops,
            inplace_ops=inplace_ops,  # type: ignore[arg-type]
            force_store_random=force_store_random,
        )

    def _get_inplace_metadata(
        self, func: Any, out_storages: Set[UntypedStorage]
    ) -> Tuple[int, Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
        # 1. Get the current index of the metadata obtained so far
        curr_idx = len(self._sac_metadata)
        # 2. Get the set of active modules that are not leaf
        active_mod_fqns: Set[str] = {
            par for par in self._mod_tracker.parents if par not in self._leaf_modules
        }
        # 3. Output ids are the identifies of the storage objects corresponding to the tensors
        output_ids = tuple(hash(st) for st in out_storages)
        # 4. If the function is not inplace, return
        if not is_inplace(func):
            return curr_idx, output_ids, {mod_fqn: () for mod_fqn in active_mod_fqns}

        op_idx = curr_idx
        # 5. Initialize the parent op ids of the inplace op for each of the active modules
        mod_op_parent_idxs: Dict[str, int] = {
            mod_fqn: -1 for mod_fqn in active_mod_fqns
        }
        for i, d in enumerate(self._sac_metadata):
            # 6. Find the first occurence of a tensor corresponding to each module that
            # shares the same storage as the current tensor
            past_output_ids = d.output_ids
            if set(output_ids).issubset(set(past_output_ids)):
                for mod_fqn, op_parent_idx in mod_op_parent_idxs.items():
                    if op_parent_idx == -1:
                        if acm_stats := self._sac_mod_metadata.get(mod_fqn, None):
                            if i >= acm_stats.start_idx:
                                mod_op_parent_idxs[mod_fqn] = i
                        else:
                            assert mod_fqn == "Global"
                            mod_op_parent_idxs[mod_fqn] = i
        # 7. If no parent tensor is found, then it's probably an inplace op on the arguments
        # so one can just store the current-op idx as parent idx
        for mod_fqn, op_parent_idx in mod_op_parent_idxs.items():
            if op_parent_idx < 0:
                mod_op_parent_idxs[mod_fqn] = op_idx
        mod_inplace_info = {
            mod_fqn: (op_idx, mod_op_parent_idxs[mod_fqn])
            for mod_fqn in active_mod_fqns
        }
        return curr_idx, output_ids, mod_inplace_info  # type: ignore[return-value]

    def __torch_dispatch__(  # type: ignore[no-untyped-def]
        self, func, types, args=..., kwargs=None
    ):
        # 1. Get the runtime estimate
        out, op_time = self._estimate_runtime(func, args, kwargs)
        flat_outs, _ = tree_flatten(out)
        out_storages_cuda: Set[UntypedStorage] = set()
        out_storages_cpu: Set[UntypedStorage] = set()
        cuda_devices: Set[torch.device] = set()
        for o in flat_outs:
            if isinstance(o, torch.Tensor):
                if o.device.type == "cuda":
                    out_storages_cuda.update(_get_untyped_storages(o))
                    cuda_devices.add(o.device)
                else:
                    out_storages_cpu.update(_get_untyped_storages(o))

        # Check if there's more than 1 CUDA device
        assert (
            len(cuda_devices) <= 1
        ), f"{func.__name__}'s output has more than 1 CUDA devices {cuda_devices}"

        # 2. Get the memory consumed by output
        nbytes_cuda = sum(
            math.ceil(st.nbytes() / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
            for st in out_storages_cuda
        )
        nbytes_cpu = sum(st.nbytes() for st in out_storages_cpu)
        nbytes = nbytes_cuda + nbytes_cpu
        # 3. Get the current operator index, output storage identifiers and inplace metadata
        out_storages = out_storages_cuda | out_storages_cpu
        curr_idx, output_ids, mod_inplace_info = self._get_inplace_metadata(
            func, out_storages
        )
        # 4. Determine if the function is in-place, random-op or a view-like
        is_view_like = is_view_fn(func) or is_inplace_view_fn(func)
        is_rand_op = torch.Tag.nondeterministic_seeded in func.tags
        if is_view_like:
            nbytes = 0
        # sdpa has non-deterministic seed, but might be deterministic
        # if no dropout is applied
        if func.overloadpacket.__name__ == "_scaled_dot_product_flash_attention":
            is_rand_op = kwargs.get("dropout_p", 0) != 0
        # 5. Create metadata information per active non-leaf module
        for mod_fqn in self._mod_tracker.parents:
            if mod_fqn in self._leaf_modules:
                continue
            acm = _SACMetadata(
                func=func,
                time_taken=op_time,
                memory_used=nbytes,
                curr_idx=curr_idx,
                output_ids=output_ids,
                inplace_info=mod_inplace_info[mod_fqn],
                is_view_like=is_view_like,
                is_rand_op=is_rand_op,
            )
            if acm_stats := self._sac_mod_metadata.get(mod_fqn, None):
                acm_stats.sac_metadata.append(acm)
            else:
                assert (
                    mod_fqn == "Global"
                ), f"Module {mod_fqn} not found in AC Mod Stats"
                self._sac_metadata.append(acm)

        return out

    def _get_greedy_order_meta(self, sac_stats: SACStats) -> SACGreedyOrderMeta:
        # An inplace-op group is a set of inplace-ops that operate on the same underlying tensor storage.
        # 1. inplace_op_groups: A dictionary from the top-most parent of inplace-ops to the inplace-ops in the group
        #   The top-most op can itself be an inplace-op or can be a non-inplace op.
        # 2. inplace_op_to_group_head: A dictionary that maps all the inplace-ops to their respective group heads.
        inplace_op_groups: Dict[int, Set[int]] = {}
        inplace_op_to_group_head: Dict[int, int] = dict(sac_stats.inplace_ops)

        # Initialize inplace_op_groups using inplace_op_to_group_head
        for op_idx, group_head_idx in inplace_op_to_group_head.items():
            op_group = inplace_op_groups.setdefault(group_head_idx, {group_head_idx})
            op_group.add(op_idx)

        # Like inplace ops, all of the random ops in the function/module should all be either recomputed or saved
        # as a group. This is because, they affect the ranom seed generator. If force_store_random is set True,
        # all of the random ops will be stored by default. For easy of manageability, we store the top-most random op
        # as the leader of the random_ops_group.
        random_ops_group: Dict[int, Set[int]] = {}
        random_group_head_idx = min(sac_stats.rand_ops, default=-1)
        has_rand_ops = bool(sac_stats.rand_ops)
        if has_rand_ops:
            random_ops_group[random_group_head_idx] = set(sac_stats.rand_ops)

        # 1. Random ops are stored if force_store_random is set
        # 2. View-like ops are recomputed by default
        # 3. For inplace_op_groups:
        #   a) If the head of this group is an inplace op, then we have to store the entire group.
        #   b) If any op in the group is random and force_store_random is set, then entire group will be stored.
        #   c) If none of ops in the group are random and the head of the group is not an in-place op, then
        #       this group can be considered for recomputation in its entireity
        stored_ops: Set[int] = set()
        recomputed_ops: Set[int] = set()
        # Case 1:
        if has_rand_ops and sac_stats.force_store_random:
            stored_ops.add(random_group_head_idx)
        # Case 2:
        recomputed_ops.update(set(sac_stats.view_like_ops))

        for group_head_idx, op_group in inplace_op_groups.items():
            # Case 3a:
            if group_head_idx in inplace_op_to_group_head:
                stored_ops.add(group_head_idx)
            # Case 3b:
            if (
                sac_stats.force_store_random & len(op_group & set(sac_stats.rand_ops))
                > 0
            ):
                stored_ops.add(group_head_idx)

        # The potential recompute candidates are populated as:
        recompute_candidates: Set[int] = set()
        # 1) The random group head if it is not stored
        if has_rand_ops and random_group_head_idx not in stored_ops:
            recompute_candidates.add(random_group_head_idx)
        # 2) The in-place op group heads that are not stored
        recompute_candidates.update(set(inplace_op_groups.keys()) - stored_ops)
        # 3) The non-inplace and non-random ops that are neither stored nor recomputed by default
        recompute_candidates.update(
            set(range(len(sac_stats.memory)))
            - recomputed_ops
            - stored_ops
            - set(inplace_op_to_group_head.keys())
            - set(sac_stats.rand_ops)
        )

        # We define msps for a recomp candidate as the ratio of memory/runtime aka memory savings per second
        msps_meta: List[MSPS] = []
        for cand_idx in recompute_candidates:
            op_indices = {cand_idx}
            if cand_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[cand_idx])
            if has_rand_ops and cand_idx == random_group_head_idx:
                op_indices.update(sac_stats.rand_ops)

            mem = sum(sac_stats.memory[op_idx] for op_idx in op_indices)
            runtime = sum(sac_stats.runtimes[op_idx] for op_idx in op_indices)
            func_names = {sac_stats.func_names[op_idx] for op_idx in op_indices}
            msps = (mem / runtime) if runtime > 0 else sys.float_info.max
            msps_meta.append(MSPS(func_names, cand_idx, mem, runtime, msps))
        # We choose canidates to be recomputed based on increasing msps
        msps_meta.sort(key=lambda x: x.msps, reverse=True)
        return SACGreedyOrderMeta(
            recomputed_ops, stored_ops, inplace_op_groups, random_ops_group, msps_meta
        )

    def _get_sac_tradeoff_pwlf_stats(
        self,
        sac_stats: SACStats,
        greedy_order_meta: SACGreedyOrderMeta,
        n_segments: int = 2,
        save_tradeoff_graph: bool = False,
        filename: str = "ac_tradeoff",
    ) -> SACTradeOffStats:
        try:
            import numpy as np  # type: ignore[import-not-found]
            import pwlf  # type: ignore[import-untyped, import-not-found]
        except ImportError as err:
            raise ImportError("Please install pwlf and numpy package.") from err

        stored_ops, recomputed_ops, inplace_op_groups, random_ops_group, msps_meta = (
            greedy_order_meta.stored_ops,
            greedy_order_meta.recomputed_ops,
            greedy_order_meta.inplace_op_groups,
            greedy_order_meta.random_ops_group,
            greedy_order_meta.msps_meta,
        )
        # 1. Intitialize the discarded memory and recomputation runtime to sum of already chosen recomputed_ops
        recomp_indices: Set[int] = set()
        for r_idx in recomputed_ops:
            recomp_indices.add(r_idx)
            if r_idx in inplace_op_groups:
                recomp_indices.update(inplace_op_groups[r_idx])
            if r_idx in random_ops_group:
                recomp_indices.update(random_ops_group[r_idx])

        discarded_mem = sum(sac_stats.memory[op_idx] for op_idx in recomp_indices)
        recomp_runtime = sum(sac_stats.runtimes[op_idx] for op_idx in recomp_indices)
        # 2. Initialize the max recomputation time and total recomputation memory
        sac_runtime = sum(sac_stats.runtimes)
        sac_memory = sum(sac_stats.memory)
        # 3. Tradeoff curve stores the KV pair of the dicarded memory to total memory and,
        # recomputation time to total runtime incurred.
        delta = 1e-2
        tradeoff_curve = OrderedDict()
        # 4. Initialize the trade-off curve with the stats of of already chosen recomputed_ops
        tradeoff_curve[(discarded_mem / sac_memory) + delta] = (
            recomp_runtime / sac_runtime
        )
        # 5. Update the trade-off curve with memory and runtime stats of SAC candidates in the
        # greedy order of their ``MSPS``.
        for cand in msps_meta:
            discarded_mem += cand.memory
            recomp_runtime += cand.runtime
            tradeoff_curve[(discarded_mem / sac_memory) + delta] = (
                recomp_runtime / sac_runtime
            )
        # 6. Finally, we add the memory and recomputation time of the always stored ops.
        stored_indices: Set[int] = set()
        for s_idx in stored_ops:
            stored_indices.add(s_idx)
            if s_idx in inplace_op_groups:
                stored_indices.update(inplace_op_groups[s_idx])
            if s_idx in random_ops_group:
                stored_indices.update(random_ops_group[s_idx])
        discarded_mem += sum(sac_stats.memory[op_idx] for op_idx in stored_indices)
        recomp_runtime += sum(sac_stats.runtimes[op_idx] for op_idx in stored_indices)
        tradeoff_curve[(discarded_mem / sac_memory) + delta] = (
            recomp_runtime / sac_runtime
        )
        x_ = list(tradeoff_curve.keys())
        y_ = list(tradeoff_curve.values())
        # 7. We shift the y values to left and x values to right to upperbound the trade-off function
        # TODO: Write a better explanation why this needs to be done
        x = x_[: len(x_) - 1]
        y = y_[1:]
        tradeoff_pwlf = pwlf.PiecewiseLinFit(x, y)
        # 8. Fit a piecewise linear function with the specified number of segments to the trade-off curve.
        n_segments = max(min(len(x) - 2, n_segments), 1)
        tradeoff_pwlf.fit(n_segments=n_segments)

        # save prediction graph
        def save_prediction_graph(
            pwlf_: pwlf.PiecewiseLinFit, x: List[float], y: List[float], filename: str
        ) -> None:
            try:
                import matplotlib.pyplot as plt  # type: ignore[import-not-found]
                import numpy as np  # type: ignore[import-not-found]
            except ImportError as err:
                raise ImportError(
                    "Install matplotlib and numpy using pip: pip install matplotlib numpy"
                ) from err
            # predict for the determined points
            xHat = np.linspace(min(x), max(x), num=10000)
            yHat = pwlf_.predict(xHat)

            # plot the results
            plt.figure()
            plt.plot(x, y, "o", label="Shifted")
            plt.plot(xHat, yHat, "-", label="Predicted")
            plt.plot(x_, y_, "x", label="Original")
            plt.ylabel("Recomp time / Total recomp time")
            plt.xlabel("Memory discarded / Total memory")
            plt.legend()
            plt.title(f"{filename}")
            plt.suptitle(
                f"Total Memory = {sac_memory} B Total Runtime = {sac_runtime:.4f} ms",
                fontsize=10,
            )
            folder_name = "tradeoff_graphs"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            # Save the plots in the folder
            plt.savefig(os.path.join(folder_name, f"{filename}.png"))

        if save_tradeoff_graph:
            save_prediction_graph(tradeoff_pwlf, x, y, filename)
        # 9. Obtain the slopes, intercepts and breakpoints of the fitted piecewise linear functions
        slopes = tradeoff_pwlf.calc_slopes().tolist()
        assert isinstance(tradeoff_pwlf.intercepts, np.ndarray) and isinstance(
            tradeoff_pwlf.fit_breaks, np.ndarray
        )
        intercepts = tradeoff_pwlf.intercepts.tolist()
        fit_breaks = tradeoff_pwlf.fit_breaks.tolist()
        return SACTradeOffStats(
            n_segments=n_segments,
            slopes=slopes,
            intercepts=intercepts,
            fit_breaks=fit_breaks,
            tradeoff_curve=tradeoff_curve,
            sac_memory=sac_memory,
            sac_runtime=sac_runtime,
        )

    def display_sac_stats(
        self, sac_stats: SACStats, print_tabular: bool = False
    ) -> None:
        """
        Displays the SAC statistics.

        Args:
            sac_stats (SACStats): The SAC statistics to display.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            1. Total Memory: The total memory usage in bytes.
            2. Total Runtime: The total runtime in milliseconds.
            3. Store Random: A flag indicating whether to force store random operator results.

            Followed by a table with the following columns:
            1. Op Idx: The operator index.
            2. Op Name: The operator name.
            3. Runtimes (ms): The operator runtime in milliseconds.
            4. Memory (B): The operator memory usage in bytes.
            5. View-like: A flag indicating whether the operator is view-like.
            6. Random: A flag indicating whether the operator is random.
            7. Saved Autograd: A flag indicating whether the operator's result is saved by autograd engine.
            8. In-place: The index of the operator's first parent, or None if not in-place.

        If print_tabular is True, the table is printed in a tabular format.
        Otherwise, the table is printed in a plain text format.
        """
        print(
            f"Total Memory: {sum(sac_stats.memory)} B Total Runtime: {sum(sac_stats.runtimes)} ms"
            f" Store Random: {sac_stats.force_store_random}"
        )
        table_data = []
        op_parent = dict(sac_stats.inplace_ops)
        for i, fn_name in enumerate(sac_stats.func_names):
            row = [
                str(i),
                fn_name,
                f"{sac_stats.runtimes[i]:.4f}",
                str(sac_stats.memory[i]),
                str(i in sac_stats.view_like_ops),
                str(i in sac_stats.rand_ops),
                str(i in sac_stats.saved_autograd_ops),
                str(op_parent.get(i, None)),
            ]
            table_data.append(row)
        # Define headers
        headers = [
            "Op Idx",
            "Op Name",
            "Runtimes(ms)",
            "Memory (B)",
            "View-like",
            "Random",
            "Saved Autograd",
            "In-place",
        ]
        if print_tabular:
            _display_stats_tabular(headers, table_data)
        else:
            max_widths = [0 for _ in range(len(headers))]
            table_data.insert(0, headers)
            for row in table_data:
                for i, elem in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(elem))
            for row in table_data:
                print(
                    "\t".join(
                        [f"{elem:<{max_widths[i]}}" for i, elem in enumerate(row)]
                    )
                )

    def display_sac_tradeoff_stats(
        self,
        greedy_order_meta: SACGreedyOrderMeta,
        sac_stats: SACStats,
        print_tabular: bool = False,
    ) -> None:
        """
        Displays the SAC trade-off statistics.

        Args:
            greedy_order_meta (SACGreedyOrderMeta): The SAC greedy order metadata.
            sac_stats (SACStats): The SAC statistics.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            A table with the following columns:
            1. Op Id(s): The operator index(es).
            2. Op Name(s): The operator name(s).
            3. Discarded Mem (%): The percentage of discarded memory.
            4. Discarded Mem (B): The discarded memory in bytes.
            5. Recomp time (%): The percentage of recomputed time.
            6. Recomp time (ms): The recomputed time in milliseconds.
            7. MSPS: The memory per second.
            8. Always Stored: A flag indicating whether the operator is always stored.
            9. Always Recomputed: A flag indicating whether the operator is always recomputed.

        If print_tabular is True, the table is printed in a tabular format.
        Otherwise, the table is printed in a plain text format.
        """
        table_data = []
        total_memory, total_runtime = sum(sac_stats.memory), sum(sac_stats.runtimes)
        discarded_mem: int = 0
        recomp_runtime: float = 0.0

        def append_row(
            op_indices: Set[int],
            func_names: Set[str],
            msps: Optional[float] = None,
            stored: Optional[bool] = False,
            recomputed: Optional[bool] = False,
        ) -> None:
            row = [
                str(op_indices),
                str(func_names),
                f"{discarded_mem / total_memory:.4f}",
                str(discarded_mem),
                f"{recomp_runtime / total_runtime:.4f}",
                str(recomp_runtime),
                f"{msps:.2e}" if msps is not None else str(nan),
                str(stored),
                str(recomputed),
            ]
            table_data.append(row)

        stored_ops, recomputed_ops, inplace_op_groups, random_ops_group, msps_meta = (
            greedy_order_meta.stored_ops,
            greedy_order_meta.recomputed_ops,
            greedy_order_meta.inplace_op_groups,
            greedy_order_meta.random_ops_group,
            greedy_order_meta.msps_meta,
        )

        for op_idx in recomputed_ops:
            op_indices: Set[int] = {op_idx}
            if op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[op_idx])
            if op_idx in random_ops_group:
                op_indices.update(random_ops_group[op_idx])
            discarded_mem += sum(sac_stats.memory[i] for i in op_indices)
            recomp_runtime += sum(sac_stats.runtimes[i] for i in op_indices)
            func_names = {sac_stats.func_names[i] for i in op_indices}
            append_row(op_indices, func_names, recomputed=True)

        for cand in msps_meta:
            discarded_mem += cand.memory
            recomp_runtime += cand.runtime
            op_indices = {cand.op_idx}
            if cand.op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[cand.op_idx])
            if cand.op_idx in random_ops_group:
                op_indices.update(random_ops_group[cand.op_idx])
            append_row(op_indices, cand.func_names, msps=cand.msps)

        for op_idx in stored_ops:
            op_indices = {op_idx}
            if op_idx in inplace_op_groups:
                op_indices.update(inplace_op_groups[op_idx])
            if op_idx in random_ops_group:
                op_indices.update(random_ops_group[op_idx])
            discarded_mem += sum(sac_stats.memory[i] for i in op_indices)
            recomp_runtime += sum(sac_stats.runtimes[i] for i in op_indices)
            func_names = {sac_stats.func_names[i] for i in op_indices}
            append_row(op_indices, func_names, stored=True)

        headers = [
            "Op Id(s)",
            "Op Name(s)",
            "Discarded Mem (%)",
            "Discarded Mem (B)",
            "Recomp time (%)",
            "Recomp time (ms)",
            "MSPS",
            "Always Stored",
            "Always Recomputed",
        ]
        if print_tabular:
            _display_stats_tabular(headers, table_data)
        else:
            max_widths = [0 for _ in range(len(headers))]
            table_data.insert(0, headers)
            for row in table_data:
                for i, elem in enumerate(row):
                    max_widths[i] = max(max_widths[i], len(elem))
            for row in table_data:
                print(
                    "\t".join(
                        [f"{elem:<{max_widths[i]}}" for i, elem in enumerate(row)]
                    )
                )

    def pwlf_sac_tradeoff_curve(
        self,
        n_segments: int = 2,
        save_tradeoff_graphs: bool = False,
    ) -> None:
        """
        Fits a piecewise linear function with the specified sumber of segments to the SAC trade-off curve of
        discarded memory vs recomputation time.

        Args:
            n_segments (int, optional): The number of segments to be used for fitting the piecewise linear function to
                the trade-off curve. Defaults to 2.
            save_tradeoff_graphs (bool, optional): Whether to save the trade-off graphs to file. Defaults to False.

        If save_tradeoff_graphs is True, the trade-off graphs are saved to file using the module FQN as the filename.
        """
        for mod_fqn, sac_stats in self.sac_mod_stats.items():
            self.sac_mod_tradeoff_stats[mod_fqn] = self._get_sac_tradeoff_pwlf_stats(
                sac_stats=sac_stats,
                greedy_order_meta=self.sac_mod_greedy_order_meta[mod_fqn],
                n_segments=n_segments,
                save_tradeoff_graph=save_tradeoff_graphs,
                filename=mod_fqn,
            )

    def display_modulewise_sac_stats(
        self, depth: int = 2, print_tabular: bool = False
    ) -> None:
        """
        Displays the SAC and trade-off statistics for each module.

        Args:
            depth (int, optional): The maximum depth of modules to display. Defaults to 2.
            print_tabular (bool, optional): Whether to print the statistics in a tabular format. Defaults to False.

        Prints:
            For each module with depth less than or equal to the specified depth:
            1. The SAC statistics for the module (using display_sac_stats).
            2. The SAC trade-off statistics for the module (using display_sac_tradeoff_stats).

        If print_tabular is True, the statistics are printed in a tabular format.
        Otherwise, the statistics are printed in a plain text format.
        """
        for mod_fqn, sac_stats in self.sac_mod_stats.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module: {mod_fqn}")
            self.display_sac_stats(sac_stats, print_tabular)
            print(f"AC Trade-off for Module: {mod_fqn} MSPS = Memory/Runtime")
            self.display_sac_tradeoff_stats(
                self.sac_mod_greedy_order_meta[mod_fqn], sac_stats, print_tabular
            )

    def __call__(self, estimate_mode_type: str) -> Self:
        """
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            SACEstimator: The SAC estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        """
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate_runtime = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate_runtime = RuntimeEstimator._roofline_estimate
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        return self

    def __enter__(self) -> Self:  # type: ignore[no-untyped-def]
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "SAC Estimator should be called in FakeTensorMode"
        RuntimeEstimator.fake_mode = fake_mode
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=self._pre_fw_hook,
            post_fw_hook=self._post_fw_hook,
        )
        self._mod_tracker.__enter__()
        self._saved_tensor_hook_ctx.__enter__()
        return super().__enter__()

    def __exit__(self, *args: Any) -> None:  # type: ignore[no-untyped-def]
        self._saved_tensor_hook_ctx.__exit__()
        self._mod_tracker.__exit__(*args)
        super().__exit__(*args)
