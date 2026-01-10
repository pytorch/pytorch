from __future__ import annotations

from typing import TYPE_CHECKING, Union

from torch._logging import trace_structured
from .memory import estimate_peak_memory_allocfree


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet
    from .memory import FreeableInputBuffer, SNodeMemory
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


def _debug_iterative_memory_recompute(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_names: str,
    snodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
    peak_memory: int,
    iter_curr_memory: dict[BaseSchedulerNode, tuple[int, int]],
    snodes_allocfree: dict[BaseSchedulerNode, SNodeMemory],
    tlparse_name: str,
    gn_to_bufs_last_use: dict[
        BaseSchedulerNode, list[Union[FreeableInputBuffer, SchedulerBuffer]]
    ],
) -> bool:
    iterative_recompute_error = False
    candidate_allocfree = snodes_allocfree[candidate]
    est_peak_memory, snodes_curr_memory, snodes_allocfree, _ = (
        estimate_peak_memory_allocfree(
            snodes, name_to_freeable_input_buf, graph_outputs
        )
    )
    est_curr_memory = dict(zip(snodes, snodes_curr_memory))
    iter_cm = iter_curr_memory[candidate]
    new_cm = est_curr_memory[candidate]
    log = ""
    if est_peak_memory > peak_memory:
        log = "ITERATIVE PEAK DOES NOT MATCH"
        iterative_recompute_error = True
    if iter_cm != new_cm:
        log = "ITERATIVE CURR MEMORY CANDIDATE DOES NOT MATCH"
        iterative_recompute_error = True
    for gn in gns:
        iter_gnm = iter_curr_memory[gn]
        new_gnm = est_curr_memory[gn]
        if iter_gnm != new_gnm:
            log = f"ITERATIVE GN CURR MEMORY DOES NOT MATCH:{gn.get_name()}"
            iterative_recompute_error = True
    if iterative_recompute_error:
        log += (
            f"\nCANDIDATE:{candidate.get_name()}"
            f"\nGROUP:{group_names}"
            f"\nPEAK_MEMORY_BEFORE:{peak_memory}"
            f"\nPEAK_MEMORY_AFTER_SWAP:{est_peak_memory}"
            f"\nCANDIDATE:{candidate.debug_str()}"
            f"\nCANDIDATE_ITER_CURR_MEMORY:{iter_cm}"
            f"\nCANDIDATE_NEW__CURR_MEMORY:{new_cm}"
            f"\nCANDIDATE_ITER_ALLOCFREE:{candidate_allocfree}"
            f"\nCANDIDATE_NEW_ALLOCFREE:{snodes_allocfree[candidate]}"
        )
        peak_log = ""
        for i, (pre, _post) in enumerate(snodes_curr_memory):
            if est_peak_memory == pre:
                n = snodes[i]
                peak_log = (
                    f"\nNEW_PEAK:{est_peak_memory}(BASE:{peak_memory})"
                    f" @ SNODE[{i}/{len(snodes)}]:{n.get_name()} {n.debug_str()}"
                )
                break
        group_log = ""
        for i, gn in enumerate(gns):
            iter_gnm = iter_curr_memory[gn]
            new_gnm = est_curr_memory[gn]
            group_log += (
                f"\nGROUP_NODE[{i}]:{gn.debug_str()}"
                f"\nGROUP_NODE[{i}] ITER_GNM[{gn.get_name()}]:{iter_gnm}"
                f"\nGROUP_NODE[{i}] ESTM_GNM[{gn.get_name()}]:{new_gnm}"
                f"\nGROUP_NODE[{i}] ITER_allocfree:{snodes_allocfree[gn]}"
                f"\nGROUP_NODE[{i}] ESTM_allocfree:{snodes_allocfree[gn]}"
            )
        log += peak_log
        log += group_log
        log += f"\nGN_TO_BUFS_LAST_USE:{gn_to_bufs_last_use}"
        log += "\n\n".join(
            [
                (
                    f"\nSNODE[{i}]\n{n.debug_str()}"
                    f"\nITER_cur_mem:{iter_curr_memory[n]}"
                    f"\nESTM_cur_mem:{est_curr_memory[n]}"
                    f"\nITER_allocfree:{snodes_allocfree[n]}"
                    f"\nESTM_allocfree:{snodes_allocfree[n]}"
                )
                for i, n in enumerate(snodes)
            ]
        )
        tname = f"{tlparse_name}_ITERATIVE_RECOMPUTE_ERROR"
        print(f"{tname}:\n{log}")
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": tname,
                "encoding": "string",
            },
            payload_fn=lambda: log,
        )
    return iterative_recompute_error
