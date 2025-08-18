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
    _snodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
    peak_memory: int,
    _curr_memory: dict[BaseSchedulerNode, tuple[int, int]],
    snodes_allocfree: dict[BaseSchedulerNode, SNodeMemory],
    tlparse_name: str,
    gn_to_bufs_last_use: dict[
        BaseSchedulerNode, list[Union[FreeableInputBuffer, SchedulerBuffer]]
    ],
) -> bool:
    iterative_recompute_error = False
    candidate_allocfree = snodes_allocfree[candidate]
    _peak_memory, _snodes_curr_memory, _snodes_allocfree, _ = (
        estimate_peak_memory_allocfree(
            _snodes, name_to_freeable_input_buf, graph_outputs
        )
    )
    __curr_memory = dict(zip(_snodes, _snodes_curr_memory))
    iter_cm = _curr_memory[candidate]
    new_cm = __curr_memory[candidate]
    log = ""
    if _peak_memory > peak_memory:
        log = "ITERATIVE PEAK DOES NOT MATCH"
        iterative_recompute_error = True
    if iter_cm != new_cm:
        log = "ITERATIVE CURR MEMORY CANDIDATE DOES NOT MATCH"
        iterative_recompute_error = True
    for i, gn in enumerate(gns):
        iter_gnm = _curr_memory[gn]
        new_gnm = __curr_memory[gn]
        if iter_gnm != new_gnm:
            log = f"ITERATIVE GN CURR MEMORY DOES NOT MATCH:{gn.get_name()}"
            iterative_recompute_error = True
    if iterative_recompute_error:
        log += (
            f"\nCANDIDATE:{candidate.get_name()}"
            f"\nGROUP:{group_names}"
            f"\nPEAK_MEMORY_BEFORE:{peak_memory}"
            f"\nPEAK_MEMORY_AFTER_SWAP:{_peak_memory}"
            f"\nCANDIDATE:{candidate.debug_str()}"
            f"\nCANDIDATE_ITER_CURR_MEMORY:{iter_cm}"
            f"\nCANDIDATE_NEW__CURR_MEMORY:{new_cm}"
            f"\nCANDIDATE_ITER_ALLOCFREE:{candidate_allocfree}"
            f"\nCANDIDATE_NEW_ALLOCFREE:{_snodes_allocfree[candidate]}"
        )
        peak_log = ""
        for i, (pre, post) in enumerate(_snodes_curr_memory):
            if _peak_memory == pre:
                n = _snodes[i]
                peak_log = (
                    f"\nNEW_PEAK:{_peak_memory}(BASE:{peak_memory})"
                    f" @ SNODE[{i}/{len(_snodes)}]:{n.get_name()} {n.debug_str()}"
                )
                break
        group_log = ""
        for i, gn in enumerate(gns):
            iter_gnm = _curr_memory[gn]
            new_gnm = __curr_memory[gn]
            group_log += (
                f"\nGROUP_NODE[{i}]:{gn.debug_str()}"
                f"\nGROUP_NODE[{i}] ITER_GNM[{gn.get_name()}]:{iter_gnm}"
                f"\nGROUP_NODE[{i}] ESTM_GNM[{gn.get_name()}]:{new_gnm}"
                f"\nGROUP_NODE[{i}] ITER_allocfree:{_snodes_allocfree[gn]}"
                f"\nGROUP_NODE[{i}] ESTM_allocfree:{_snodes_allocfree[gn]}"
            )
        log += peak_log
        log += group_log
        log += f"\nGN_TO_BUFS_LAST_USE:{gn_to_bufs_last_use}"
        log += "\n\n".join(
            [
                (
                    f"\nSNODE[{i}]\n{n.debug_str()}"
                    f"\nITER_cur_mem:{_curr_memory[n]}"
                    f"\nESTM_cur_mem:{__curr_memory[n]}"
                    f"\nITER_allocfree:{snodes_allocfree[n]}"
                    f"\nESTM_allocfree:{_snodes_allocfree[n]}"
                )
                for i, n in enumerate(_snodes)
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
