from typing import Dict, Tuple, Callable, Optional

from .mappings import known_function_fusion_patterns_and_replacements
from .utils import (
    FusionInfo,
    SeenQOpInfo,
    get_users_of_seen_q_op_info,
)

def _identity(x):
    return x

def pattern_is_match(
    fusion_pattern: Tuple[Callable, ...],
    cur_seen_q_op_info: Optional[SeenQOpInfo],
    idx_to_seen_q_op_infos: Dict[int, SeenQOpInfo],
) -> bool:
    is_match = True
    for el_type in fusion_pattern:
        if cur_seen_q_op_info is not None and el_type == cur_seen_q_op_info.type:
            next_seen_q_op_infos = get_users_of_seen_q_op_info(
                idx_to_seen_q_op_infos, cur_seen_q_op_info)
            if len(next_seen_q_op_infos) == 1:
                cur_seen_q_op_info = next_seen_q_op_infos[0]
            else:
                cur_seen_q_op_info = None
            continue
        else:
            is_match = False
            break
    return is_match

def match_fusion_patterns(
    idx_to_seen_q_op_infos: Dict[int, SeenQOpInfo],
):
    """
    Matches fusion patterns to elements of `idx_to_seen_q_op_infos`.
    Modifies them inplace if matches are found.

    Note:
    1. The matching is local to the ops seen by a single parent module,
       it does not cross module boundaries. This is for simplicity, and
       there are no plans to relax this at the moment.
    2. The matching only supports linear patterns of ops where all of
       of the arguments needed to execute the fusion are passed to the first
       op in the sequence. This is for simplicity, and can be relaxed
       in a future PR if there is a need.
    3. Currently the matching does not look at non quantizeable ops,
       this will be fixed in the next PR.
    """

    # Walk the subgraphs and find the function fusions. For now, this is
    # brute forced for simplicity, can be optimized later if necessary.
    for idx, seen_q_op_info in idx_to_seen_q_op_infos.items():
        for fusion_pattern, replacement in \
                known_function_fusion_patterns_and_replacements.items():
            is_match = pattern_is_match(
                fusion_pattern, seen_q_op_info, idx_to_seen_q_op_infos)
            if not is_match:
                continue

            cur_seen_q_op_info = seen_q_op_info
            for idx in range(len(fusion_pattern)):
                if idx > 0:
                    users = get_users_of_seen_q_op_info(
                        idx_to_seen_q_op_infos, cur_seen_q_op_info)
                    cur_seen_q_op_info = users[0]

                is_first_element = idx == 0
                is_last_element = idx == len(fusion_pattern) - 1
                replacement_type = replacement if is_first_element \
                    else _identity
                fusion_info = FusionInfo(
                    fusion_pattern, replacement_type, is_first_element,
                    is_last_element)
                cur_seen_q_op_info.fusion_info = fusion_info
            break
