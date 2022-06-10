from typing import List

import torch

from .function_fusion import pattern_is_match

from .utils import (
    get_users_of_seen_q_op_info,
)

from .mappings import (
    known_module_fusion_patterns,
)

def get_module_fusion_fqns(
    module: torch.nn.Module,
) -> List[List[str]]:
    """
    Input: a module with auto quantization state

    Walks the subgraphs and determines which modules should be
    fused.

    Output: a list of FQNs of modules which should be fused.
    """
    results = []
    for _, child in module.named_modules():
        if not hasattr(child, '_auto_quant_state'):
            continue
        qstate = child._auto_quant_state

        # Walk the subgraphs and record the FQNs of all known module fusions.
        # For now, this is brute forced for simplicity, can be optimized later if
        # necessary.
        # TODO(future PR): if a pattern is matched, add it to "seen" items
        # and do not use it in future matching.
        for idx, seen_q_op_info in qstate.idx_to_seen_q_op_infos.items():
            for fusion_pattern in known_module_fusion_patterns:
                is_match = pattern_is_match(
                    fusion_pattern, seen_q_op_info, qstate.idx_to_seen_q_op_infos)
                if is_match:
                    cur_fqns = [seen_q_op_info.fqn]
                    cur_seen_q_op_info = seen_q_op_info
                    for _element in fusion_pattern[:-1]:
                        users = get_users_of_seen_q_op_info(
                            qstate.idx_to_seen_q_op_infos, cur_seen_q_op_info)
                        cur_seen_q_op_info = users[0]
                        cur_fqns.append(cur_seen_q_op_info.fqn)

                    # we check for existence to ensure the final fusion list
                    # is deduplicated, in case the same op is called multiple
                    # times in a single forward
                    if cur_fqns not in results:
                        results.append(cur_fqns)

    return results
