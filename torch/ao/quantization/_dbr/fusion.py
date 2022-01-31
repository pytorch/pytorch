from typing import List

import torch

from .utils import (
    get_users_of_seen_op_info,
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

    TODO: test coverage

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
        for idx, seen_op_info in qstate.idx_to_seen_op_infos.items():
            for fusion_pattern in known_module_fusion_patterns:
                cur_fqns = []
                cur_seen_op_info = seen_op_info
                is_match = True
                for mod_type in fusion_pattern:
                    if cur_seen_op_info is not None and mod_type == cur_seen_op_info.type:
                        cur_fqns.append(cur_seen_op_info.fqn)
                        next_seen_op_infos = get_users_of_seen_op_info(
                            qstate.idx_to_seen_op_infos, cur_seen_op_info)
                        if len(next_seen_op_infos) == 1:
                            cur_seen_op_info = next_seen_op_infos[0]
                        else:
                            cur_seen_op_info = None
                        continue
                    else:
                        is_match = False
                        break
                if is_match:
                    # we check for existence to ensure the final fusion list
                    # is deduplicated, in case the same op is called multiple
                    # times in a single forward
                    if cur_fqns not in results:
                        results.append(cur_fqns)

    return results
