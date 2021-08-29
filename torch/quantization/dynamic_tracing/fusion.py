from typing import List, Dict

import torch

from .utils import (
    SeenOp,
)

def _get_next_seen_ops(
    idx_to_seen_op: Dict[str, SeenOp],
    cur_seen_op: SeenOp,
) -> List[SeenOp]:
    """
    Input: cur_seen_op
    Output: list of all seen_ops which use the output of the cur_seen_op,
    """
    if len(cur_seen_op.output_tensor_ids) != 1:
        return []
    output_tensor_id = cur_seen_op.output_tensor_ids[0].id
    results = []
    for idx, seen_op in idx_to_seen_op.items():
        if len(seen_op.input_tensor_ids) != 1:
            continue
        input_tensor_id = seen_op.input_tensor_ids[0].id
        if output_tensor_id == input_tensor_id:
            results.append(seen_op)
    return results

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
    assert hasattr(module, '_auto_quant_state')
    qstate = module._auto_quant_state

    # TODO(future): reuse global mapping
    known_fusion_patterns = [
        (torch.nn.Conv2d, torch.nn.ReLU),
    ]

    # Walk the subgraphs and record the FQNs of all known module fusions.
    # For now, this is brute forced for simplicity, can be optimized later if
    # necessaary.
    results = []
    for idx, seen_op in qstate.idx_to_seen_ops.items():
        for fusion_pattern in known_fusion_patterns:
            cur_fqns = []
            cur_seen_op = seen_op
            is_match = True
            for mod_type in fusion_pattern:
                if cur_seen_op is not None and mod_type == cur_seen_op.type:
                    cur_fqns.append(cur_seen_op.fqn)
                    next_seen_ops = _get_next_seen_ops(
                        qstate.idx_to_seen_ops, cur_seen_op)
                    if len(next_seen_ops) == 1:
                        cur_seen_op = next_seen_ops[0]
                    else:
                        cur_seen_op = None
                    continue
                else:
                    is_match = False
                    break
            if is_match:
                results.append(cur_fqns)

    return results
