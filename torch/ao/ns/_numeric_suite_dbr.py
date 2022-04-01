"""
Numeric Suite Core APIs for define-by-run quantization.

Experimental, API may change at any time.
"""

import functools
from typing import Tuple, Any, Optional, List, Dict

import torch

from torch.ao.quantization._dbr.quantization_state import (
    AutoQuantizationState,
)

def _turn_on_loggers(name: str, model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, AutoQuantizationState):
            module.logging_model_name = name
            module.log_op_outputs = True

def add_loggers(
    name_a: str,
    model_a: torch.nn.Module,
    name_b: str,
    model_b: torch.nn.Module,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Enables intermediate activation logging on model_a and model_b.
    """
    _turn_on_loggers(name_a, model_a)
    _turn_on_loggers(name_b, model_b)
    return model_a, model_b

def _extract_logger_info_one_model(model: torch.nn.Module) -> Tuple[str, Any]:
    results: Optional[List[List[Any]]] = None
    model_name = None
    for _, module in model.named_modules():
        if isinstance(module, AutoQuantizationState):
            if results is None:
                # initialize results to the right length
                results = [[] for i in range(len(module.op_outputs))]
            assert results is not None

            if model_name is None:
                # model_name is the same everywhere in this model, take
                # the first one
                model_name = module.logging_model_name

            for forward_idx, outputs in enumerate(module.op_outputs):
                results[forward_idx].extend(outputs)

    # sort each forward's results by global idx
    assert results is not None
    assert model_name is not None
    for result_idx, result in enumerate(results):
        result.sort(key=functools.cmp_to_key(  # type: ignore[misc]
            lambda a, b: 1 if a[0] > b[0] else -1))  # type: ignore[index]

    return model_name, results

def extract_logger_info(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    model_name_to_use_for_layer_names: str,
) -> Any:
    """
    Extracts intermediate activations from model_a and model_b.
    """

    model_name_a, results_a = _extract_logger_info_one_model(model_a)
    model_name_b, results_b = _extract_logger_info_one_model(model_b)
    assert len(results_a) == len(results_b), 'results length mismatch'
    results: Dict[str, Any] = {}
    if len(results_a) == 0:
        return results

    for op_idx in range(len(results_a[0])):
        # currently using global_idx for layer_name
        layer_name = (
            results_a[0][op_idx][0]
            if model_name_to_use_for_layer_names == model_name_a
            else results_a[0][op_idx][0])

        values_a = [results_a[forward_idx][op_idx][3]
                    for forward_idx in range(len(results_a))]
        values_b = [results_b[forward_idx][op_idx][3]
                    for forward_idx in range(len(results_b))]
        node_output = {
            model_name_a: [{
                'type': 'node_output',
                'values': values_a,
                'ref_node_target_type': str(results_a[0][op_idx][2]),
                'fqn': str(results_a[0][op_idx][1]),
                'index_of_arg': 0,
                'index_within_arg': 0,
            }],
            model_name_b: [{
                'type': 'node_output',
                'values': values_b,
                'ref_node_target_type': str(results_b[0][op_idx][2]),
                'fqn': str(results_b[0][op_idx][1]),
                'index_of_arg': 0,
                'index_within_arg': 0,
            }],
        }

        results[layer_name] = {
            'node_output': node_output,
        }

    return results
