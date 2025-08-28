"""Performance model interface for inductor."""

from .base import PerformanceModelFunction, predict
from .core import filter_and_sort_choices, predict_and_filter_choices
from .registry import (
    clear_registry,
    get_functions_for_templates,
    get_model_function_for_key,
    list_registered_models,
    register_performance_model,
    register_performance_model_fn,
)


__all__ = [
    "PerformanceModelFunction",
    "predict",
    "filter_and_sort_choices",
    "predict_and_filter_choices",
    "clear_registry",
    "get_functions_for_templates",
    "get_model_function_for_key",
    "list_registered_models",
    "register_performance_model",
    "register_performance_model_fn",
]
