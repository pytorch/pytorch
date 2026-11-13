# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/ns/_numeric_suite_fx.py`, while adding an import statement
here.
"""

from torch.ao.ns._numeric_suite_fx import (
    _add_loggers_impl,
    _add_loggers_one_model,
    _add_shadow_loggers_impl,
    _extract_logger_info_one_model,
    _extract_weights_impl,
    _extract_weights_one_model,
    add_loggers,
    add_shadow_loggers,
    extend_logger_results_with_comparison,
    extract_logger_info,
    extract_shadow_logger_info,
    extract_weights,
    NSTracer,
    OutputLogger,
    RNNReturnType,
)
