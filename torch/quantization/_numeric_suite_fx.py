# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/ns/_numeric_suite_fx.py`, while adding an import statement
here.
"""

from torch.ao.ns._numeric_suite_fx import (
    RNNReturnType,
    OutputLogger,
    NSTracer,
    _extract_weights_one_model,
    _extract_weights_impl,
    extract_weights,
    _add_loggers_one_model,
    _add_loggers_impl,
    add_loggers,
    _extract_logger_info_one_model,
    extract_logger_info,
    _add_shadow_loggers_impl,
    add_shadow_loggers,
    extract_shadow_logger_info,
    extend_logger_results_with_comparison,
)
