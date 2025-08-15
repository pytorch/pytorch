"""
Template configuration processor that prioritizes lookup table results over heuristics.

Implements a processor that intercepts template configuration generation for autotuning.
When lookup table contains pre-optimized configurations for the given template/operation
combination, this processor discards heuristic-generated configs and yields lookup table
results instead. Falls back to heuristic configs when the lookup table is not in use.

Used as the default processor in InductorChoices for CUDA template autotuning optimization.
"""

import logging
from collections.abc import Generator
from typing import Any

from .ir import Layout
from .kernel_inputs import KernelInputs
from .lookup_table import lookup_template_configs
from .template_config_processor import TemplateConfigProcessor


log = logging.getLogger(__name__)


class LookupTableProcessor(TemplateConfigProcessor):
    """
    Template config processor that wraps the lookup table functionality.

    When the lookup table is in use and has configs for the given template/op combination,
    this processor will discard the incoming configs and yield lookup table results instead.
    Otherwise, it passes through the incoming configs unchanged.
    """

    def process(
        self,
        configs: Generator[dict[str, Any], None, None],
        kernel_inputs: KernelInputs,
        layout: Layout,  # Unused by lookup table but required by interface
        op_name: str,
        template_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Process template configurations using lookup table if available.

        Args:
            kernel_inputs: KernelInputs containing input tensor nodes and matrix indices
            layout: Output layout (unused by lookup table but required by interface)
            template_name: Template name (e.g., "bmm", "mm", "mm_persistent_tma")
            op_name: Operation name (e.g., "bmm", "baddbmm", "addmm", "mm_plus_mm")
            configs: Generator of template parameter dictionaries from heuristics

        Yields:
            Template parameter dictionaries from lookup table if available,
            otherwise passes through the incoming configs.
        """
        _ = layout  # Unused but required by interface

        # Try lookup table first
        input_nodes = kernel_inputs.nodes()
        cgen: Generator[dict[str, Any], None, None] = configs
        lookup_configs = lookup_template_configs(input_nodes, op_name, template_name)
        if lookup_configs is not None:
            # lookup table is in use
            cgen = (c for c in lookup_configs)

        yield from cgen
