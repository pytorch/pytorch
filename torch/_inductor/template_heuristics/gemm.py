from __future__ import annotations

from typing import TYPE_CHECKING

from .. import config as inductor_config
from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from ..ir import Layout
    from ..kernel_inputs import KernelInputs


class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    def should_run(self, inputs: KernelInputs, layout: Layout) -> bool:
        """
        simple base override for GEMM family templates that run only in max-autotune
        """
        return inductor_config.max_autotune or inductor_config.max_autotune_gemm
