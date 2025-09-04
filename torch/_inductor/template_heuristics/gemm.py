from __future__ import annotations

from .. import config as inductor_config
from .base import TemplateConfigHeuristics


class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    @property
    def should_run(self) -> bool:
        """
        simple base override for GEMM family templates that run only in max-autotune
        """
        return inductor_config.max_autotune or inductor_config.max_autotune_gemm
