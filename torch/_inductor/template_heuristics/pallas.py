from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import TemplateConfigHeuristics
from .registry import register_template_heuristic


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..kernel_inputs import KernelInputs


@register_template_heuristic("pallas_tpu_block_mm", "cpu", op_name="mm")
class PallasMatmulHeuristic(TemplateConfigHeuristics):
    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        tpu_configs = [
            {"bm": 128, "bk": 128, "bn": 128},
        ]
        yield from tpu_configs
