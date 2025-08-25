from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import TemplateConfigHeuristics


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..ir import Layout
    from ..kernel_inputs import KernelInputs


class ATenConfigHeuristics(TemplateConfigHeuristics):
    """
    Pseudo heuristic to make ATen choices go through the same flow as other templates

    This is a single choice without kwargs

    If you want to use this with an ATen choice that has kwargs, just subclass
    """

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        yield dict()
