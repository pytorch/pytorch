from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from .kernel_inputs import KernelInputs  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Generator


class TemplateConfigProcessor(ABC):
    """
    Base class for template configuration processors.

    Processors can modify, filter, or replace template configurations
    as they flow through the processing pipeline.
    """

    @abstractmethod
    def process(
        self,
        configs: Generator[dict[str, Any], None, None],
        kernel_inputs: KernelInputs,
        layout: Any,
        op_name: str,
        template_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Process template configurations.

        Args:
            configs: Generator of template parameter dictionaries
            kernel_inputs: KernelInputs containing input tensor nodes and matrix indices
            layout: Output layout
            template_name: Template name (e.g., "bmm", "mm", "mm_persistent_tma")
            op_name: Operation name (e.g., "bmm", "baddbmm", "addmm", "mm_plus_mm")

        Yields:
            Processed template parameter dictionaries
        """
        ...
