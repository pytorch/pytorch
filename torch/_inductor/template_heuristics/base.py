from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .params import DictKernelTemplateParams, KernelTemplateParams


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..kernel_inputs import KernelInputs


class TemplateConfigHeuristics:
    """Base class for generating sets of configs for an associated template."""

    def should_run(self, inputs: KernelInputs) -> bool:
        """
        hookup to check whether the configs are right to run at all e.g. you can check
        max-autotune specific to your heuristic here or other things
        If this returns False, get_template_configs will yield no configs

        Args:
            inputs: KernelInputs
        """
        return True

    def get_template_configs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[KernelTemplateParams, None, None]:
        """
        Get template configs for the given inputs.

        Prefer to override the _get_template_configs_impl method
        to leverage things like should_run
        """
        if not self.should_run(kernel_inputs):
            return

        # Generate configs and fuse with extra_kwargs
        for config_dict in self._get_template_configs_impl(kernel_inputs, op_name):
            # Fuse extra_kwargs into config
            yield DictKernelTemplateParams(config_dict)

    def _get_template_configs_impl(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Get template configs for the given inputs.
        This is the main entry point for template-specific logic.
        """
        # base implementation yields no entries
        yield from []

    def get_extra_kwargs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> dict[str, Any]:
        """
        Get extra kwargs for the given inputs/op for the template.

        Use this to return kwargs that are needed for the template, but
        do not change depending on the config/choice, but are rather
        always the same, for all configs
        """
        return {}

    def adjust_kernel_inputs(
        self,
        kernel_inputs: KernelInputs,
        op_name: str,
    ) -> KernelInputs:
        """
        Adjust kernel inputs for the given inputs/op for the template.

        override this to adjust the kernel inputs e.g. (un)squeezing
        """
        return kernel_inputs
