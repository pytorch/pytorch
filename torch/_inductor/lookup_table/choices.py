from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

from ..choices import InductorChoices
from ..kernel_template_choice import KernelTemplateChoice
from ..template_heuristics import get_template_heuristic
from .core import lookup_template_configs


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..codegen.common import KernelTemplate
    from ..ir import Layout
    from ..kernel_inputs import KernelInputs
    from ..select_algorithm import ExternKernelChoice


class LookupTableChoices(InductorChoices):
    """
    InductorChoices subclass that uses lookup table when available, otherwise falls back to parent.
    """

    def _adjust_mm_configs(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        layout: Layout,
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
        max_autotune: bool = False,
    ) -> list[KernelTemplateChoice]:
        """Check lookup table for hits, use those if found, otherwise fall back to parent."""
        # 1. Collect template src_hashes for validation
        template_uids = [template.uid for template in templates]
        template_hash_map = {}
        for template in templates:
            src_hash = getattr(template, "src_hash", None)
            template_hash_map[template.uid] = src_hash

        # 2. Single batch lookup for all templates
        lookup_results = lookup_template_configs(
            kernel_inputs, op_name, template_uids, template_hash_map
        )

        # 3. Early exit if no lookup table or no matches
        if not lookup_results:  # Empty dict
            return self._fallback(
                template_choices,
                kernel_inputs,
                layout,
                templates,
                op_name,
                kwarg_overrides,
                max_autotune,
            )

        # 4. Create KTCs only for templates with lookup entries
        return self._create_lookup_choices(
            lookup_results, templates, kernel_inputs, layout, op_name
        )

    def _fallback(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        layout: Layout,
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        op_name: str,
        kwarg_overrides: Optional[dict[str, dict[str, Any]]] = None,
        max_autotune: bool = False,
    ) -> list[KernelTemplateChoice]:
        """Fallback to parent if no lookup table or no matches."""
        # NOTE: this is broken out, so that subclasses are able to override this
        # to handle explicitly the situations where the lookup take had a miss vs
        # overriding the entire logic
        return super()._adjust_mm_configs(
            template_choices,
            kernel_inputs,
            layout,
            templates,
            op_name,
            kwarg_overrides,
            max_autotune,
        )

    def _create_lookup_choices(
        self,
        lookup_results: dict[str, list[dict[str, Any]]],
        templates: list[Union[KernelTemplate, ExternKernelChoice]],
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
    ) -> list[KernelTemplateChoice]:
        """Create KernelTemplateChoice objects from lookup results."""
        templates_by_uid = {template.uid: template for template in templates}
        device_type = kernel_inputs.device_type
        assert device_type is not None, "get_mm_configs requires a valid device type"
        lookup_choices: list[KernelTemplateChoice] = []

        for template_uid, configs in lookup_results.items():
            template = templates_by_uid[template_uid]
            heuristic = get_template_heuristic(template_uid, device_type, op_name)
            extra_kwargs = heuristic.get_extra_kwargs(kernel_inputs, layout, op_name)
            inputs_val = heuristic.adjust_kernel_inputs(kernel_inputs, op_name)

            for config in configs:
                ktc = KernelTemplateChoice(
                    template=template,
                    kwargs=config,
                    extra_kwargs=extra_kwargs,
                    layout=layout,
                    inputs=inputs_val,
                )
                lookup_choices.append(ktc)

        return lookup_choices
