from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING, Union

import torch._inductor.config as config

from ..kernel_template_choice import make_ktc_generator
from ..lookup_table.choices import LookupTableChoices
from ..template_heuristics.registry import get_template_heuristic
from .core import predict_and_filter_choices
from .registry import get_model_function_for_key


if TYPE_CHECKING:
    from collections.abc import Generator

    from ..codegen.common import KernelTemplate
    from ..ir import Layout
    from ..kernel_inputs import KernelInputs
    from ..kernel_template_choice import KernelTemplateChoice
    from ..select_algorithm import ExternKernelChoice


log = logging.getLogger(__name__)


class PerformanceModelChoices(LookupTableChoices):
    """
    LookupTableChoices subclass that uses a performance model for kernel selection.
    """

    def _expand(
        self,
        template_choices: dict[str, Generator[KernelTemplateChoice, None, None]],
        kernel_inputs: KernelInputs,
        layout: Layout,
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]],
    ) -> tuple[list[KernelTemplateChoice], int]:
        """
        Expand templates to exhaustive search when performance models exist and return the full list.

        Args:
            template_choices: Original template choices generators
            kernel_inputs: Kernel inputs for the operation
            layout: Layout for the operation
            op_name: Name of the operation

        Returns:
            Tuple of:
            - Flattened list of all KernelTemplateChoice objects
            - Dictionary mapping template UIDs to original generator sizes
        """
        all_choices: list[KernelTemplateChoice] = []
        original_sizes = 0
        hardware_name = kernel_inputs.device_name

        # Assert that hardware_name is not None
        assert hardware_name is not None, (
            f"hardware_name must not be None, got {hardware_name}"
        )
        for template_uid, generator in template_choices.items():
            # Check if performance model exists for this template
            model_function = get_model_function_for_key(
                template_uid, op_name, hardware_name
            )
            # Actualize original generator to count size
            original_choices = list(generator)
            original_sizes += len(original_choices)
            if model_function is not None:
                log.debug(
                    "Performance model found for template_id=%r op_name=%r, expanding search space",
                    template_uid,
                    op_name,
                )

                # Create exhaustive generator by patching config
                with config.patch(max_autotune_gemm_search_space="EXHAUSTIVE"):
                    # Get the template heuristic for exhaustive generation
                    assert kernel_inputs.device_type is not None, "device_type is None"
                    heuristic = get_template_heuristic(
                        template_uid, kernel_inputs.device_type, op_name
                    )

                    # Get exhaustive configs from heuristic
                    exhaustive_configs = heuristic.get_template_configs(
                        kernel_inputs,
                        layout,
                        op_name,
                        # always in max-autotune mode here
                        max_autotune=True,
                    )
                    # Create new generator with exhaustive configs
                    # Need to reconstruct the template and other args from original choices
                    if original_choices:
                        # Extract template and kwargs from first choice
                        first_choice = original_choices[0]
                        template = first_choice.template
                        extra_kwargs = first_choice.extra_kwargs
                        # Create new generator with exhaustive configs
                        overrides = kwarg_overrides.get(template_uid, {})
                        # Add exhaustive choices directly to the list
                        all_choices.extend(
                            make_ktc_generator(
                                template=template,
                                cs=exhaustive_configs,
                                overrides=overrides,
                                extra_kwargs=extra_kwargs,
                                layout=layout,
                                inputs=kernel_inputs,
                            )
                        )

            else:
                log.debug(
                    "No performance model found for template_id=%r op_name=%r, keeping original search space",
                    template_uid,
                    op_name,
                )
                # Keep original choices - add directly to list
                all_choices.extend(original_choices)

        return all_choices, original_sizes

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
        """
        Performance model fallback implementation.

        Implements the performance model expansion logic:
        1. Early validation of config settings
        2. Expand templates to exhaustive search when performance models exist
        3. Handle topk normalization
        4. Call predict_and_filter_choices with actualized choices
        """
        # Validate topk configuration
        topk = config.performance_model.topk
        assert topk >= -1, "performance_model.topk must be >= -1, got " + str(topk)
        enabled = True
        # Early validation - check if performance model is enabled
        # Check if max_autotune flags are enabled
        if not (config.max_autotune or config.max_autotune_gemm):
            log.debug(
                "Performance model disabled: max_autotune and max_autotune_gemm both False. "
                "Falling back to parent implementation."
            )
            enabled = False
        # If topk == 0, performance model is disabled
        if topk == 0:
            log.debug(
                "Performance model disabled: topk=0. "
                "Falling back to parent implementation."
            )
            enabled = False
        if not enabled:
            return super()._fallback(
                template_choices,
                kernel_inputs,
                layout,
                templates,
                op_name,
                kwarg_overrides,
                max_autotune,
            )

        # Expand template choices using performance model logic
        kwarg_overrides = kwarg_overrides or {}
        actualized_choices, original_sizes = self._expand(
            template_choices, kernel_inputs, layout, op_name, kwarg_overrides
        )

        # Handle topk normalization: if topk == -1, set to total original size
        if topk == -1:
            topk = original_sizes
            log.debug("Set topk=-1 to original size %d", original_sizes)

        # Create template_uid_to_ktc mapping for predict_and_filter_choices
        template_uid_to_ktc: dict[str, list[KernelTemplateChoice]] = {}
        for choice in actualized_choices:
            if choice.template.uid not in template_uid_to_ktc:
                template_uid_to_ktc[choice.template.uid] = []
            template_uid_to_ktc[choice.template.uid].append(choice)
        # Apply performance model prediction and filtering
        filtered_choices = predict_and_filter_choices(
            ktc_stack=actualized_choices,
            template_uid_to_ktc=template_uid_to_ktc,
            op_name=op_name,
            topk=topk,
            discard_unranked=config.performance_model.discard_unranked,
        )
        log.info(
            "Performance model complete: orig=%d exp=%d filt=%d topk=%d",
            original_sizes,
            len(actualized_choices),
            len(filtered_choices),
            topk,
        )

        return filtered_choices
