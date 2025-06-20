from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from .codegen.common import KernelTemplate
    from .ir import ChoiceCaller
    from .kernel_inputs import KernelInputs
    from .kernel_params.params import KernelTemplateParams

log = logging.getLogger(__name__)


class KernelChoice:
    """
    A class that encapsulates a kernel template, its parameters, and inputs
    to provide a unified interface for generating kernel choices.

    This class takes in:
    - KernelTemplate: The template used to generate the kernel
    - KernelParams: Parameters for the kernel template
    - KernelInputs: Input nodes for the kernel

    And exposes methods to:
    - Generate choices by calling maybe_append_choice on the template
    - Get template and input IDs for identification
    """

    def __init__(
        self,
        template: KernelTemplate,
        params: KernelTemplateParams,
        inputs: KernelInputs,
    ):
        """
        Initialize a KernelChoice with template, parameters, and inputs.

        Args:
            template: The KernelTemplate to use for generating choices
            params: The KernelTemplateParams containing kernel parameters
            inputs: The KernelInputs containing input nodes
        """
        self.template = template
        self.params = params
        self.inputs = inputs

    def caller(self) -> Optional[Any]:
        """
        Generate a caller by calling maybe_append_choice with an empty list
        from the template, passing in the KernelInputs.nodes() and the
        **KernelParams.kwargs().

        Returns:
            The caller that was appended to the list, or None if no caller was generated
        """
        choices: list[ChoiceCaller] = []
        self.maybe_append_choice(choices)

        # Return the caller that was appended, or None if no caller was added
        return choices[0] if choices else None

    def template_id(self) -> str:
        """
        Get the template ID.

        Returns:
            The ID of the kernel template
        """
        return self.template.id

    def inputs_id(self) -> str:
        """
        Get the inputs ID.

        Returns:
            A string representation of the input nodes for identification
        """
        return self.inputs.id

    def maybe_append_choice(
        self, choices: list[ChoiceCaller], **kwargs: dict[Any]
    ) -> None:
        """
        Append a choice to the choices list using the template's maybe_append_choice method.

        Args:
            choices: List to append the choice to
            **kwargs: Additional keyword arguments to pass to template.maybe_append_choice
        """
        error = self.template.maybe_append_choice(
            choices, input_nodes=self.inputs.nodes(), **self.params.kwargs(), **kwargs
        )

        if error is not None:
            log.debug(
                "Failed to append choice for template %s: %s", self.template.id, error
            )
