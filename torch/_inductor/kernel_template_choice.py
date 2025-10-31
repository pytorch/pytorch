from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

from .template_heuristics.params import DictKernelTemplateParams


if TYPE_CHECKING:
    from collections.abc import Generator

    from .codegen.common import KernelTemplate
    from .ir import ChoiceCaller, Layout
    from .kernel_inputs import KernelInputs
    from .select_algorithm import ExternKernelChoice
    from .template_heuristics.params import KernelTemplateParams


class KernelTemplateChoice:
    """
    A class that encapsulates all the components needed to create a ChoiceCaller from a template.

    This class implements lazy evaluation for the choice property - the actual ChoiceCaller
    is only created when first accessed via the choice property.
    """

    def __init__(
        self,
        template: Union[KernelTemplate, ExternKernelChoice],
        params: KernelTemplateParams,
        extra_kwargs: dict[str, Any],
        layout: Layout,
        inputs: KernelInputs,
    ):
        self.template = template
        self.params = params
        self.extra_kwargs = extra_kwargs
        self.layout = layout
        self.inputs = inputs
        self.annotations: dict[str, Any] = {"ktc": self}

    @property
    def choice(self) -> Optional[ChoiceCaller]:
        """
        Lazily evaluate and return the ChoiceCaller for this template choice.

        On first access, calls template.choice_or_none() with the stored parameters.
        If successful, caches and returns the ChoiceCaller. If it fails, caches
        and returns None. Subsequent accesses return the cached value.

        Returns:
            ChoiceCaller if the template choice succeeds, None otherwise
        """
        if not hasattr(self, "_choice"):
            # First time accessing choice - try to generate it
            kwargs = self.params.to_kwargs()
            self._choice = self.template.choice_or_none(
                **kwargs,
                **self.extra_kwargs,
                layout=self.layout,
                input_nodes=self.inputs.nodes(),
            )
            if self._choice is not None:
                self._choice.annotations = self.annotations
        return self._choice


def make_ktc_generator(
    template: Union[KernelTemplate, ExternKernelChoice],
    cs: Generator[KernelTemplateParams, None, None],
    extra_kwargs: dict[str, Any],
    overrides: dict[str, Any],
    layout: Layout,
    inputs: KernelInputs,
) -> Generator[KernelTemplateChoice, None, None]:
    """
    Create a generator of KernelTemplateChoice objects for a given template.

    Args:
        template: The template object (KernelTemplate or ExternKernelChoice)
        cs: Generator of KernelTemplateParams from template heuristic
        overrides: Override kwargs for the template
        layout: Layout value for the template
        inputs: KernelInputs for the op

    Yields:
        KernelTemplateChoice objects
    """
    for params in cs:
        # Apply overrides to params
        base_kwargs = params.to_kwargs()
        final_kwargs = {**base_kwargs, **overrides}
        final_params = DictKernelTemplateParams(final_kwargs)
        yield KernelTemplateChoice(
            template=template,
            params=final_params,
            extra_kwargs=extra_kwargs,
            layout=layout,
            inputs=inputs,
        )
