from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Generator

    from .codegen.common import KernelTemplate
    from .ir import ChoiceCaller, Layout
    from .kernel_inputs import KernelInputs
    from .select_algorithm import ExternKernelChoice


class KernelTemplateChoice:
    """
    A class that encapsulates all the components needed to create a ChoiceCaller from a template.

    This class implements lazy evaluation for the choice property - the actual ChoiceCaller
    is only created when first accessed via the choice property.
    """

    def __init__(
        self,
        template: Union[KernelTemplate, ExternKernelChoice],
        kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
        layout: Layout,
        inputs: KernelInputs,
    ):
        self.template = template
        self.kwargs = kwargs
        self.extra_kwargs = extra_kwargs
        self.layout = layout
        self.inputs = inputs

    @property
    def choice(self) -> Optional[ChoiceCaller]:
        """
        Lazily evaluate and return the ChoiceCaller for this template choice.

        On first access, calls template.choice_or_None() with the stored parameters.
        If successful, caches and returns the ChoiceCaller. If it fails, caches
        and returns None. Subsequent accesses return the cached value.

        Returns:
            ChoiceCaller if the template choice succeeds, None otherwise
        """
        if not hasattr(self, "_choice"):
            # First time accessing choice - try to generate it
            self._choice = self.template.choice_or_None(
                **self.kwargs,
                layout=self.layout,
                input_nodes=self.inputs.nodes(),
                **self.extra_kwargs,
            )
        return self._choice


def make_ktc_generator(
    template: Union[KernelTemplate, ExternKernelChoice],
    cs: Generator[dict[str, Any], None, None],
    overrides: dict[str, Any],
    extra_kwargs: dict[str, Any],
    layout: Layout,
    inputs: KernelInputs,
) -> Generator[KernelTemplateChoice, None, None]:
    """
    Create a generator of KernelTemplateChoice objects for a given template.

    Args:
        template: The template object (KernelTemplate or ExternKernelChoice)
        cs: Generator of configurations from template heuristic
        overrides: Override kwargs for the template
        extra_kwargs: Extra kwargs from the heuristic
        layout_val: Layout value for the template
        inputs: KernelInputs for the op

    Yields:
        KernelTemplateChoice objects
    """
    for c in cs:
        yield KernelTemplateChoice(
            template=template,
            kwargs={**c, **overrides},
            extra_kwargs=extra_kwargs,
            layout=layout,
            inputs=inputs,
        )
