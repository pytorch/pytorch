from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Union

from torch.utils._ordered_set import OrderedSet

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

    def to_bundle_dict(self) -> dict[str, Any]:
        """
        Serialize the KernelTemplateChoice to a bundle dictionary.

        Returns:
            Dictionary with 3 keys:
            - template_id: The UID of the template
            - params: Serializable dictionary from params.to_serializeable_dict()
            - extra_kwargs: The extra kwargs dictionary

        Note:
            This method only serializes the core template choice information.
            The layout and inputs are not included as they are expected to be
            available when deserializing (e.g., from the calling context).
        """
        return {
            "template_id": self.template.uid,
            "params": self.params.to_serializeable_dict(),
            "extra_kwargs": self.extra_kwargs.copy(),
        }

    @classmethod
    def from_bundle_dict(
        cls,
        bundle_dict: dict[str, Any],
        layout: Layout,
        inputs: KernelInputs,
    ) -> KernelTemplateChoice:
        """
        Deserialize a KernelTemplateChoice from a bundle dictionary.

        Args:
            bundle_dict: Dictionary with keys 'template_id', 'params', 'extra_kwargs'
            layout: Layout object for the template choice
            inputs: KernelInputs object for the template choice

        Returns:
            Reconstructed KernelTemplateChoice instance

        Raises:
            KeyError: If the template_id is not found in the registry
            KeyError: If required keys are missing from bundle_dict
        """
        from .template_heuristics.params import DictKernelTemplateParams
        from .template_heuristics.registry import get_template_by_uid

        # Validate bundle_dict keys
        required_keys = OrderedSet(["template_id", "params", "extra_kwargs"])
        missing_keys = required_keys - bundle_dict.keys()
        if missing_keys:
            raise KeyError(f"Missing required keys in bundle_dict: {missing_keys}")

        # Retrieve template from registry
        template = get_template_by_uid(bundle_dict["template_id"])

        # Reconstruct params - we use DictKernelTemplateParams as a fallback
        # since we don't know the original params class type from serialization
        # TODO(coconutruben): when we have more than DictKernelTemplateParams
        # we need to a way to find the right params again (likely a registry + type hint when serializing)
        params = DictKernelTemplateParams(bundle_dict["params"])

        return cls(
            template=template,
            params=params,
            extra_kwargs=bundle_dict["extra_kwargs"],
            layout=layout,
            inputs=inputs,
        )


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
