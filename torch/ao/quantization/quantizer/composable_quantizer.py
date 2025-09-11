from __future__ import annotations

from typing import TYPE_CHECKING

from .quantizer import QuantizationAnnotation, Quantizer


if TYPE_CHECKING:
    import torch
    from torch.fx import Node

__all__ = [
    "ComposableQuantizer",
]


class ComposableQuantizer(Quantizer):
    """
    ComposableQuantizer allows users to combine more than one quantizer into a single quantizer.
    This allows users to quantize a model with multiple quantizers. E.g., embedding quantization
    maybe supported by one quantizer while linear layers and other ops might be supported by another
    quantizer.

    ComposableQuantizer is initialized with a list of `Quantizer` instances.
    The order of the composition matters since that is the order in which the quantizers will be
    applies.
    Example:
    ```
    embedding_quantizer = EmbeddingQuantizer()
    linear_quantizer = MyLinearQuantizer()
    xnnpack_quantizer = (
        XNNPackQuantizer()
    )  # to handle ops not quantized by previous two quantizers
    composed_quantizer = ComposableQuantizer(
        [embedding_quantizer, linear_quantizer, xnnpack_quantizer]
    )
    prepared_m = prepare_pt2e(model, composed_quantizer)
    ```
    """

    def __init__(self, quantizers: list[Quantizer]):
        super().__init__()
        self.quantizers = quantizers
        self._graph_annotations: dict[Node, QuantizationAnnotation] = {}

    def _record_and_validate_annotations(
        self, gm: torch.fx.GraphModule, quantizer: Quantizer
    ) -> None:
        for n in gm.graph.nodes:
            if "quantization_annotation" in n.meta:
                # check if the annotation has been changed by
                # comparing QuantizationAnnotation object id
                if n in self._graph_annotations and (
                    id(self._graph_annotations[n])
                    != id(n.meta["quantization_annotation"])
                ):
                    raise RuntimeError(
                        f"Quantizer {quantizer.__class__.__name__} has changed annotations on node {n}"
                    )
                else:
                    self._graph_annotations[n] = n.meta["quantization_annotation"]
            else:
                if n in self._graph_annotations:
                    raise RuntimeError(
                        f"Quantizer {quantizer.__class__.__name__} has removed annotations on node {n}"
                    )

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        for quantizer in self.quantizers:
            quantizer.annotate(model)
            self._record_and_validate_annotations(model, quantizer)
        return model

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        for quantizer in self.quantizers:
            model = quantizer.transform_for_annotation(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
