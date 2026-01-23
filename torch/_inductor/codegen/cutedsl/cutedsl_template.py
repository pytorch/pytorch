# mypy: allow-untyped-defs
import functools
import itertools
from collections.abc import Iterable
from typing import Any, Optional
from unittest.mock import patch

from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V
from torch._logging import getArtifactLogger

from ...autotune_process import CuteDSLBenchmarkRequest, TensorMeta
from ...ir import Buffer, ChoiceCaller, CuteDSLTemplateBuffer, IRNode, Layout, TensorBox
from ..common import KernelTemplate
from .cutedsl_kernel import CuteDSLTemplateKernel


log = getArtifactLogger(__name__, "output_code")


class CuteDSLTemplate(KernelTemplate):
    """Template for generating CuteDSL (CUTLASS Python DSL) kernels."""

    kernel_type: type[Any] = CuteDSLTemplateKernel
    index_counter = itertools.count()
    all_templates: dict[str, "CuteDSLTemplate"] = {}

    def __init__(
        self,
        name: str,
        source: str,
        subgraph_fn: Optional[Any] = None,
        mask_fn: Optional[Any] = None,
    ) -> None:
        super().__init__(name)
        self.source = source
        self.subgraph_fn = subgraph_fn
        self.mask_fn = mask_fn
        self.template = CuteDSLTemplate._template_from_string(source)
        assert name not in self.all_templates, f"duplicate template name, {name}"
        CuteDSLTemplate.all_templates[name] = self

    @staticmethod
    @functools.lru_cache(None)
    # pyrefly: ignore [bad-override]
    def _template_from_string(source: str) -> Any:
        return KernelTemplate._template_from_string(source)

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> Optional[NotImplementedError]:
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.
        """
        try:
            choices.append(self.generate(**kwargs))
            return None
        except NotImplementedError as e:
            log.debug("CuteDSL template choice generation failed: %s", e)  # noqa: G200
            return e
        except Exception as e:
            log.debug("CuteDSL template choice generation error: %s", e)  # noqa: G200
            return NotImplementedError(f"CuteDSL template failed: {e}")

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        """Generate the CuteDSL kernel caller."""
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        mutated_inputs = kwargs.pop("mutated_inputs", None)
        subgraphs = kwargs.pop("subgraphs", None)

        kernel_name = f"cutedsl_{self.name}_{next(self.index_counter)}"

        if self.template is None:
            raise RuntimeError("Template compilation failed (Jinja2 required)")

        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        # Patch V.graph.get_dtype to handle the fake buf_out buffer
        with patch.object(
            V.graph, "get_dtype", KernelTemplate._fake_get_dtype(self.output_node)
        ):
            kernel = self.kernel_type(
                kernel_name=kernel_name,
                input_nodes=input_nodes,
                output_node=self.output_node,
                subgraphs=subgraphs,
            )
            code = kernel.render(self.template, **kwargs)

            log.debug("Generated CuteDSL Code:\n%s", code)

            bmreq = CuteDSLBenchmarkRequest(
                kernel_name=kernel_name,
                input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
                output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
                extra_args=tuple(),
                source_code=code,
            )

            def make_kernel_render(out_node, hint_override: Optional[int] = None):
                """
                Factory function that creates a kernel renderer for the final output.

                This closure captures the current template and parameters, but allows
                the output node to be specified later. This is used during the final
                kernel selection phase when the actual output buffer is available.
                """
                render_kernel = self.kernel_type(
                    kernel_name=str(Placeholder.KERNEL_NAME),
                    input_nodes=input_nodes,
                    output_node=out_node,
                    subgraphs=subgraphs,
                )

                def render():
                    return render_kernel.render(self.template, **kwargs)

                return render_kernel, render

            return CuteDSLTemplateCaller(
                name=kernel_name,
                input_nodes=input_nodes,
                layout=layout,
                make_kernel_render=make_kernel_render,
                bmreq=bmreq,
                template=self,
                mutated_inputs=mutated_inputs,
            )


class CuteDSLTemplateCaller(ChoiceCaller):
    """Caller for CuteDSL templates that integrates with the autotuning system."""

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Any,
        bmreq: CuteDSLBenchmarkRequest,
        template: "CuteDSLTemplate",
        mutated_inputs: Optional[Iterable[IRNode]] = None,
    ):
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"CuteDSL template {name}",
        )
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.mutated_inputs = mutated_inputs

    def __str__(self) -> str:
        return f"CuteDSLTemplateCaller({self.name})"

    def benchmark(self, *args, out) -> float:
        """Benchmark the kernel execution."""
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        """Create the output node for this template choice."""
        buffer = CuteDSLTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            mutated_inputs=self.mutated_inputs,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return TensorBox.create(buffer)

    def call_name(self) -> str:
        """Return the kernel call name."""
        return self.name

    def to_callable(self) -> Any:
        """Return callable that can execute this kernel."""
        return self.make_kernel_render

    def hash_key(self) -> str:
        """Return unique hash key for this choice."""
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def info_dict(self) -> dict[str, Any]:
        """Return information about this kernel."""
        return {
            "name": self.name,
            "backend": "CuteDSL",
            "template": self.template.name,
        }
