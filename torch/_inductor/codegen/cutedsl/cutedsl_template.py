# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable, Optional, Union

import torch
from torch._inductor.codecache import PyCodeCache
from torch._inductor.ir import ShapeAsConstantBuffer
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.utils import Placeholder
from torch._logging import getArtifactLogger

from ...autotune_process import BenchmarkRequest, GPUDeviceBenchmarkMixin, TensorMeta
from ...ir import Buffer, ChoiceCaller, CuteDSLTemplateBuffer, Layout, TensorBox
from ..common import KernelTemplate
from .cutedsl_kernel import CuteDSLTemplateKernel


log = getArtifactLogger(__name__, "output_code")


class CuteDSLBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for CuteDSL (CUTLASS Python DSL) kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, list[TensorMeta]],
        extra_args: tuple[Any, ...],
        source_code: PartialRender,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)

        finalized_code = source_code.finalize_all()
        self.module_cache_key, self.module_path = PyCodeCache.write(finalized_code)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, out: torch.Tensor
    ) -> Callable[[], None]:
        """
        Create a function to run the CuteDSL kernel with the given input and output tensors.
        Similar to TritonBenchmarkRequest.make_run_fn but for CuteDSL kernels.
        """
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)

        # Logic replicated async_compile
        from .cutedsl_kernel import MAIN_SUFFIX

        main_func_name = f"{self.kernel_name}_{MAIN_SUFFIX}"

        if not hasattr(mod, main_func_name):
            available = [name for name in dir(mod) if callable(getattr(mod, name))]
            raise RuntimeError(
                f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
            )

        kernel_func = getattr(mod, main_func_name)

        def run_kernel():
            return kernel_func(*input_tensors, out)

        return run_kernel

    def cleanup_run_fn(self) -> None:
        """Clean up any resources used by the kernel."""


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
            log.debug("CuteDSL template choice generation failed: %s", e)
            return e
        except Exception as e:
            log.debug("CuteDSL template choice generation error: %s", e)
            return NotImplementedError(f"CuteDSL template failed: {e}")

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        """Generate the CuteDSL kernel caller."""
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")

        kernel_name = f"cutedsl_{self.name}_{next(self.index_counter)}"

        if self.template is None:
            raise RuntimeError("Template compilation failed (Jinja2 required)")

        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)

        kernel = self.kernel_type(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=self.output_node,
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
            render_kernel = self.kernel_type(
                kernel_name=str(Placeholder.KERNEL_NAME),
                input_nodes=input_nodes,
                output_node=out_node,
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

    def __str__(self) -> str:
        return f"CuteDSLTemplateCaller({self.name})"

    def benchmark(self, *args, out) -> float:
        """Benchmark the kernel execution."""
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> Union[TensorBox, ShapeAsConstantBuffer]:
        """Create the output node for this template choice."""
        return TensorBox.create(
            CuteDSLTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                template=self.template,
            )
        )

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
