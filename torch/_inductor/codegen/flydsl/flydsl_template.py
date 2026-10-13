# mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

from torch._inductor.utils import OrderedSet, Placeholder
from torch._inductor.virtualized import V
from torch._logging import getArtifactLogger

from ...autotune_process import FlyDSLBenchmarkRequest, TensorMeta
from ...ir import Buffer, ChoiceCaller, FlyDSLTemplateBuffer, IRNode, Layout, TensorBox
from ..common import KernelTemplate
from . import flydsl_utils
from .flydsl_kernel import FlyDSLTemplateKernel


log = logging.getLogger(__name__)
output_code_log = getArtifactLogger(__name__, "output_code")


class FlyDSLTemplate(KernelTemplate):
    """Template implementation for generating and compiling FlyDSL kernels."""

    kernel_type: type[Any] = FlyDSLTemplateKernel
    caller_type: type[Any] | None = None
    index_counter = itertools.count()
    all_templates: dict[str, "FlyDSLTemplate"] = {}

    def __init__(self, name: str, source: str) -> None:
        super().__init__(name)
        self.source = source
        self.template = FlyDSLTemplate._template_from_string(source)
        existing = self.all_templates.get(name)
        if existing is not None and existing.source != source:
            raise AssertionError(f"duplicate template name, {name}")
        FlyDSLTemplate.all_templates[name] = self

    @staticmethod
    @functools.lru_cache(None)
    # pyrefly: ignore [bad-override]
    def _template_from_string(source: str) -> Any:
        return KernelTemplate._template_from_string(source)

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> NotImplementedError | None:
        if getattr(V.graph, "cpp_wrapper", False) or getattr(
            V.graph, "aot_mode", False
        ):
            return NotImplementedError("FlyDSL does not support AOT wrappers")
        if not flydsl_utils.runtime_available():
            return NotImplementedError("FlyDSL runtime is unavailable")
        try:
            choices.append(self.generate(**kwargs))
            return None
        except NotImplementedError as e:
            log.debug("FlyDSL template choice generation failed: %s", e)
            return e
        except Exception as e:
            log.debug("FlyDSL template choice generation error: %s", e, exc_info=True)
            return NotImplementedError(f"FlyDSL template failed: {e}")

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        mutated_inputs = kwargs.pop("mutated_inputs", None)
        template_kwargs = dict(kwargs)
        benchmark_kernel_name = f"flydsl_{self.name}"
        choice_name = f"{benchmark_kernel_name}_{next(self.index_counter)}"

        if self.template is None:
            raise RuntimeError("Template compilation failed (Jinja2 required)")

        output_node = Buffer(name="buf_out", layout=layout)
        with patch.object(
            V.graph, "get_dtype", KernelTemplate._fake_get_dtype(output_node)
        ):
            kernel = self.kernel_type(
                kernel_name=benchmark_kernel_name,
                input_nodes=input_nodes,
                output_node=output_node,
            )
            code = kernel.render(self.template, **kwargs)

            input_call_args = tuple(kernel.args.input_buffers.keys())
            # KernelArgs deduplicates input buffers by string value, so mirror
            # that contract even when equal names are distinct Python objects.
            expected_input_args = tuple(
                OrderedSet(node.get_name() for node in input_nodes)
            )
            if input_call_args[: len(expected_input_args)] != expected_input_args:
                raise RuntimeError(
                    "FlyDSL template input registration order changed. "
                    f"Got {input_call_args}, expected prefix {expected_input_args}."
                )

            arg_defs, call_args, _, _ = kernel.args.python_argdefs()
            expected_args = list(input_call_args)
            expected_args.append(output_node.get_name())
            if list(call_args)[: len(expected_args)] != expected_args:
                raise RuntimeError(
                    "FlyDSL template benchmark argument order changed. "
                    f"Expected prefix {expected_args}, got {list(call_args)}."
                )
            extra_args = tuple(
                V.graph.sizevars.optimization_hints(call_args[len(expected_args) :])
            )

            bmreq = FlyDSLBenchmarkRequest(
                kernel_name=benchmark_kernel_name,
                input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
                output_tensor_meta=TensorMeta.from_irnodes(output_node),
                extra_args=extra_args,
                source_code=code,
            )
            output_code_log.debug("Generated FlyDSL Code:\n%s", bmreq.source_code)

            def make_kernel_render(out_node, hint_override: int | None = None):
                render_kernel = self.kernel_type(
                    kernel_name=str(Placeholder.KERNEL_NAME),
                    input_nodes=input_nodes,
                    output_node=out_node,
                )

                def render():
                    return render_kernel.render(self.template, **kwargs)

                return render_kernel, render

            caller_type = self.caller_type or FlyDSLTemplateCaller
            return caller_type(
                name=choice_name,
                input_nodes=input_nodes,
                layout=layout,
                make_kernel_render=make_kernel_render,
                bmreq=bmreq,
                template=self,
                mutated_inputs=mutated_inputs,
                template_kwargs=template_kwargs,
            )


class FlyDSLTemplateCaller(ChoiceCaller):
    """Caller for FlyDSL templates."""

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Any,
        bmreq: FlyDSLBenchmarkRequest,
        template: "FlyDSLTemplate",
        mutated_inputs: Iterable[IRNode] | None = None,
        template_kwargs: dict[str, Any] | None = None,
    ):
        description = self._build_description(name, template_kwargs)
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=description,
        )
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.mutated_inputs = mutated_inputs

    @staticmethod
    def _build_description(name: str, template_kwargs: dict[str, Any] | None) -> str:
        if not template_kwargs:
            return f"FlyDSL template {name}"
        kwargs_desc = ", ".join(f"{k}={v}" for k, v in template_kwargs.items())
        return f"FlyDSL template {name} ({kwargs_desc})"

    def __str__(self) -> str:
        return f"FlyDSLTemplateCaller({self.name})"

    def benchmark(self, *args, out) -> float:
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        buffer = FlyDSLTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            mutated_inputs=self.mutated_inputs,
        )
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return TensorBox.create(buffer)

    def call_name(self) -> str:
        return self.name

    def to_callable(self) -> Any:
        return self.make_kernel_render

    def hash_key(self) -> str:
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def info_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": "FlyDSL",
            "template": self.template.name,
        }
