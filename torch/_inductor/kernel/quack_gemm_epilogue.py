# mypy: allow-untyped-defs
import itertools
from typing import Any

from torch._inductor import ir
from torch._inductor.codecache import code_hash
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import Buffer, ChoiceCaller, Layout, TensorBox


class QuackGemmEpilogueTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self) -> None:
        super().__init__("quack_gemm_epilogue")

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        config = kwargs.pop("config")
        mutated_inputs = kwargs.pop("mutated_inputs", None)
        if kwargs:
            raise RuntimeError(f"unexpected QuACK GEMM epilogue options: {kwargs}")
        return QuackGemmEpilogueTemplateCaller(
            name=f"quack_gemm_epilogue_{next(self.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            config=config,
            mutated_inputs=mutated_inputs,
        )


class QuackGemmEpilogueTemplateCaller(ChoiceCaller):
    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        config: ir.QuackGemmEpilogueConfig,
        mutated_inputs: list[Buffer] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"QuACK GEMM epilogue template {config.epilogue_name}",
        )
        self.config = config
        self.mutated_inputs = mutated_inputs

    def benchmark(self, *args: Any, out: Any) -> float:
        return 0.0

    def output_node(self) -> TensorBox:
        return TensorBox.create(
            ir.QuackGemmEpilogueTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                config=self.config,
                mutated_inputs=self.mutated_inputs,
            )
        )

    def call_name(self) -> str:
        return self.name

    def to_callable(self) -> Any:
        raise NotImplementedError("QuACK GEMM epilogue templates are codegen-only")

    def hash_key(self) -> str:
        return code_hash(repr(self.config))

    def info_dict(self) -> dict[str, Any]:
        return {
            "backend": "QuACK",
            "template": "quack_gemm_epilogue",
            "tuned": self.config.tuned,
        }


quack_gemm_epilogue_template = QuackGemmEpilogueTemplate()
