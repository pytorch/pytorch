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

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> NotImplementedError | None:
        choices.append(self.generate(**kwargs))
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        epilogue_name = kwargs.pop("epilogue_name")
        epilogue_source = kwargs.pop("epilogue_source")
        gemm_op = kwargs.pop("gemm_op")
        alpha = kwargs.pop("alpha")
        beta = kwargs.pop("beta")
        out_dtype = kwargs.pop("out_dtype", None)
        return QuackGemmEpilogueTemplateCaller(
            name=f"quack_gemm_epilogue_{next(self.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            epilogue_name=epilogue_name,
            epilogue_source=epilogue_source,
            gemm_op=gemm_op,
            alpha=alpha,
            beta=beta,
            out_dtype=out_dtype,
        )


class QuackGemmEpilogueTemplateCaller(ChoiceCaller):
    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        epilogue_name: str,
        epilogue_source: str,
        gemm_op: str,
        alpha: float,
        beta: float,
        out_dtype: Any | None = None,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"QuACK GEMM epilogue template {epilogue_name}",
        )
        self.epilogue_name = epilogue_name
        self.epilogue_source = epilogue_source
        self.gemm_op = gemm_op
        self.alpha = alpha
        self.beta = beta
        self.out_dtype = out_dtype

    def benchmark(self, *args: Any, out: Any) -> float:
        return 0.0

    def output_node(self) -> TensorBox:
        return TensorBox.create(
            ir.QuackGemmEpilogueTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                epilogue_name=self.epilogue_name,
                epilogue_source=self.epilogue_source,
                gemm_op=self.gemm_op,
                alpha=self.alpha,
                beta=self.beta,
                out_dtype=self.out_dtype,
            )
        )

    def call_name(self) -> str:
        return self.name

    def to_callable(self) -> Any:
        raise NotImplementedError("QuACK GEMM epilogue templates are codegen-only")

    def hash_key(self) -> str:
        return code_hash(
            f"{self.gemm_op}\n{self.alpha}\n{self.beta}\n{self.out_dtype}\n"
            f"{self.epilogue_name}\n{self.epilogue_source}"
        )

    def info_dict(self) -> dict[str, Any]:
        return {"backend": "QuACK", "template": "quack_gemm_epilogue"}


quack_gemm_epilogue_template = QuackGemmEpilogueTemplate()
