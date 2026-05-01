# mypy: allow-untyped-defs
import itertools
from typing import Any

from torch._inductor import ir
from torch._inductor.codecache import code_hash
from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import Buffer, ChoiceCaller, Layout, TensorBox


class QuackSplitKTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self) -> None:
        super().__init__("quack_split_k")

    def maybe_append_choice(
        self, choices: list[Any], **kwargs: Any
    ) -> NotImplementedError | None:
        choices.append(self.generate(**kwargs))
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        return QuackSplitKTemplateCaller(
            name=f"quack_split_k_{next(self.index_counter)}",
            input_nodes=kwargs.pop("input_nodes"),
            layout=kwargs.pop("layout"),
            k_split=kwargs.pop("k_split"),
        )


class QuackSplitKTemplateCaller(ChoiceCaller):
    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        k_split: int,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"QuACK split-K partial GEMM {k_split=}",
        )
        self.k_split = k_split

    def benchmark(self, *args: Any, out: Any) -> float:
        return 0.0

    def output_node(self) -> TensorBox:
        return TensorBox.create(
            ir.QuackSplitKTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                k_split=self.k_split,
            )
        )

    def call_name(self) -> str:
        return self.name

    def to_callable(self) -> Any:
        raise NotImplementedError("QuACK split-K templates are codegen-only")

    def hash_key(self) -> str:
        return code_hash(f"{self.k_split}\n")

    def info_dict(self) -> dict[str, Any]:
        return {"backend": "QuACK", "template": "quack_split_k"}


quack_split_k_template = QuackSplitKTemplate()
