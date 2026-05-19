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
        epilogue_arg_indices = kwargs.pop("epilogue_arg_indices", ())
        local_reduce_out_index = kwargs.pop("local_reduce_out_index", None)
        aux_out_index = kwargs.pop("aux_out_index", None)
        local_reduce_group = kwargs.pop("local_reduce_group", None)
        local_reduce_dim = kwargs.pop("local_reduce_dim", None)
        local_reduce_op = kwargs.pop("local_reduce_op", None)
        local_reduce_scale = kwargs.pop("local_reduce_scale", 1.0)
        local_reduce_max_power = kwargs.pop("local_reduce_max_power", 8)
        local_reduce_feeds_main = kwargs.pop("local_reduce_feeds_main", False)
        main_output_transform = kwargs.pop("main_output_transform", None)
        main_output_transform_group = kwargs.pop("main_output_transform_group", None)
        mutated_inputs = kwargs.pop("mutated_inputs", None)
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
            epilogue_arg_indices=epilogue_arg_indices,
            local_reduce_out_index=local_reduce_out_index,
            aux_out_index=aux_out_index,
            local_reduce_group=local_reduce_group,
            local_reduce_dim=local_reduce_dim,
            local_reduce_op=local_reduce_op,
            local_reduce_scale=local_reduce_scale,
            local_reduce_max_power=local_reduce_max_power,
            local_reduce_feeds_main=local_reduce_feeds_main,
            main_output_transform=main_output_transform,
            main_output_transform_group=main_output_transform_group,
            mutated_inputs=mutated_inputs,
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
        epilogue_arg_indices: tuple[int, ...] = (),
        local_reduce_out_index: int | None = None,
        aux_out_index: int | None = None,
        local_reduce_group: int | None = None,
        local_reduce_dim: int | None = None,
        local_reduce_op: str | None = None,
        local_reduce_scale: float = 1.0,
        local_reduce_max_power: int = 8,
        local_reduce_feeds_main: bool = False,
        main_output_transform: str | None = None,
        main_output_transform_group: int | None = None,
        mutated_inputs: list[Buffer] | None = None,
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
        self.epilogue_arg_indices = epilogue_arg_indices
        self.local_reduce_out_index = local_reduce_out_index
        self.aux_out_index = aux_out_index
        self.local_reduce_group = local_reduce_group
        self.local_reduce_dim = local_reduce_dim
        self.local_reduce_op = local_reduce_op
        self.local_reduce_scale = local_reduce_scale
        self.local_reduce_max_power = local_reduce_max_power
        self.local_reduce_feeds_main = local_reduce_feeds_main
        self.main_output_transform = main_output_transform
        self.main_output_transform_group = main_output_transform_group
        self.mutated_inputs = mutated_inputs

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
                epilogue_arg_indices=self.epilogue_arg_indices,
                local_reduce_out_index=self.local_reduce_out_index,
                aux_out_index=self.aux_out_index,
                local_reduce_group=self.local_reduce_group,
                local_reduce_dim=self.local_reduce_dim,
                local_reduce_op=self.local_reduce_op,
                local_reduce_scale=self.local_reduce_scale,
                local_reduce_max_power=self.local_reduce_max_power,
                local_reduce_feeds_main=self.local_reduce_feeds_main,
                main_output_transform=self.main_output_transform,
                main_output_transform_group=self.main_output_transform_group,
                mutated_inputs=self.mutated_inputs,
            )
        )

    def call_name(self) -> str:
        return self.name

    def to_callable(self) -> Any:
        raise NotImplementedError("QuACK GEMM epilogue templates are codegen-only")

    def hash_key(self) -> str:
        return code_hash(
            f"{self.gemm_op}\n{self.alpha}\n{self.beta}\n{self.out_dtype}\n"
            f"{self.epilogue_arg_indices}\n{self.local_reduce_out_index}\n{self.aux_out_index}\n"
            f"{self.local_reduce_group}\n{self.local_reduce_dim}\n{self.local_reduce_op}\n"
            f"{self.local_reduce_scale}\n{self.local_reduce_max_power}\n{self.local_reduce_feeds_main}\n"
            f"{self.main_output_transform}\n{self.main_output_transform_group}\n"
            f"{self.epilogue_name}\n{self.epilogue_source}"
        )

    def info_dict(self) -> dict[str, Any]:
        return {"backend": "QuACK", "template": "quack_gemm_epilogue"}


quack_gemm_epilogue_template = QuackGemmEpilogueTemplate()
