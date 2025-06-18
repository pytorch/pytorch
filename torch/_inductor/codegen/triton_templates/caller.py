import typing
from typing import Any, Optional, Union

from torch.utils._ordered_set import OrderedSet

from ... import config, ir
from ...codegen.common import WorkspaceArg
from ...ir import PrimitiveInfoType
from ...utils import do_bench_using_profiling


if typing.TYPE_CHECKING:
    from ...autotune_process import TritonBenchmarkRequest


class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def __init__(
        self,
        name,
        input_nodes,
        layout,
        make_kernel_render,
        description,
        bmreq,
        log_info: Optional[
            dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]
        ] = None,
        mutated_inputs=None,
        workspace_arg: Optional[WorkspaceArg] = None,
        allowed_prologue_inps: Optional[OrderedSet[str]] = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self.make_kernel_render = make_kernel_render
        self.bmreq: TritonBenchmarkRequest = bmreq
        if log_info is None:
            log_info = {}
        self.log_info: dict[str, Any] = log_info
        self.log_info.update(
            {
                "backend": "Triton",
                "num_stages": self.bmreq.num_stages,
                "num_warps": self.bmreq.num_warps,
            }
        )
        self.mutated_inputs = mutated_inputs
        self.workspace_arg = workspace_arg
        self.allowed_prologue_inps = (
            allowed_prologue_inps if allowed_prologue_inps is not None else OrderedSet()
        )

    def benchmark(self, *args, out):
        assert self.bmreq is not None
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def precompile(self):
        assert self.bmreq is not None
        self.bmreq.precompile()

    def __str__(self) -> str:
        return f"TritonTemplateCaller({self.bmreq.module_path}, {self.description})"

    def call_name(self):
        return f"template_kernels.{self.name}"

    def hash_key(self):
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                self.bmreq.module_cache_key,
            ]
        )

    def output_node(self):
        return ir.TensorBox.create(
            ir.TritonTemplateBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                make_kernel_render=self.make_kernel_render,
                mutated_inputs=self.mutated_inputs,
                allowed_prologue_inps=self.allowed_prologue_inps,
            )
        )

    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return self.log_info

    def get_make_kernel_render(self):
        return self.make_kernel_render

    def autoheuristic_id(self):
        type_name = "triton"
        info = self.info_dict()
        # TODO(AlnisM): Does tile_shape always exist?
        tile = info["tile_shape"]
        tile_vals = eval(tile)  # type: ignore[arg-type]
        BLOCK_M = tile_vals[0]
        BLOCK_K = tile_vals[1]
        BLOCK_N = tile_vals[2]
        num_stages = info["num_stages"]
        num_warps = info["num_warps"]
        return f"type={type_name}_BLOCK-M={BLOCK_M}_BLOCK-K={BLOCK_K}_BLOCK-N={BLOCK_N}_numstages={num_stages}_numwarps={num_warps}"
