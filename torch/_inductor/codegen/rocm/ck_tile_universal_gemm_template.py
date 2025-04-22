# mypy: allow-untyped-defs, disable-error-code="attr-defined, valid-type"
import functools
import logging
import random
from dataclasses import asdict, dataclass

import torch
from torch._inductor import config
from torch._inductor.codegen.rocm.ck_tile_template import CKTileTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.ir import Buffer, Layout

from ...utils import IndentedBuffer


log = logging.getLogger(__name__)


def is_static_int(number):
    import sympy

    return isinstance(number, (int, sympy.Integer))


def torch_layout_to_ck_layout(torch_layout):
    if torch_layout.stride[-1] == 1:
        return "Row"
    elif torch_layout.stride[-2] == 1:
        return "Col"
    else:
        return None


@dataclass
class CKTileGemmOperation:
    layout_a: str
    layout_b: str
    layout_c: str

    datatype_a: str
    datatype_b: str
    datatype_c: str

    tile_m: int
    tile_n: int
    tile_k: int

    warp_m: int
    warp_n: int
    warp_k: int

    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    m_is_padded: str
    n_is_padded: str
    k_is_padded: str

    pipeline: str
    scheduler: str
    epilogue: str

    def layout_repr(self):
        return f"{self.layout_a[0]}{self.layout_b[0]}{self.layout_c[0]}"

    def dtype_repr(self):
        return f"{self.datatype_a}{self.datatype_b}{self.datatype_c}"

    def tile_sizes(self):
        return "_".join(
            [
                f"{self.tile_m}{self.tile_n}{self.tile_k}",
                f"{self.warp_m}{self.warp_n}{self.warp_k}",
                f"{self.warp_tile_m}{self.warp_tile_n}{self.warp_tile_k}",
            ]
        )

    def name(self):
        return "cktile_gemm_universal" + "_".join(
            [
                f"{self.layout_repr()}",
                f"{self.dtype_repr()}",
                f"{self.tile_sizes()}",
                f"{self.pipeline}",
                f"{self.scheduler}",
                f"{self.epilogue}",
            ]
        )

    def dict_items(self):
        return asdict(self).items()

@functools.lru_cache(None)
def _default_ops_list():
    return [
        CKTileGemmOperation(
            layout_a=layout_a,
            layout_b=layout_b,
            layout_c=layout_c,
            datatype_a=datatype_a,
            datatype_b=datatype_b,
            datatype_c=datatype_c,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_tile_k=warp_tile_k,
            m_is_padded=m_is_padded,
            n_is_padded=n_is_padded,
            k_is_padded=k_is_padded,
            pipeline=pipeline,
            scheduler=scheduler,
            epilogue=epilogue,
        )
        for (layout_a, layout_b, layout_c) in [
            ("Row", "Row", "Row"),
            ("Row", "Col", "Row"),
        ]
        for (datatype_a, datatype_b, datatype_c) in [("FP16",) * 3, ("BF16",) * 3]
        for (tile_m, tile_n, tile_k) in [(256, 256, 32), (256, 256, 64)]
        for (warp_m, warp_n, warp_k) in [(2, 2, 1)]
        for (warp_tile_m, warp_tile_n, warp_tile_k) in [(32, 32, 16)]
        for m_is_padded in ["true", "false"]
        for n_is_padded in ["true", "false"]
        for k_is_padded in ["true", "false"]
        for pipeline in ["CompV3", "Mem"]
        for scheduler in ["Interwave", "Intrawave"]
        for epilogue in ["Default", "CShuffle"]
    ]


class CKTileGemmTemplate(CKTileTemplate):
    """
    This class is used for rendering CK-Tile Universal GEMM kernels
    """

    gemm_template = r"""{{version_comment}}
    {{headers}}
    {{globals}}
    {{instance_definition}}
    extern "C" {
    PT_EXPORT {{kernel_definition}} {

        constexpr int32_t kBatch = 1;

        auto kargs = ck_tile::GemmKernelArgs {
           X,
           W,
           Y,
           M,
           N,
           K,
           LDA,
           LDB,
           LDC,
           kBatch
        };

        if (workspace_size) {
            *workspace_size = 0;
            return 0;
        }

        // run the kernel
        const auto Dispatch = [&](const auto has_hot_loop_, const auto tail_number_) constexpr {
            using Kernel = {{instance_namespace}}::Kernel<has_hot_loop_.value, tail_number_.value>;
            if (!Kernel::IsSupportedArgument(kargs)) {
                // we do our best to statically avoid this case in `filter_op`
                throw std::runtime_error("invalid argument");
            }
            auto stream_config = ck_tile::stream_config{stream};
            auto grid_size = Kernel::GridSize(M, N, kBatch);
            constexpr auto block_size = Kernel::BlockSize();
            constexpr auto lds_bytes = 0;
            constexpr auto kBlockPerCU = 1;
            auto gemm = ck_tile::make_kernel<block_size.x, kBlockPerCU>(Kernel{}, grid_size, block_size, lds_bytes, kargs);
            float elapsed_time = ck_tile::launch_kernel(stream_config, gemm);
        };

        const ck_tile::index_t k_grain     = kBatch * {{instance_namespace}}::TileK;
        const ck_tile::index_t K_split     = (K + k_grain - 1) / k_grain * {{instance_namespace}}::TileK;
        const ck_tile::index_t num_loop    = {{instance_namespace}}::TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = {{instance_namespace}}::BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = {{instance_namespace}}::BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        {{rendered_dispatch}}

        return 0;
    } // kernel definition
    } // extern C
    """

    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
    ) -> None:
        super().__init__(
            "ck_tile_gemm_template",
            input_nodes=input_nodes,
            layout=layout,
        )

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK GEMM header(s)

                #include "ck_tile/ops/gemm.hpp"
                #include "ck_tile/ops/epilogue.hpp"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                // CK GEMM globals

                using Row = ck_tile::tensor_layout::gemm::RowMajor;
                using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

            """
        )
        return res

    def filter_op(self, op: "CKTileGemmOperation"):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
        metas = [T.get_layout() for T in [*self.input_nodes, self.output_node]]
        X_meta = metas[0]
        W_meta = metas[1]
        Y_meta = metas[-1]
        # disable the instance if dtypes don't match
        if op.datatype_a != self._TORCH_DTYPE_TO_CK[X_meta.dtype]:
            return None
        if op.datatype_b != self._TORCH_DTYPE_TO_CK[W_meta.dtype]:
            return None
        if op.datatype_c != self._TORCH_DTYPE_TO_CK[Y_meta.dtype]:
            return None
        # disable the instance if layouts don't match
        if op.layout_a != torch_layout_to_ck_layout(X_meta):
            return None
        if op.layout_b != torch_layout_to_ck_layout(W_meta):
            return None
        if op.layout_c != torch_layout_to_ck_layout(Y_meta):
            return None
        return op

    def emit_ck_instance(self, op: "CKTileGemmOperation"):
        """
        This method is used to generate code which defines the type alias for the generated kernel class
        """
        template_definition = r"""
    // Gemm operator {{operation_name}}

    namespace {{operation_name}} {

        constexpr int32_t TileM = {{tile_m}};
        constexpr int32_t TileN = {{tile_n}};
        constexpr int32_t TileK = {{tile_k}};

        constexpr int32_t WarpM = {{warp_m}};
        constexpr int32_t WarpN = {{warp_n}};
        constexpr int32_t WarpK = {{warp_k}};

        constexpr int32_t WarpTileM = {{warp_tile_m}};
        constexpr int32_t WarpTileN = {{warp_tile_n}};
        constexpr int32_t WarpTileK = {{warp_tile_k}};

        constexpr bool kPadM = {{m_is_padded}};
        constexpr bool kPadN = {{n_is_padded}};
        constexpr bool kPadK = {{k_is_padded}};

        using ALayout = {{layout_a}};
        using BLayout = {{layout_b}};
        using CLayout = {{layout_c}};

        using ADataType = {{datatype_a}};
        using BDataType = {{datatype_b}};
        using CDataType = {{datatype_c}};
        using AccDataType = F32;

        constexpr bool permuteA = false;
        constexpr bool permuteB = false;
        constexpr bool DoubleSmemBuffer = false;
        constexpr bool TransposeC = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                   ck_tile::sequence<WarpM, WarpN, WarpK>,
                                   ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
                                   permuteA,
                                   permuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       TileParitionerGroupNum,
                                                       TileParitionerM01>;

        using Traits  =
            ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                             ALayout, BLayout, CLayout, TransposeC>;

        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        {{rendered_scheduler}}

        template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  scheduler,
                                                  has_hot_loop_v,
                                                  tail_number_v>;

        {{rendered_pipeline}}

        {{rendered_epilogue}}

        template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline<has_hot_loop_v, tail_number_v>, GemmEpilogue>;
    }

"""

        def render_epilogue(epilogue_type):
            if epilogue_type == "Default":
                return r"""
            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                                                          BDataType,
                                                                          AccDataType,
                                                                          CDataType,
                                                                          CLayout,
                                                                          kPadM,
                                                                          kPadN,
                                                                          WarpTileM,
                                                                          WarpTileN,
                                                                          WarpTileK,
                                                                          UniversalGemmProblem::TransposeC>;
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
        """
            elif epilogue_type == "CShuffle":
                return r"""
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                    BDataType,
                                                    AccDataType,
                                                    CDataType,
                                                    CLayout,
                                                    GemmPipelineProblem::kBlockSize,
                                                    TilePartitioner::MPerBlock,
                                                    TilePartitioner::NPerBlock,
                                                    WarpM,
                                                    WarpN,
                                                    WarpTileM,
                                                    WarpTileN,
                                                    WarpTileK,
                                                    TransposeC>;

            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
        """
            else:
                raise AssertionError("Epilogue must be set")

        def render_pipeline(pipeline_type):
            return rf"""
            using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCr{pipeline_type}<GemmPipelineProblem>;

            template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
            using GemmPipeline = ck_tile::GemmPipelineAgBgCr{pipeline_type}<UniversalGemmProblem<has_hot_loop_v, tail_number_v>>;
        """

        def render_scheduler(scheduler_type):
            return rf"""
            constexpr auto scheduler = ck_tile::GemmPipelineScheduler::{scheduler_type};
        """

        rendered_definition = self._template_from_string(template_definition).render(
            operation_name=op.name(),
            **asdict(op),
            rendered_scheduler=render_scheduler(op.scheduler),
            rendered_pipeline=render_pipeline(op.pipeline),
            rendered_epilogue=render_epilogue(op.epilogue),
        )
        return rendered_definition

    def render(  # type: ignore[override]
        self, kernel: ROCmTemplateKernel, op: "CKTileGemmOperation", **kwargs
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        """
        epilogue_nodes = kwargs.get("epilogue_nodes", None)
        assert epilogue_nodes is None or 0 == len(epilogue_nodes)
        template_buffer_node = kwargs.get("template_buffer_node", None)
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        assert 2 == len(self.input_nodes)
        X, W = self.input_nodes
        Y = self.output_node

        instance_definition = self.emit_ck_instance(op)

        version_comment = rf"""/**
* Generated code for CK inductor backend
* See {type(self).__module__}.{type(self).__qualname__}
*
* Template instance {op}
*
* {torch.__version__=}
* torch.version.git_version={getattr(torch.version, "git_version", "None")}
*/
"""

        def render_dispatch(pipeline_type, op_name):
            # TBD unhardcode dispatch based on pipeline type
            switch_tailnum_template = r"""
            switch (tail_num) {
                {% for tail_num in valid_tailnums %}
                case ck_tile::TailNumber::{{tail_num}}:
                    Dispatch({{has_hot_loop}},
                             ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::{{tail_num}}>{});
                    break;
                {% endfor %}
                default:
                    std::ostringstream err;
                    err << "Unsupported dispatch: "
                        << "Pipeline: " << {{pipeline}}
                        << "Prefetch stages: " << {{instance_namespace}}::BaseGemmPipeline::PrefetchStages
                        << "Tail num: " << tail_num;
                    throw std::runtime_error(err.str());
            } // switch tail_num
            """
            dispatch_template = r"""
        if (has_hot_loop) {
            {{rendered_with_hot_loop}}
        }
        else { // has_hot_loop == false
            {{rendered_without_hot_loop}}
        } // if has_hot_loop
        """
            pipeline_to_valid_tailnums = {
                "CompV3": ("Full", "Odd", "Even"),
                "Mem": ("Full", "One", "Two", "Three", "Four", "Five", "Six", "Seven"),
            }
            return self._template_from_string(dispatch_template).render(
                rendered_with_hot_loop=self._template_from_string(
                    switch_tailnum_template
                ).render(
                    has_hot_loop="ck_tile::integral_constant<bool, true>",
                    valid_tailnums=pipeline_to_valid_tailnums.get(
                        pipeline_type, tuple()
                    ),
                    pipeline=pipeline_type,
                    instance_namespace=op_name,
                ),
                rendered_without_hot_loop=self._template_from_string(
                    switch_tailnum_template
                ).render(
                    has_hot_loop="ck_tile::integral_constant<bool, false>",
                    valid_tailnums=("Full", "Odd", "Even"),
                    pipeline=pipeline_type,
                    instance_namespace=op_name,
                ),
            )

        return self._template_from_string(self.gemm_template).render(
            headers=self.header().getvalue(),
            globals=self.globals().getvalue(),
            instance_definition=instance_definition,
            kernel_definition=kernel.def_kernel(
                inputs=[X, W],  # type: ignore[list-item]
                outputs=[Y],
                names_str="X, W, Y",
                size_args=[
                    f"int32_t {arg}" for arg in ["M", "N", "K", "LDA", "LDB", "LDC"]
                ],
            ),
            instance_namespace=op.name(),
            version_comment=version_comment,
            rendered_dispatch=render_dispatch(op.pipeline, op.name()),
        )

    def gen_ops(self):
        """
        Creates a list of `CKTileGemmOperation` instances that match the GEMM operation this template represents.
        The instances are guaranteed to have the correct layout, dtype and dimension padding for the GEMM input arguments.

        An instance may invalidate the GEMM configuration at runtime.
        Such instances will be assigned +inf runtime by the autotune process.
        """
        unfiltered_instances = _default_ops_list()
        filtered_instances = list(
            filter(lambda op: self.filter_op(op), unfiltered_instances)
        )
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = (
            random.sample(
                filtered_instances,
                min(len(filtered_instances), config.rocm.n_max_profiling_configs),
            )
            if config.rocm.n_max_profiling_configs
            else filtered_instances
        )
        log.debug(
            "generated %d ck instances after filter: %s",
            len(chosen_instances),
            chosen_instances,
        )
        return chosen_instances

    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
    ):
        """
        Add Composable Kernel Universal GEMM instance choices to the auto-tuning list.
        """
        template = CKTileGemmTemplate(
            input_nodes,
            layout,
        )
        ops = template.gen_ops()
        for op in ops:
            template.maybe_append_choice(
                choices,
                op=op,
            )

    def size_args(self):
        X = self.input_nodes[0]
        W = self.input_nodes[1]
        Y = self.output_node

        M = X.get_size()[0]
        K = X.get_size()[1]
        N = W.get_size()[1]
        LDA = X.get_stride()[0 if X.get_stride()[1] == 1 else 1]
        LDB = W.get_stride()[0 if W.get_stride()[1] == 1 else 1]
        LDC = Y.get_stride()[0 if Y.get_stride()[1] == 1 else 1]

        return M, N, K, LDA, LDB, LDC
