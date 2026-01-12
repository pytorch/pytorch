# mypy: allow-untyped-defs, disable-error-code="attr-defined, valid-type"
import functools
import logging
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.codegen.rocm.ck_tile_template import CKTileTemplate
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateKernel
from torch._inductor.codegen.rocm.rocm_template import ArgInfo
from torch._inductor.ir import Buffer, Layout
from torch.utils._ordered_set import OrderedSet
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
        return "ck_tile_gemm_universal_" + "_".join(
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


@functools.cache
def ops():
    """
    Generate the supported instance dataclasses
    """
    import itertools

    compute_v3_instances = [
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
            pipeline="CompV3",
            scheduler="Intrawave",
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
        for epilogue in ["Default", "CShuffle"]
    ]

    compute_v4_instances = [
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
            pipeline="CompV4",
            scheduler="Intrawave",
            epilogue=epilogue,
        )
        for (layout_a, layout_b, layout_c) in [
            ("Row", "Row", "Row"),
            ("Row", "Col", "Row"),
        ]
        for (datatype_a, datatype_b, datatype_c) in [("FP16",) * 3, ("BF16",) * 3]
        for (tile_m, tile_n, tile_k) in [
            (256, 256, 32)
        ]  # half the tile size since it has double buffering
        for (warp_m, warp_n, warp_k) in [(2, 2, 1)]
        for (warp_tile_m, warp_tile_n, warp_tile_k) in [(32, 32, 16)]
        for m_is_padded in ["true", "false"]
        for n_is_padded in ["true", "false"]
        for k_is_padded in ["true", "false"]
        for epilogue in ["Default", "CShuffle"]
    ]

    mem_instances = [
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
            pipeline="Mem",
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
        for scheduler in ["Intrawave", "Interwave"]
        for epilogue in ["Default", "CShuffle"]
    ]

    return list(
        itertools.chain(compute_v3_instances, compute_v4_instances, mem_instances)
    )


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

        using {{instance_namespace}}::BaseGemmPipeline;
        using {{instance_namespace}}::TilePartitioner;

        constexpr auto TileK = {{instance_namespace}}::TileK;
        constexpr auto kPrefetchStages = BaseGemmPipeline::PrefetchStages;

        const auto BiasTerms = std::array<const void*, 0> ();
        const auto BiasStrides = std::array<int32_t, 0> ();

        auto kargs = ck_tile::UniversalGemmKernelArgs<> {
           {X},
           {W},
           BiasTerms,
           Y,
           M,
           N,
           K,
           {LDA},
           {LDB},
           BiasStrides,
           LDC,
           kBatch
        };

        if (workspace_size) {
            *workspace_size = 0;
            return 0;
        }

        // run the kernel
        const auto dispatch = [&](const auto has_hot_loop_, const auto tail_number_) constexpr {
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

        const ck_tile::index_t k_grain     = kBatch * TileK;
        const ck_tile::index_t K_split     = (K + k_grain - 1) / k_grain * TileK;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

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

                template <ck_tile::index_t PrefetchStages, typename Dispatcher>
                void dispatch_memory_pipeline_hot_loop(const ck_tile::TailNumber tail_num, Dispatcher dispatch)
                {
                    if(tail_num == ck_tile::TailNumber::One)
                    {
                        dispatch(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
                    }
                    else if(tail_num == ck_tile::TailNumber::Full)
                    {
                        dispatch(ck_tile::bool_constant<true>{},
                            ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
                    }

                    if constexpr(PrefetchStages > 2)
                    {
                        if(tail_num == ck_tile::TailNumber::Two)
                        {
                            dispatch(ck_tile::bool_constant<true>{},
                                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
                        }
                    }
                    if constexpr(PrefetchStages > 3)
                    {
                        if(tail_num == ck_tile::TailNumber::Three)
                        {
                            dispatch(ck_tile::bool_constant<true>{},
                                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
                        }
                    }
                    if constexpr(PrefetchStages > 4)
                    {
                        if(tail_num == ck_tile::TailNumber::Four)
                        {
                            dispatch(ck_tile::bool_constant<true>{},
                                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
                        }
                    }
                    if constexpr(PrefetchStages > 5)
                    {
                        if(tail_num == ck_tile::TailNumber::Five)
                        {
                            dispatch(ck_tile::bool_constant<true>{},
                                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
                        }
                    }
                    if constexpr(PrefetchStages > 6)
                    {
                        if(tail_num == ck_tile::TailNumber::Six)
                        {
                            dispatch(ck_tile::bool_constant<true>{},
                                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
                        }
                    }
                    if constexpr(PrefetchStages > 7)
                    {
                        if(tail_num == ck_tile::TailNumber::Seven)
                        {
                            dispatch(ck_tile::bool_constant<true>{},
                                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
                        }
                    }
                }
            """
        )
        return res

    def check_dtypes(self, op: "CKTileGemmOperation"):
        X_dtype, W_dtype, out_dtype = [
            T.get_layout().dtype for T in [*self.input_nodes, self.output_node]
        ]
        if op.datatype_a != self._TORCH_DTYPE_TO_CK[X_dtype]:
            return False
        if op.datatype_b != self._TORCH_DTYPE_TO_CK[W_dtype]:
            return False
        if op.datatype_c != self._TORCH_DTYPE_TO_CK[out_dtype]:
            return False
        return True

    def check_layouts(self, op: "CKTileGemmOperation"):
        X_layout, W_layout, out_layout = [
            torch_layout_to_ck_layout(T.get_layout())
            for T in [*self.input_nodes, self.output_node]
        ]
        if op.layout_a != X_layout:
            return False
        if op.layout_b != W_layout:
            return False
        if op.layout_c != out_layout:
            return False
        return True

    def get_gemm_problem_size(self):
        X_size, W_size = [T.get_layout().size for T in [*self.input_nodes]]

        M, K = X_size
        _, N = W_size

        return M, N, K

    def check_block_tiles(self, op: "CKTileGemmOperation"):
        """
        The contiguous dimension of a tensor must be divisible by the block tile size
        This helper function enforces it for the inputs and the output.
        """
        M, N, K = self.get_gemm_problem_size()

        def check(dim_size, tile_size, is_padded):
            if (
                is_static_int(dim_size)
                and dim_size % tile_size != 0
                and is_padded == "false"
            ):
                return False
            return True

        if op.layout_a == "Row":
            # handle in kBatch check
            return True
        elif op.layout_a == "Col":
            if not check(M, op.tile_m, op.m_is_padded):
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_a=}")

        if op.layout_b == "Row":
            if not check(N, op.tile_n, op.n_is_padded):
                return False
        elif op.layout_b == "Col":
            # handle in kBatch check
            return True
        else:
            raise AssertionError(f"Invalid {op.layout_b=}")

        if op.layout_c == "Row":
            if not check(N, op.tile_n, op.n_is_padded):
                return False
        elif op.layout_c == "Col":
            if not check(M, op.tile_m, op.m_is_padded):
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_c=}")

        return True

    def check_alignments(self, op: "CKTileGemmOperation"):
        """
        The contiguous dimension of a tensor must be divisible by the vector load size.
        """
        M, N, K = self.get_gemm_problem_size()

        def max_alignment(contiguous_elements_per_tile, elements_per_thread, ck_dtype):
            for vector_load_bytes in (16, 8, 4, 2, 1):
                alignment = vector_load_bytes // self.ck_dtype_to_size[ck_dtype]
                if (
                    alignment > 0
                    and contiguous_elements_per_tile % alignment == 0
                    and elements_per_thread % alignment == 0
                ):
                    return alignment

        threads_per_block = (
            op.warp_m * op.warp_n * op.warp_k * self.gfx9_threads_per_warp
        )
        a_elements_per_thread = op.tile_m * op.tile_k / threads_per_block
        b_elements_per_thread = op.tile_n * op.tile_k / threads_per_block

        if op.layout_a == "Row":
            # K is contiguous tensor dimension
            a_max_vector_size = max_alignment(
                op.tile_k, a_elements_per_thread, op.datatype_a
            )
            if is_static_int(K) and K % a_max_vector_size != 0:
                return False
        elif op.layout_a == "Col":
            # M is contiguous tensor dimension
            a_max_vector_size = max_alignment(
                op.tile_m, a_elements_per_thread, op.datatype_a
            )
            if is_static_int(M) and M % a_max_vector_size != 0:
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_a=}")

        if op.layout_b == "Row":
            # N is contiguous tensor dimension
            b_max_vector_size = max_alignment(
                op.tile_n, b_elements_per_thread, op.datatype_b
            )
            if is_static_int(N) and N % b_max_vector_size != 0:
                return False
        elif op.layout_b == "Col":
            # K is contiguous tensor dimension
            b_max_vector_size = max_alignment(
                op.tile_k, b_elements_per_thread, op.datatype_b
            )
            if is_static_int(K) and K % b_max_vector_size != 0:
                return False
        else:
            raise AssertionError(f"Invalid layout {op.layout_b=}")

        # the `default` epilogue writes C to memory by 1 tensor element
        # (divisibility check not necessary)
        # the `cshuffle` epilogue writes C to memory by 16 bytes
        # (so the contiguous C dimension size must be divisible by the number of tensor elements in 16 bytes)
        if op.epilogue == "CShuffle":
            if (
                op.layout_c == "Row"
                and is_static_int(N)
                and N % (16 / self.ck_dtype_to_size[op.datatype_c]) != 0
            ):
                return False

        return True

    def check_warp_tiles(self, op: "CKTileGemmOperation"):
        if op.tile_m % (op.warp_m * op.warp_tile_m) != 0:
            return False
        if op.tile_n % (op.warp_n * op.warp_tile_n) != 0:
            return False
        if op.tile_k % (op.warp_k * op.warp_tile_k) != 0:
            return False
        return True

    def check_block_tile_size(self, op: "CKTileGemmOperation"):
        # assuming LDS size is 64KB
        if op.pipeline == "CompV4":
            max_block_tile_size = 2**15
        else:
            max_block_tile_size = 2**16

        block_tile_size = (
            self.ck_dtype_to_size[op.datatype_a] * op.tile_m * op.tile_k
            + self.ck_dtype_to_size[op.datatype_b] * op.tile_n * op.tile_k
        )
        if block_tile_size > max_block_tile_size:
            return False
        return True

    def filter_op(self, op: "CKTileGemmOperation"):
        """
        Determines whether a given op definition is suitable for the current
        input / output of the operation that this template implements.

        Filter is based on inputs' dtype, layout and statically inferred size.

        Returns None if the op is not suitable, otherwise returns the op to be used.
        """
        if not self.check_dtypes(op):
            return None
        if not self.check_layouts(op):
            return None
        if not self.check_block_tiles(op):
            return None
        if not self.check_alignments(op):
            return None

        return op

    def emit_ck_instance(self, op: "CKTileGemmOperation"):
        """
        This method is used to generate code which defines the type alias for the generated kernel class
        """
        template_definition = r"""
    // Gemm operator {{operation_name}}

    namespace {{operation_name}} {
         // block tile
        constexpr int32_t TileM = {{tile_m}};
        constexpr int32_t TileN = {{tile_n}};
        constexpr int32_t TileK = {{tile_k}};
        // warps per block
        constexpr int32_t WarpM = {{warp_m}};
        constexpr int32_t WarpN = {{warp_n}};
        constexpr int32_t WarpK = {{warp_k}};
        // xdl tile
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
        constexpr bool DoubleSmemBuffer = {{has_double_smem_buffer}};
        constexpr bool TransposeC = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
        constexpr ck_tile::index_t TilePartitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                   ck_tile::sequence<WarpM, WarpN, WarpK>,
                                   ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
                                   permuteA,
                                   permuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       TilePartitionerGroupNum,
                                                       TilePartitionerM01>;

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
                                                                          TransposeC>;
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
        """
            elif epilogue_type == "CShuffle":
                return r"""
            constexpr auto kMemoryOperation = ck_tile::memory_operation_enum::set;
            using DsDataType = ck_tile::tuple<>; // no bias terms for vanilla GEMM
            using DsLayout = ck_tile::tuple<>;
            constexpr auto ELayout = CLayout;
            using CDEElementWise = ck_tile::element_wise::PassThrough; // no-op
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                     BDataType,
                                                                     DsDataType,
                                                                     AccDataType,
                                                                     CDataType,
                                                                     DsLayout,
                                                                     ELayout,
                                                                     CDEElementWise,
                                                                     GemmPipelineProblem::kBlockSize,
                                                                     TileM,
                                                                     TileN,
                                                                     WarpM,
                                                                     WarpN,
                                                                     WarpTileM,
                                                                     WarpTileN,
                                                                     WarpTileK,
                                                                     TransposeC,
                                                                     kMemoryOperation>;

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
            has_double_smem_buffer=("true" if op.pipeline == "CompV4" else "false"),
        )
        return rendered_definition

    def render(  # type: ignore[override]
        self, kernel: ROCmTemplateKernel, op: "CKTileGemmOperation", **kwargs
    ) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        """
        epilogue_nodes = kwargs.get("epilogue_nodes")
        assert epilogue_nodes is None or 0 == len(epilogue_nodes)
        template_buffer_node = kwargs.get("template_buffer_node")
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
            switch_tailnum_template = r"""
            switch (tail_num) {
                {% for tail_num in valid_tailnums %}
                case ck_tile::TailNumber::{{tail_num}}:
                    dispatch({{has_hot_loop}},
                             ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::{{tail_num}}>{});
                    break;
                {% endfor %}
                default:
                    std::ostringstream err;
                    err << "Unsupported dispatch: "
                        << "Pipeline: " << "{{pipeline}}"
                        << "Prefetch stages: " << kPrefetchStages
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
            if pipeline_type == "CompV3":
                return self._template_from_string(dispatch_template).render(
                    rendered_with_hot_loop=self._template_from_string(
                        switch_tailnum_template
                    ).render(
                        has_hot_loop="ck_tile::integral_constant<bool, true>{}",
                        valid_tailnums=("Full", "Odd", "Even"),
                        pipeline=pipeline_type,
                    ),
                    rendered_without_hot_loop=self._template_from_string(
                        switch_tailnum_template
                    ).render(
                        has_hot_loop="ck_tile::integral_constant<bool, false>{}",
                        valid_tailnums=("Full", "Odd", "Even"),
                        pipeline=pipeline_type,
                    ),
                )
            elif pipeline_type == "Mem":
                return self._template_from_string(dispatch_template).render(
                    rendered_with_hot_loop="dispatch_memory_pipeline_hot_loop<kPrefetchStages>(tail_num, dispatch);",
                    rendered_without_hot_loop=self._template_from_string(
                        switch_tailnum_template
                    ).render(
                        has_hot_loop="ck_tile::integral_constant<bool, false>{}",
                        valid_tailnums=("Full", "Odd", "Even"),
                        pipeline=pipeline_type,
                    ),
                )
            elif pipeline_type == "CompV4":
                return self._template_from_string(dispatch_template).render(
                    rendered_with_hot_loop=self._template_from_string(
                        switch_tailnum_template
                    ).render(
                        has_hot_loop="ck_tile::integral_constant<bool, true>{}",
                        valid_tailnums=("Two", "Three"),
                        pipeline=pipeline_type,
                    ),
                    rendered_without_hot_loop=self._template_from_string(
                        switch_tailnum_template
                    ).render(
                        has_hot_loop="ck_tile::integral_constant<bool, false>{}",
                        valid_tailnums=("Full", "Odd", "Even"),
                        pipeline=pipeline_type,
                    ),
                )
            else:
                raise AssertionError(f"Pipeline {pipeline_type} is not supported")

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
        instances = ops()
        if not instances:
            raise AssertionError(
                "No Composable Kernel Universal GEMM instances found. "
                "Please check if the library is installed."
            )
        filtered_instances = list(filter(self.filter_op, instances))
        # NB: when using a fixed list order, most likely we will pick the subset of instances
        # which are very similar to each other. Randomizing the choice seems to solve this.
        random.seed(-11)
        chosen_instances = (
            random.sample(
                filtered_instances,
                min(len(filtered_instances), config.rocm.ck_tile_max_profiling_configs),
            )
            if config.rocm.ck_tile_max_profiling_configs
            else filtered_instances
        )
        log.debug(
            "generated %d ck instances after sample: %s",
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
            for k_batch in template.k_batch_choices(op):
                template.maybe_append_choice(
                    choices,
                    op=op,
                    kBatch=k_batch,
                )

    def k_batch_choices(self, op: "CKTileGemmOperation") -> tuple[int, ...]:
        """
        Returns a list of k_batch choices for the template.
        """
        default_choices = (1, 2, 4, 8, 16, 32)

        def check(dim_size, tile_size, is_padded):
            if (
                is_static_int(dim_size)
                and dim_size % tile_size != 0
                and is_padded == "false"
            ):
                return False
            return True

        _, _, K, _, _, _ = self.size_args()
        if op.layout_a == "Row" or op.layout_b == "Col":
            choices = tuple(
                filter(
                    lambda k_batch: check(K, op.tile_k * k_batch, op.k_is_padded),
                    default_choices,
                )
            )
        else:
            choices = default_choices

        if op.epilogue == "Default":
            choices = (1,)

        return choices

    def size_args(self):
        """
        Sizes and strides to be used for the kernel call
        """
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

    def get_runtime_arg_info(self) -> list[ArgInfo]:
        return [ArgInfo("kBatch", "int32_t")]

    def get_runtime_arg_values(self, **kwargs: Any) -> list[Any]:
        # maybe_append_choice kwarg for k_batch must match the name of the argument
        arg_names = OrderedSet([arg.name for arg in self.get_runtime_arg_info()])
        if not arg_names.issubset(kwargs):
            raise ValueError(
                "Missing runtime arguments: " + ", ".join(arg_names - kwargs.keys())
            )
        return [kwargs[k] for k in arg_names]
