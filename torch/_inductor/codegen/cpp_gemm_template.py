# mypy: allow-untyped-defs
import contextlib
import logging
import math
from functools import lru_cache
from typing import Any, Callable, cast, Dict, List, Optional, Union
from unittest.mock import patch

import torch
import torch.utils
from torch.utils._ordered_set import OrderedSet

from ..._dynamo.utils import counters
from .. import config, ir, lowering as L
from ..kernel.mm_common import mm_args
from ..select_algorithm import DataProcessorTemplateWrapper
from ..utils import (
    has_free_symbols,
    is_same_mkldnn_tensor,
    is_same_tensor,
    parallel_num_threads,
)
from ..virtualized import ops, V
from .cpp import get_export_declaration
from .cpp_micro_gemm import (
    CppMicroBrgemm,
    CppMicroGemm,
    CppMicroGemmAMX,
    create_micro_gemm,
    LayoutType,
)
from .cpp_template import CppTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import (
    create_epilogue_with_attr,
    DTYPE_TO_CPP,
    GemmBlocking,
    get_gemm_template_output_and_compute_dtype,
)


log = logging.getLogger(__name__)

GEMM_TEMPLATE_INIT_BLOCKING = r"""
    constexpr int64_t num_threads = {{num_threads}};
    constexpr int64_t N = {{N}};
    constexpr int64_t K = {{K}};
    constexpr int64_t Mr = {{micro_gemm.register_blocking.block_m}};
    constexpr int64_t Nr = {{micro_gemm.register_blocking.block_n}};
    constexpr int64_t Kr = {{micro_gemm.register_blocking.block_k}};
    constexpr int64_t Nr_blocks = (N + Nr - 1) / Nr;
    constexpr int64_t Kr_blocks = (K + Kr - 1) / Kr;

{%- if is_dynamic_M %}
    const int64_t M = {{kernel.size(GemmOut, 0)}};
    const int64_t Mr_blocks = (M + Mr - 1) / Mr;
    {%- if num_threads > 1 %}
    int64_t Mt_blocks, Nt_blocks, Kt_blocks;
    mm_get_thread_blocking(num_threads, {{config.cpp.gemm_max_k_slices}}, M, N, K, Mr, Nr, Kr, Mt_blocks, Nt_blocks, Kt_blocks);
    {%- else %}
    const auto Mt_blocks = Mr_blocks;
    const auto Nt_blocks = Nr_blocks;
    const auto Kt_blocks = Kr_blocks;
    {%- endif %}
    int64_t Mc_blocks, Nc_blocks, Kc_blocks;
    uint32_t L1_cache_size = {{L1_cache_size}};
    uint32_t L2_cache_size = {{L2_cache_size}};
    mm_get_cache_blocking<{{kernel.dtype(X)}}, {{kernel.dtype(W)}}>(
        num_threads,
        M,
        N,
        K,
        Mr,
        Nr,
        Kr,
        Mt_blocks,
        Nt_blocks,
        Kt_blocks,
        Mc_blocks,
        Nc_blocks,
        Kc_blocks,
        L1_cache_size,
        L2_cache_size
    );
    const int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    const int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    const int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    const int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    const int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;
{%- else %}
    constexpr int64_t M = {{kernel.size(GemmOut, 0)}};
    constexpr int64_t Mr_blocks = (M + Mr - 1) / Mr;
    constexpr int64_t Mt_blocks = {{template.thread_blocking(num_threads).block_m}};
    constexpr int64_t Nt_blocks = {{template.thread_blocking(num_threads).block_n}};
    constexpr int64_t Kt_blocks = {{template.thread_blocking(num_threads).block_k}};
    constexpr int64_t Mc_blocks = {{template.cache_blocking(num_threads).block_m}};
    constexpr int64_t Nc_blocks = {{template.cache_blocking(num_threads).block_n}};
    constexpr int64_t Kc_blocks = {{template.cache_blocking(num_threads).block_k}};
    constexpr int64_t num_Mc_blocks = (Mr_blocks + Mc_blocks - 1) / Mc_blocks;
    constexpr int64_t num_Nc_blocks = (Nr_blocks + Nc_blocks - 1) / Nc_blocks;
    constexpr int64_t num_Mt_blocks = (Mr_blocks + Mt_blocks - 1) / Mt_blocks;
    constexpr int64_t num_Nt_blocks = (Nr_blocks + Nt_blocks - 1) / Nt_blocks;
    constexpr int64_t num_Kt_blocks = (Kr_blocks + Kt_blocks - 1) / Kt_blocks;
{%- endif %}

    // make sure all partitions are assigned
    {{kernel.assert_function}}(
        Mt_blocks * Nt_blocks * Kt_blocks * {{num_threads}} >= Mr_blocks * Nr_blocks * Kr_blocks,
        "Not all partitions are assigned."
    );
"""

GEMM_TEMPLATE_MULTI_THREADS_PARAMS = r"""
const int tid = omp_get_thread_num();
const int64_t k_group_id = tid / num_Kt_blocks;
const int64_t k_slice_id = tid % num_Kt_blocks;
const int64_t n_group_id = k_group_id / num_Nt_blocks;
const int64_t n_slice_id = k_group_id % num_Nt_blocks;
const int64_t k_block_start = k_slice_id * Kt_blocks;
const int64_t k_block_end = std::min(k_block_start + Kt_blocks, Kr_blocks);
const int64_t n_block_start = n_slice_id * Nt_blocks;
const int64_t n_block_end = std::min(n_block_start + Nt_blocks, Nr_blocks);
const int64_t m_block_start = std::min(n_group_id * Mt_blocks, Mr_blocks);
const int64_t m_block_end = std::min(m_block_start + Mt_blocks, Mr_blocks);
const int64_t num_Mc_blocks_per_thread = (m_block_end - m_block_start + Mc_blocks - 1) / Mc_blocks;
"""

GEMM_TEMPLATE_SINGLE_THREAD_PARAMS = r"""
constexpr int tid = 0;
constexpr int64_t k_group_id = 0;
constexpr int64_t k_slice_id = 0;
constexpr int64_t n_group_id = 0;
constexpr int64_t n_slice_id = 0;
constexpr int64_t m_block_start = 0;
constexpr int64_t n_block_start = 0;
constexpr int64_t n_block_end = Nr_blocks;
constexpr int64_t k_block_start = 0;
constexpr int64_t k_block_end = Kr_blocks;
{%- if is_dynamic_M %}
const int64_t num_Mc_blocks_per_thread = num_Mc_blocks;
const int64_t m_block_end = Mr_blocks;
{%- else %}
constexpr int64_t num_Mc_blocks_per_thread = num_Mc_blocks;
constexpr int64_t m_block_end = Mr_blocks;
{%- endif %}
"""

GEMM_TEMPLATE_M_LOOP_PARAMS = r"""
const int64_t my_mc_block_id = (mc_block_id + n_slice_id) % num_Mc_blocks_per_thread;
const int64_t mc = m_block_start + my_mc_block_id * Mc_blocks;
const int64_t m_start = mc * Mr;
const int64_t m_end = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
const int64_t m_size = m_end - m_start;
"""

GEMM_TEMPLATE_N_LOOP_PARAMS = r"""
const int64_t n_start = nc * Nr;
const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
const int64_t n_size = n_end - n_start;
// NB: assume we pad N, nc_block_end won't exceed padded N here.
const int64_t nc_block_end = std::min(nc + Nc_blocks, n_block_end);
"""

GEMM_TEMPLATE_MICROKERNEL_DEF = r"""
{{template.header().getvalue()}}

{{micro_gemm.codegen_define(kernel)}}
"""

GEMM_TEMPLATE_STUB_DEF = r"""
{%- if x_scale is not none %}
    {%- set kernel_args = {"X": X, "W": W, "inp": inp, "x_scale": x_scale, "x_zp": x_zp, "w_scale": w_scale, "w_zp": w_zp,} %}
{%- else %}
    {%- set kernel_args = {"X": X, "W": W, "inp": inp} %}
{%- endif %}

extern "C" {{export_declaration}}
{{kernel.def_kernel(inputs=kernel_args, outputs={"Y": Y}, aliases=aliases)}}
"""

GEMM_TEMPLATE = r"""
{{ template.codegen_gemm_stub_def() }}
{
    {{ kernel.maybe_codegen_profile() }}
    {{ template.codegen_blocks(
        num_threads, N, K, micro_gemm, is_dynamic_M, kernel, GemmOut, config, L1_cache_size, L2_cache_size, X, W
    ) }}

{%- if maybe_k_slicing %}
    std::unique_ptr<std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[]> local_buf_ptrs;
    if (num_Kt_blocks > 1) {
        local_buf_ptrs.reset(new std::unique_ptr<{{DTYPE_TO_CPP[acc_buf_dtype]}}[]>[num_Mc_blocks * num_Nc_blocks * num_Kt_blocks]);
    }
{%- endif %}

{%- if num_threads > 1 %}
    #pragma omp parallel num_threads({{num_threads}})
    {
        {{ template.codegen_multi_threads_params()|indent(8, false) }}
{%- else %}
    {
        {{ template.codegen_single_thread_params(is_dynamic_M)|indent(8, false) }}
{%- endif %}
        {{ micro_gemm.codegen_init(kernel) }}
{%- if use_local_acc %}
    {%- set acc_buf_name = "local_acc_buf" %}
        {{ kernel.define_buffer(acc_buf_name, ["Mc_blocks*Mr", "Nc_blocks*Nr"], acc_buf_dtype) }}
{%- endif %}
        for (int64_t mc_block_id = 0; mc_block_id < num_Mc_blocks_per_thread; mc_block_id++) {
            {{ template.codegen_m_loop_params()|indent(12, false) }}
            for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                {{ template.codegen_n_loop_params()|indent(16, false) }}
{%- if use_local_acc %}
    {%- set acc = kernel.local_buffers[acc_buf_name] %}
                {{ kernel.reinit_buffer_if_null(acc_buf_name) }}
{%- else %}
    {%- set acc = kernel.slice_nd(GemmOut, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- endif %}
                for (int64_t kc = k_block_start; kc < k_block_end; kc += Kc_blocks) {
                    int64_t k_start = kc * Kr;
                    int64_t k_end = std::min(std::min(kc + Kc_blocks, k_block_end) * Kr, K);
{%- set tile_X = kernel.slice_nd(X, [("m_start", "m_end"), ("k_start", "k_end")]) %}
                    for (int64_t nci = nc; nci < nc_block_end; nci++) {
{%- set acc_slice = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("(nci - nc)*Nr", "(nci - nc + 1)*Nr")]) %}
{%- if template.should_block_weights %}
{%- set tile_W_3d = kernel.slice_nd(W, [("nci", "nci + 1"), ("k_start", "k_end"), ()]) %}
{%- set tile_W = kernel.view(tile_W_3d, ["k_end - k_start", micro_gemm.register_blocking.block_n]) %}
{%- else %}
{%- set tile_W = kernel.slice_nd(W, [("k_start", "k_end"), ("n_start", "n_start + n_size")]) %}
{%- endif %}
                        if (kc == k_block_start) {
                            {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc_slice, accum=False)|indent(28, false) }}
                        } else {
                            {{ micro_gemm.codegen_call(kernel, tile_X, tile_W, acc_slice, accum=True)|indent(28, false) }}
                        }
                    }
                }
{%- if maybe_k_slicing %}
                if (num_Kt_blocks > 1) {
                    const int64_t mxn_cache_block_id = (mc / Mc_blocks) * num_Nc_blocks + nc;
                    local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks + k_slice_id].reset(
                        {{ kernel.release_buffer(acc_buf_name) }});
                } else
{%- endif %}
                {
{%- set tile_Y = kernel.slice_nd(Y_2d, [("m_start", "m_end"), ("n_start", "n_end")]) %}
{%- set tile_acc = kernel.slice_nd(acc, [("0", "m_end - m_start"), ("0", "n_end - n_start")]) %}
                    {{ kernel.store_output(
                        tile_Y, tile_acc, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
{%- if maybe_k_slicing %}
        if (num_Kt_blocks > 1) {
            #pragma omp barrier
            for (int64_t mc = m_block_start; mc < m_block_end; mc += Mc_blocks) {
                // We slice M-dim and each thread in the k-slicing group works on a slice
                const int64_t m_start_unsliced = mc * Mr;
                const int64_t m_end_unsliced = std::min(std::min(mc + Mc_blocks, m_block_end) * Mr, M);
                const int64_t m_size_unsliced = m_end_unsliced - m_start_unsliced;
                const int64_t m_slice_size = (m_size_unsliced + num_Kt_blocks - 1) / num_Kt_blocks;
                const int64_t m_start = std::min(m_start_unsliced + m_slice_size * k_slice_id, m_end_unsliced);
                const int64_t m_end = std::min(m_start_unsliced + m_slice_size * (k_slice_id + 1), m_end_unsliced);
                const int64_t m_size = m_end - m_start;
                const int64_t m_offset = m_start - m_start_unsliced;
                for (int64_t nc = n_block_start; nc < n_block_end; nc += Nc_blocks) {
                    const int64_t n_start = nc * Nr;
                    const int64_t n_end = std::min(std::min(nc + Nc_blocks, n_block_end) * Nr, N);
                    const int64_t n_size = n_end - n_start;
                    const int64_t mxn_cache_block_id = (mc / Mc_blocks) * num_Nc_blocks + nc;
                    auto {{acc_buf_name}} = local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks].get();
                    for (int64_t other_slice = 1; other_slice < num_Kt_blocks; other_slice++) {
                        auto other_acc = local_buf_ptrs[mxn_cache_block_id * num_Kt_blocks + other_slice].get();
                        for (int64_t m = m_offset; m < m_offset + m_size; m++) {
                            #pragma omp simd
                            for (int64_t n = 0; n < n_size; n++) {
                                {{acc_buf_name}}[m*Nr + n] += other_acc[m*Nr + n];
                            }
                        }
                    }
    {%- set tile_acc_m_slice = kernel.slice_nd(tile_acc, [("m_offset", "m_offset + m_end - m_start"), ()]) %}
                    {{ kernel.store_output(
                        tile_Y, tile_acc_m_slice, GemmOut, epilogue_nodes, offsets=("m_start", "n_start"), reindexers=reindexers
                    )|indent(20, false)
                    }}
                }
            }
        }
{%- endif %}
        {{ micro_gemm.codegen_finalize(kernel) }}
    }
}
"""


def get_padded_n(n, block_n):
    return (n + block_n - 1) // block_n * block_n


def transpose_w(
    W: Union[ir.IRNode, torch.Tensor], trans_w: bool
) -> Union[ir.IRNode, torch.Tensor]:
    """
    Transpose W based on the trans_w flag.
    """
    if isinstance(W, ir.IRNode):
        if trans_w:
            if not isinstance(W, ir.TensorBox):
                W = ir.TensorBox(W)
            W = L.permute(W, [1, 0])
    else:
        if trans_w:
            assert isinstance(W, torch.Tensor)
            W = W.transpose(0, 1)
    return W


def expand_bias(
    B: Union[ir.IRNode, torch.Tensor, None], X: Union[ir.IRNode, torch.Tensor]
) -> Optional[Union[ir.IRNode, torch.Tensor]]:
    """
    Expand Bias to the same size of X.
    """
    if B is not None:
        if isinstance(B, ir.IRNode):
            if not isinstance(B, ir.TensorBox):
                B = ir.TensorBox(B)
            assert hasattr(X, "get_size")
            B = L.expand(B, (X.get_size()[0], B.get_size()[-1]))
        else:
            assert isinstance(B, torch.Tensor)
            assert isinstance(X, torch.Tensor)
            B = B.expand(X.shape[0], B.shape[-1])
    return B


def prune_tensors(input_nodes: List[ir.TensorBox], new_input_nodes: List[ir.TensorBox]):
    """
    Prune unused tensors from `V.graph` since the GEMM Template use new packed weight.
    """

    def share_storage(base_tensor: torch.Tensor, comp_tensor: torch.Tensor):
        return base_tensor.is_mkldnn == comp_tensor.is_mkldnn and (
            is_same_tensor(base_tensor, comp_tensor)
            or is_same_mkldnn_tensor(base_tensor, comp_tensor)
        )

    def get_candidates(input_nodes, new_input_nodes):
        # Only Constant Buffer like weight and bias might be changed in GEMM Template.
        # The Inductor IR Node may changed, but still share the storage. For example:
        # bias in bfloat16 case which only do the expand
        return [
            node
            for node in input_nodes
            if (
                node not in new_input_nodes
                and isinstance(node, (ir.TensorBox, ir.StorageBox))
                and node.get_name() in V.graph.constants
                and not any(
                    (
                        isinstance(new_node, (ir.TensorBox, ir.StorageBox))
                        and new_node.get_name() in V.graph.constants
                        and share_storage(
                            V.graph.constants[node.get_name()],
                            V.graph.constants[new_node.get_name()],
                        )
                    )
                    for new_node in new_input_nodes
                )
            )
        ]

    for candidate_node in get_candidates(input_nodes, new_input_nodes):
        # By using the new packed weight for the GEMM template, we can prune the
        # old weight if it has no other users. This saves memory but makes the FX graph
        # non-retraceable. To support retracing, we can add a repack node to the
        # FX graph. For example:
        # mkldnn._linear_pointwise <- repack_linear_wgt <- packed_wgt_for_template
        candidate_tensor_users = 0
        candidate_tensor = V.graph.constants[candidate_node.get_name()]
        for node in reversed(V.graph.graph.nodes):
            # Case may happen when the candidate tensor is used by more than 1 get_attr node
            # https://github.com/pytorch/pytorch/issues/134998
            if node.op == "get_attr" and hasattr(
                V.graph.module, node.name
            ):  # candidate tensor might already be deleted
                comp_tensor = getattr(V.graph.module, node.name)
                if isinstance(comp_tensor, torch.Tensor) and share_storage(
                    candidate_tensor, comp_tensor
                ):
                    candidate_tensor_users += 1

        for node in reversed(V.graph.graph.nodes):
            # The get_attr node has only 1 user fx node
            # The candidate tensor has been used by only 1 get_attr node
            if (
                node.name == candidate_node.get_name()
                and len(node.users) == 1
                and candidate_tensor_users == 1
            ):
                del V.graph.constants[node.name]
                delattr(V.graph.module, node.name)
                delattr(V.graph.graph.owning_module, node.name)


def gen_2d_view_of_epilogue_buf(
    Y: ir.Buffer,
    template_buffer: ir.Buffer,
    epilogue_nodes: List[ir.IRNode],
    reindexers: List[Optional[Callable[[List[Any]], List[Any]]]],
    default_reindexers: List[Optional[Callable[[List[Any]], List[Any]]]],
) -> tuple[
    Union[ir.Buffer, ir.ReinterpretView],
    List[Optional[Callable[[List[Any]], List[Any]]]],
]:
    """
    The dimension and the indexing could be different between the GEMM output, i.e. `template_buffer`, which is
    2D with MxN) and the output from the template after epilogues, i.e. `Y`. In the GEMM template code,
    we are not aware of the dimension and the indexing of the epilogues and always work on 2D tiles according to
    the indexing of the GEMM output.
    In this function, we return a 2D buffer (`Y_2d`) according to GEMM output (reinterpreted from `Y` if needed) and
    build a reindexer that converts the indexing of `Y` into `Y_2d`.
    """
    Y_2d: Union[ir.Buffer, ir.ReinterpretView] = Y
    if (
        Y.get_size() == template_buffer.get_size()
        and Y.get_stride() == template_buffer.get_stride()
    ):
        reindexers.extend(default_reindexers)
        Y_2d = Y
    else:

        def get_reindexer(epilogue_node, default_reindexer=None):
            # From template_buffer to epilogue_node_ordered (ordered by stride decreasingly, in dense format), for example:
            #   template_buffer:
            #       size (324, 512), stride (512, 1)
            #   epilogue_node_ordered (ordered by stride decreasingly, in dense format):
            #       size (1, 18, 18, 512), stride (165888, 9216, 512, 1)
            stride_order = list(
                ir.get_stride_order(
                    V.graph.sizevars.size_hints(epilogue_node.get_stride())
                )
            )
            fill_order = ir.stride_order2fill_order(stride_order)
            reversed_fill_order = list(reversed(fill_order))
            size_with_stride_ordered_decreasingly = [
                epilogue_node.get_size()[i] for i in reversed_fill_order
            ]
            reshape_reindex = ir.View.dynamic_reshape_indexer(
                size_with_stride_ordered_decreasingly,
                template_buffer.get_size(),
            )
            if default_reindexer:
                reshape_reindex = ir.fuse_reindexing(reshape_reindex, default_reindexer)

            # From epilogue_node_ordered (ordered by stride decreasingly, in dense format) to epilogue_node, for example:
            #   epilogue_node_ordered (ordered by stride decreasingly, in dense format):
            #       size (1, 18, 18, 512), stride (165888, 9216, 512, 1)
            #   epilogue_node:
            #       size (1, 18, 18, 512), stride (165888, 1, 9216, 512)
            from_stride_ordered_decreasingly_to_epilogue_node_order = [
                (len(stride_order) - 1) - stride_order[i]
                for i in range(len(stride_order))
            ]
            stride_reindex = ir.same_reorder(
                from_stride_ordered_decreasingly_to_epilogue_node_order
            )

            reindexer = ir.fuse_reindexing(stride_reindex, reshape_reindex)
            return reindexer

        if default_reindexers is None:
            default_reindexers = [None] * len(epilogue_nodes)
        new_reindexers = [
            get_reindexer(epilogue_node, default_reindexer)
            for epilogue_node, default_reindexer in zip(
                epilogue_nodes, default_reindexers
            )
        ]
        reindexers.extend(new_reindexers)
        if isinstance(Y, ir.BaseView):
            storage = ir.StorageBox(Y.unwrap_view())
        else:
            assert isinstance(Y, ir.Buffer)
            storage = ir.StorageBox(Y)
        Y_2d = ir.ReinterpretView(data=storage, layout=template_buffer.get_layout())
    return Y_2d, reindexers


class CppGemmTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
        should_block_weights: bool = True,
        name="packed_gemm",
    ) -> None:
        assert layout.dtype in [torch.float, torch.bfloat16, torch.half, torch.uint8]
        super().__init__(
            name,
            input_nodes,
            layout,
            num_threads,
            epilogue_creator=epilogue_creator,
        )
        self.beta = beta
        self.alpha = alpha
        self.has_bias = has_bias
        self.register_blocking = register_blocking
        m, n = layout.size[-2:]
        k = input_nodes[0].get_size()[-1]
        self.m, self.n, self.k = m, n, k
        self.padded_n = get_padded_n(n, self.register_blocking.block_n)
        self.is_dynamic_M = has_free_symbols((m,))
        self.should_block_weights = should_block_weights
        self.thread_blocking = self.make_thread_blocking_cache()
        self.cache_blocking = self.make_cache_blocking_cache()

    def make_thread_blocking_cache(self):
        cache = lru_cache()(self._thread_blocking)

        def thread_blocking(num_threads: int) -> GemmBlocking:
            return cache(num_threads)

        return thread_blocking

    def _thread_blocking(self, num_threads: int) -> GemmBlocking:
        """
        NOTE [Thread blocking in Cpp GEMM]
        We use simple heuristics to decide the thread blocking:
        1. Make sure all threads are occupied as much as possible.
        2. For (m, n) blocks, favor more square-sized thread blocks for better data reuse.
        3. If (m, n) blocks cannot occupy all the threads, we consider k-slicing.
        TODO(jgong5): allow tuning various blocking options
        """

        def get_factors(number):
            factors = []
            for i in range(int(number**0.5), 0, -1):
                if number % i == 0:
                    factors.append(number // i)
                    factors.append(i)
            return factors

        def get_blocking(m_factor, n_factor, k_factor, m_blocks, n_blocks, k_blocks):
            thread_block_k = math.ceil(k_blocks / k_factor)
            thread_block_n = math.ceil(n_blocks / n_factor)
            thread_block_m = math.ceil(m_blocks / m_factor)
            return GemmBlocking(thread_block_m, thread_block_n, thread_block_k)

        assert (
            not self.is_dynamic_M
        ), "Unable to determine thread blocking for dynamic M."
        register_blocking = self.register_blocking
        m_blocks = math.ceil(self.m / register_blocking.block_m)
        n_blocks = math.ceil(self.n / register_blocking.block_n)
        k_blocks = math.ceil(self.k / register_blocking.block_k)
        factors = get_factors(num_threads)
        assert len(factors) > 0

        if config.cpp.gemm_thread_factors is not None:
            factors = [int(i) for i in config.cpp.gemm_thread_factors.split(",")]
            assert len(factors) == 3
            assert math.prod(factors) == self.num_threads
            return get_blocking(
                factors[0], factors[1], factors[2], m_blocks, n_blocks, k_blocks
            )

        # we favor square-sized thread blocks for good data reuse
        def get_better_blocking(blocking, best_blocking):
            if best_blocking is None:
                best_blocking = blocking
            else:
                block_m_size = blocking.block_m * register_blocking.block_m
                block_n_size = blocking.block_n * register_blocking.block_n
                best_block_m_size = best_blocking.block_m * register_blocking.block_m
                best_block_n_size = best_blocking.block_n * register_blocking.block_n
                if blocking.block_k > best_blocking.block_k:
                    best_blocking = blocking
                elif (
                    blocking.block_k == best_blocking.block_k
                    and block_m_size + block_n_size
                    < best_block_m_size + best_block_n_size
                ):
                    best_blocking = blocking
            return best_blocking

        best_blocking = None
        # check if we can have a thread-blocking to occupy all threads without k-slicing
        for n_factor in factors:
            m_factor = num_threads // n_factor
            if n_blocks >= n_factor and m_blocks >= m_factor:
                blocking = get_blocking(
                    m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks
                )
                best_blocking = get_better_blocking(blocking, best_blocking)

        if best_blocking is None:
            for k_factor in factors:
                if k_blocks >= k_factor and (
                    config.cpp.gemm_max_k_slices == 0
                    or k_factor <= config.cpp.gemm_max_k_slices
                ):
                    n_factors = get_factors(num_threads // k_factor)
                    for n_factor in n_factors:
                        m_factor = (num_threads // k_factor) // n_factor
                        if n_blocks >= n_factor and m_blocks >= m_factor:
                            blocking = get_blocking(
                                m_factor,
                                n_factor,
                                k_factor,
                                m_blocks,
                                n_blocks,
                                k_blocks,
                            )
                            best_blocking = get_better_blocking(blocking, best_blocking)

        if best_blocking is None:
            for n_factor in factors:
                m_factor = num_threads // n_factor
                if n_blocks >= n_factor or m_blocks >= m_factor:
                    blocking = get_blocking(
                        m_factor, n_factor, 1, m_blocks, n_blocks, k_blocks
                    )
                    best_blocking = get_better_blocking(blocking, best_blocking)

        assert best_blocking is not None
        return best_blocking

    def make_cache_blocking_cache(self):
        cache = lru_cache()(self._cache_blocking)

        def cache_blocking(num_threads: int) -> GemmBlocking:
            return cache(num_threads)

        return cache_blocking

    def _cache_blocking(self, num_threads: int) -> GemmBlocking:
        def get_cache_blocking(register_blocking, thread_blocking):
            Mr = register_blocking.block_m
            Nr = register_blocking.block_n
            Kr = register_blocking.block_k

            Mt_blocks = thread_blocking.block_m
            Nt_blocks = thread_blocking.block_n
            Kt_blocks = thread_blocking.block_k

            if config.cpp.gemm_cache_blocking is not None:
                blockings = [int(i) for i in config.cpp.gemm_cache_blocking.split(",")]
                assert len(blockings) == 3
                Mc_blocks, Nc_blocks, Kc_blocks = blockings
                return (
                    min(Mc_blocks, Mt_blocks),
                    min(Nc_blocks, Nt_blocks),
                    min(Kc_blocks, Kt_blocks),
                )

            # The ratios below are empirically determined to decide
            # the effective sizes of L1 and L2.
            # TODO: tune the factor here
            L1_limit_factor = 0.8
            L2_limit_factor = 0.5

            L1_cache_size = (
                torch._C._cpu._L1d_cache_size()
            )  # per core cache size in Bytes
            assert (
                L1_cache_size > 0
            ), f"Expect L1_cache_size > 0 but got {L1_cache_size}"
            L1 = L1_cache_size * L1_limit_factor

            L2_cache_size = (
                torch._C._cpu._L2_cache_size()
            )  # per core cache size in Bytes
            assert (
                L2_cache_size > 0
            ), f"Expect L2_cache_size > 0 but got {L2_cache_size}"
            L2 = L2_cache_size * L2_limit_factor

            def get_num_byte(dtype):
                return torch.tensor([], dtype=dtype).element_size()

            dtype_A = self.input_nodes[0].get_dtype()
            dtype_B = self.input_nodes[1].get_dtype()
            num_byte_A = get_num_byte(dtype_A)
            num_byte_B = get_num_byte(dtype_B)
            if dtype_A is torch.bfloat16 and dtype_B is torch.int8 and Kr != 1:
                # We will cache dequantized weights (BF16) in L1D for AMX micro-kernel.
                # In this case, the choice of the micro-kernel being used can't be decoupled from
                # the cache blocking.
                # TODO: Decouple the choice of micro-kernel from cache blocking
                num_byte_B *= num_byte_A

            # NOTE [CPP GEMM Cache Blocking Algorithm]
            # Our overall strategy is to
            # 1) Make cache blocks of B L1-reside and reused by multiple rows of A, i.e. Mc.
            #    Here, B is Kc x Nr where Nr is a single register block. We use L1 size to
            #    decide Kc. We want to make Mc large enough to better reuse B.
            # 2) Make cache blocks of A L2-reside, which would limit Mc. We want to reuse A
            #    along N, where we have two sub-strategies (see notes below) to decide Mc and Nc.

            # Step 1: Decide Kc assuming B block is L1-reside.
            size_cache_B = Kr * Kt_blocks * Nr * num_byte_B

            Kc_blocks = Kt_blocks
            if size_cache_B > L1:
                Kc_blocks = math.floor(L1 / (Kr * Nr * num_byte_B))

            # Step 2: Decide Mc assuming A block is L2-reside.
            min_Mc_ratio = 2  # TODO(jgong5): something to tune?
            min_Mc_blocks = math.ceil(min_Mc_ratio * Mr / Nr)
            assert min_Mc_blocks >= 1
            Kt_bytes = Kt_blocks * Kr * num_byte_A
            if min_Mc_blocks * Mr * Kt_bytes < L2:
                # Strategy 1: A (Mc x Kt) resides in L2 and reused by all Nt
                # when Nc_blocks is kept 1. Mc should be large enough (>= min_Mc_blocks)
                # to reuse B (Kc x Nr) in L1. This makes C (Mc x Nr) small enough to reside
                # in L1.
                Mc_blocks = min(Mt_blocks, math.floor(L2 / (Mr * Kt_bytes)))
                Nc_blocks = 1
            else:
                # Strategy 2: Kt is too large to hold A (Mc x Kt) in L2, we reuse
                # A (Mc x Kc) in L2 by B (Kc x Nc). C (Mc x Nc) resides in L2.
                Mc_blocks = Mt_blocks
                Nc_blocks = min(math.ceil(Mc_blocks * Mr / Nr), Nt_blocks)
                Nc_bytes = Nc_blocks * Nr * 4  # assume C or acc is float32/int32
                Kc_bytes = Kc_blocks * Kr * num_byte_A
                if Mc_blocks * Mr * (Kc_bytes + Nc_bytes) > L2:
                    # The following is the solution for 4*Mc*Nc + Mc*Kc_bytes = L2,
                    # assuming Mc == Nc for good data reuse.
                    M_max = (math.sqrt(Kc_bytes * Kc_bytes + 16 * L2) - Kc_bytes) / 8
                    if M_max < Mc_blocks * Mr:
                        Mc_blocks = math.floor(M_max / Mr)
                        Nc_blocks = min(math.ceil(Mc_blocks * Mr / Nr), Nt_blocks)

            return Mc_blocks, Nc_blocks, Kc_blocks

        assert (
            not self.is_dynamic_M
        ), "Unable to determine cache blocking for dynamic M."
        register_blocking = self.register_blocking
        thread_blocking = self.thread_blocking(num_threads)

        return GemmBlocking(*get_cache_blocking(register_blocking, thread_blocking))

    def log_blockings(self):
        log.debug(f"Register blocking: {self.register_blocking}")  # noqa: G004
        if self.is_dynamic_M:
            # thread and cache blockings are determined at runtime for dynamic shapes
            return
        log.debug(
            f"Cache blocking: {self.cache_blocking(self.num_threads)}"  # noqa: G004
        )
        thread_blocking = self.thread_blocking(self.num_threads)
        log.debug(f"Thread blocking: {thread_blocking}")  # noqa: G004

        def get_occupancy():
            m_blocks = math.ceil(self.m / self.register_blocking.block_m)
            n_blocks = math.ceil(self.n / self.register_blocking.block_n)
            k_blocks = math.ceil(self.k / self.register_blocking.block_k)
            m = math.ceil(m_blocks / thread_blocking.block_m)
            n = math.ceil(n_blocks / thread_blocking.block_n)
            k = math.ceil(k_blocks / thread_blocking.block_k)
            return (m, n, k)

        log.debug(
            f"Number of threads: {self.num_threads}, occupancy: {get_occupancy()}"  # noqa: G004
        )

    def maybe_k_slicing(self):
        if self.num_threads == 1:
            return False
        if self.is_dynamic_M:
            # TODO(jgong5): perhaps use size hint to decide?
            return True
        register_blocking = self.register_blocking
        k_blocks = math.ceil(self.k / register_blocking.block_k)
        thread_blocking = self.thread_blocking(self.num_threads)
        return k_blocks > thread_blocking.block_k

    @classmethod
    def add_choices(
        cls,
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        has_bias=False,
        trans_w=False,
        input_indices=None,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
        da8w8_gemm_sym_act=False,
    ):
        if input_indices is None:
            input_indices = list(range(len(input_nodes)))
        only_one_input = (
            input_nodes[0] == input_nodes[1] if len(input_nodes) > 1 else False
        )

        def reorder_and_filter(inputs, layout_or_out):
            if has_bias:
                assert len(input_indices) >= 3
                # Assume the input order is [inp, x, w] and we reorder it to [x, w, inp]
                inp_idx = input_indices[0]
                x_idx = input_indices[1]
                w_idx = input_indices[2]
                return [
                    inputs[x_idx],
                    inputs[w_idx],
                    inputs[inp_idx],
                    *[inputs[idx] for idx in input_indices[3:]],
                ], layout_or_out
            elif len(inputs) >= len(input_indices):
                assert len(input_indices) >= 2
                return [inputs[idx] for idx in input_indices], layout_or_out
            else:
                # For when input is used for x and w, i.e. X@X.T or similar
                # Assumes the first input is the only input
                assert len(inputs) == 1
                return [inputs[0]] * len(input_indices), layout_or_out

        new_inputs, new_layout = reorder_and_filter(input_nodes, layout)
        is_mkldnn_wgt = (
            new_inputs[1].get_name() in V.graph.constants
            and V.graph.constants[new_inputs[1].get_name()].is_mkldnn
        )
        if is_mkldnn_wgt:
            # It shouldn't happen as viewing an mkldnn tensor, we can extend the
            # implementation if it does.
            assert not isinstance(new_inputs[1], ir.BaseView)
        # Note that the layout of MKLDNN Tensor is with the wrong stride
        view_size = new_inputs[1].layout.size
        view_stride = new_inputs[1].layout.stride
        view_offset = new_inputs[1].layout.offset

        def maybe_to_dense(inputs, layout_or_out):
            new_inputs = list(inputs)
            if isinstance(inputs[1], torch.Tensor):
                W = inputs[1]
                new_inputs[1] = W.to_dense() if W.is_mkldnn else W
            return new_inputs, layout_or_out

        def normalize_shapes(inputs, layout_or_out):
            new_inputs = list(inputs)
            if not is_mkldnn_wgt and isinstance(new_inputs[1], torch.Tensor):
                if view_size[0].is_symbol:
                    # If batch size B is dynamic, we need to infer the batch size from the input
                    assert all(not dim.is_symbol for dim in view_size[1:])
                    size = torch.tensor(new_inputs[1].size()).prod()
                    fixed_size = torch.tensor(view_size[1:], dtype=torch.int).prod()
                    view_size[0] = (size // fixed_size).item()
                # With the assumptation that W is the storage of unwrap view
                # thus view it back here
                new_inputs[1] = new_inputs[1].as_strided(
                    view_size, view_stride, view_offset
                )

            if not trans_w:
                return new_inputs, layout_or_out
            X = new_inputs[0]
            W = new_inputs[1]
            B = new_inputs[2] if has_bias else None
            W = transpose_w(W, trans_w)
            B = expand_bias(B, X)  # type:ignore[arg-type]
            new_inputs[1] = W
            if B is not None:
                new_inputs[2] = B
            return new_inputs, layout_or_out

        # TODO(jgong5): decide proper number of threads per problem size
        num_threads = parallel_num_threads()
        new_inputs, _ = normalize_shapes(*maybe_to_dense(new_inputs, new_layout))
        m, n, k, *_ = mm_args(new_inputs[0], new_inputs[1])
        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            new_inputs[0].get_dtype()
        )
        micro_gemm = create_micro_gemm(
            "micro_gemm",
            m,
            n,
            k,
            input_dtype=new_inputs[0].get_dtype(),
            input2_dtype=new_inputs[1].get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=alpha,
            num_threads=num_threads,
        )
        assert micro_gemm is not None
        block_weights = cls.check_if_block_weight(new_inputs[1], micro_gemm)

        def preprocessor(inputs, layout):
            new_inputs, new_layout = normalize_shapes(
                *maybe_to_dense(*reorder_and_filter(inputs, layout))
            )
            if only_one_input and isinstance(new_inputs[0], torch.Tensor):
                return new_inputs[1:], new_layout
            return cls.prep_weight(
                new_inputs, new_layout, micro_gemm, block_weights, da8w8_gemm_sym_act
            )

        def postprocessor(output):
            if isinstance(output, ir.TensorBox):
                # prepack the weight as input to the template buffer
                template_buffer = ir.InputsKernel.unwrap_storage_for_input(output)
                assert isinstance(template_buffer, ir.CppTemplateBuffer)
                new_input_nodes, _ = reorder_and_filter(input_nodes, layout)

                W_node = new_input_nodes[1]
                if W_node.get_name() not in V.graph.constants:
                    return output
                W = V.graph.constants[W_node.get_name()]
                new_input_nodes[1] = W
                new_input_nodes, new_layout = normalize_shapes(
                    *maybe_to_dense(new_input_nodes, layout)
                )
                new_input_nodes, _ = cls.prep_weight(
                    new_input_nodes,
                    new_layout,
                    micro_gemm,
                    block_weights,
                    da8w8_gemm_sym_act,
                )
                W_packed = new_input_nodes[1]
                W_packed_constant = V.graph.add_tensor_constant(W_packed)
                new_input_nodes[1] = W_packed_constant

                # Prune unused tensors
                prune_tensors(input_nodes, new_input_nodes)

                template_buffer.inputs[1] = ir.InputsKernel.unwrap_storage_for_input(
                    W_packed_constant
                )
            return output

        template = DataProcessorTemplateWrapper(
            cls,
            preprocessor,
            postprocessor,
            input_nodes=input_nodes,
            layout=layout,
            num_threads=num_threads,
            register_blocking=micro_gemm.register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            should_block_weights=block_weights,
        )
        template.maybe_append_choice(choices)
        return template

    @staticmethod
    def get_padded_size(n, block_n, k, should_block_weight):
        padded_n = get_padded_n(n, block_n)
        # We assume that all GEMM weight tensors should be blocked and padded
        new_size = [padded_n // block_n, k, block_n]
        return new_size, padded_n

    @classmethod
    def prep_weight(
        cls,
        inputs,
        layout: ir.Layout,
        micro_gemm: CppMicroGemm,
        should_block_weight: bool,
        da8w8_gemm_sym_act: bool,
    ):
        """
        NOTE Weight prep consists of 2 separate steps:
        1. Blocking the weight tensor into a 3D shape: [n//block_n, k, block_n]
           This is always done if the weight tensor is contant, i.e. for all GEMM and some BMM.
           For BMM, we also block non-contiguous weight tensors, since they would be reshaped anyway.
           This assumes that blocked, contiguous weights will be more efficient for the GEMM kernel,
           and is worth the overhead of reshape and blocking.

           This blocking includes additional padding, when n is not a multiple of block_n.
           This padding allows a more efficient microkernel implementation. For BMM, this is only done
           if reshape would happen anyway, i.e.  if the weight tensor is constant, is not contiguous,
           or is using AMX VNNI layout.
        2. Packing the weight tensor into a VNNI-friendly shape. For constant input,
           this is done at the same time as the weight blocking.

        At compile time, the constant weight tensors are blocked and packed. For non-constant tensors (e.g. BMM)
        which will be blocked (non-contiguous or VNNI-layout tensors), the weight tensor is blocked and packed at runtime.

        CppBmmTemplate overrides the methods get_padded_size, and block_weight in order to accommodate
        an additional dimension for the batch size and to determine if the weight tensor should be blocked.
        """
        W = inputs[1]
        new_inputs = list(inputs)
        if isinstance(W, ir.IRNode):
            k, n = W.get_size()[-2:]
        else:
            k, n = W.shape[-2:]
        _, block_n, _ = micro_gemm.register_blocking
        new_size, padded_n = cls.get_padded_size(n, block_n, k, should_block_weight)
        padding = padded_n - n

        if should_block_weight:
            blocked_w = cls.block_weight(W, new_size, padding)
        else:
            blocked_w = W
        new_inputs[1] = cls.pack_vnni_weight(blocked_w, micro_gemm, new_size)

        def _is_int8_gemm(inputs):
            return (
                isinstance(inputs[0], ir.IRNode)
                and inputs[0].get_dtype() in [torch.uint8, torch.int8]
            ) or (
                isinstance(inputs[0], torch.Tensor)
                and inputs[0].dtype in [torch.uint8, torch.int8]
            )

        if _is_int8_gemm(new_inputs) and not da8w8_gemm_sym_act:
            BCompensate = None
            if isinstance(W, ir.IRNode):
                BCompensate = V.graph.add_tensor_constant(
                    V.graph.constants[W.get_name() + "_BMatrixCompens"],
                    W.get_name() + "_BMatrixCompens",
                )
            else:
                # Use the original W, not the blocked_w in new_inputs[1] to calculate BCompensate
                BCompensate = torch.sum(W.to_dense().to(torch.float), dim=0)  # type: ignore[assignment]
            new_inputs.append(BCompensate)
        return new_inputs, layout

    @staticmethod
    def check_if_block_weight(W, micro_gemm):
        return True

    @classmethod
    def block_weight(cls, W, new_size, padding):
        # These are separated into two methods to allow subclasses to override them separately
        if isinstance(W, ir.IRNode):
            if W.get_name() in V.graph.constants:
                # Create a new buffer, representing the constant blocked tensor
                blocked_w = ir.Buffer(
                    name=W.get_name(),  # Borrow the registered buffer name
                    layout=ir.FixedLayout(
                        W.get_device_or_error(),
                        W.get_dtype(),
                        new_size,
                        ir.FlexibleLayout.contiguous_strides(new_size),
                        0,
                    ),
                )
            else:
                if not isinstance(W, ir.TensorBox):
                    W = ir.TensorBox(W)
                permute_dims = list(range(len(new_size)))
                permute_dims[-2], permute_dims[-3] = permute_dims[-3], permute_dims[-2]
                permute_size = list(new_size)
                permute_size[-2], permute_size[-3] = permute_size[-3], permute_size[-2]
                blocked_w = L.constant_pad_nd(W, (0, padding))
                blocked_w = L.permute(
                    L.view(blocked_w, permute_size),
                    permute_dims,
                )
        else:
            assert isinstance(W, torch.Tensor)
            # Pad the weight tensor and reshape it into a 3D blocked shape
            blocked_size = list(new_size)
            blocked_size[-2], blocked_size[-3] = blocked_size[-3], blocked_size[-2]
            blocked_w = (
                torch.nn.functional.pad(W, (0, padding))  # type: ignore[assignment]
                .reshape(*blocked_size)
                .transpose(-3, -2)
                .contiguous()
            )
        return blocked_w

    @classmethod
    def pack_vnni_weight(cls, W, micro_gemm, new_size):
        # These are separated into two methods to allow subclasses to override them separately
        if isinstance(W, ir.IRNode):
            if isinstance(W, ir.Buffer) and W.get_name() in V.graph.constants:
                return W
            k = new_size[-2]
            if not isinstance(W, ir.TensorBox):
                W = ir.TensorBox(W)
            if micro_gemm.get_b_layout() != LayoutType.NORMAL:
                permute_dims = list(range(len(new_size) + 1))
                permute_dims[-1], permute_dims[-2] = permute_dims[-2], permute_dims[-1]
                vnni_size = 4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
                vnni_view_size = list(new_size)
                vnni_view_size[-2] = k // vnni_size
                vnni_view_size.insert(-1, vnni_size)
                W = L.view(
                    L.permute(L.view(W, vnni_view_size), permute_dims),
                    new_size,
                )
            W = ir.ExternKernel.realize_input(W)
            W = ir.ExternKernel.require_contiguous(W)
            return W
        else:
            k = new_size[-2]
            # Apply VNNI packing to the weight tensor
            if micro_gemm.get_b_layout() != LayoutType.NORMAL:
                # TODO: Move VNNI weight packing for non-constant tensors into the template,
                # to improve cache locality and avoid full-tensor copy.
                layout_str = (
                    "VNNI4"
                    if micro_gemm.get_b_layout() == LayoutType.VNNI4
                    else "VNNI2"
                )
                assert micro_gemm.get_b_layout() in [
                    LayoutType.VNNI2,
                    LayoutType.VNNI4,
                ], f"We only support {layout_str} for now"
                vnni_size = 4 if micro_gemm.get_b_layout() == LayoutType.VNNI4 else 2
                assert (
                    k % vnni_size == 0
                ), f"k should be divisible by vnni_size for {layout_str} layout"
                vnni_view_size = list(new_size)
                vnni_view_size[-2] = k // vnni_size
                vnni_view_size.insert(-1, vnni_size)
                W = W.view(vnni_view_size).transpose(-1, -2).contiguous().view(new_size)
            # normalize stride to be "contiguous_strides" per size
            # this avoids the problems in L.view during template codegen
            new_stride = [1]
            for sz in reversed(W.shape[1:]):
                new_stride.insert(0, new_stride[0] * sz)
            W = W.as_strided(W.shape, new_stride)
            return W

    def get_default_reindexers(self, epilogue_nodes):
        return [None] * len(epilogue_nodes)

    def get_options(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
    ) -> Dict[str, Any]:
        assert len(self.input_nodes) >= 2

        int8_gemm = self.input_nodes[0].get_dtype() in [torch.uint8, torch.int8]
        x_scale = None
        x_zp = None
        w_scale = None
        w_zp = None
        if int8_gemm:
            X, W = self.input_nodes[0], self.input_nodes[1]
            bias_idx = 2 if self.has_bias else 1
            inp = self.input_nodes[bias_idx] if self.has_bias else None
            x_scale = self.input_nodes[bias_idx + 1]
            x_zp = self.input_nodes[bias_idx + 2]
            w_scale = self.input_nodes[bias_idx + 3]
            w_zp = self.input_nodes[bias_idx + 4]
            Y = self.output_node
        else:
            X, W = self.input_nodes[0], self.input_nodes[1]
            Y = self.output_node
            inp = self.input_nodes[2] if self.has_bias else None

        template_buffer_has_other_users = None

        if template_buffer_node is not None:
            # Use the updated prepacked weight buffer
            W = template_buffer_node.inputs[1]
            Y = template_buffer_node

            assert flag_template_buffer_has_other_users is not None
            template_buffer_has_other_users = flag_template_buffer_has_other_users

        template_buffer = Y
        gemm_output_buffer = template_buffer

        epilogues: List[ir.IRNode] = []
        reindexers: List[Optional[Callable[[List[Any]], List[Any]]]] = []
        epilogue_creators: List[Callable[[ir.Buffer], ir.Pointwise]] = []
        fake_buffers: List[ir.Buffer] = []
        Y_aliases = OrderedSet[str]()

        use_local_acc = (
            self.layout.dtype != torch.float
            or template_buffer_has_other_users
            or int8_gemm
            or self.padded_n != self.n
            or self.maybe_k_slicing()
        )

        # TODO(jgong5): for int8 gemm, bias-add is handled outside of gemm template,
        # but we'd better move it here to align with fp.
        if inp is not None and self.beta != 0 and not int8_gemm:
            # add an epilogue for bias add
            def _bias_add_epilogue(buf):
                return create_epilogue_with_attr(
                    buf, "bias_add", other=inp, beta=self.beta, dtype=self.layout.dtype
                )

            epilogue_creators.append(_bias_add_epilogue)

        if self.epilogue_creator is not None:
            epilogue_creators.append(self.epilogue_creator)

        # When the GEMM output buffer is localized but it has users other than the epilogue nodes,
        # we need to copy the value in the GEMM output local buffer to a global buffer.
        def need_copy_from_local_to_global_buffer_epilogue(
            use_local_acc, template_buffer_has_other_users, epilogue_creators
        ):
            # The GEMM output buffer is a global buffer, thus copy is not needed.
            if not use_local_acc:
                return False

            # The possible value of template_buffer_has_other_users is (None, False, True)
            # It is None when generating the gemm template during autotune and it will have value during scheduler codegen.
            # extra copy_from_local_to_global_buffer_epilogue is not needed in either of the below two cases:
            #   1. template_buffer_has_other_users is None (i.e. when doing the codegen during autotune)
            #   2. template_buffer_has_other_users is False, which means it's safe to keep the value in the
            #       GEMM output buffer in local buffer only (no users outside of the epilogues will use its value).
            if not template_buffer_has_other_users:
                return False

            # When bias is not None or self.epilogue_creator is not None,
            # there will be epilogue_creators after the GEMM.
            # The GEMM output buffer is localized while
            # the output buffer of the epilogue_creators is a global buffer.
            if epilogue_creators:
                return False

            return True

        if need_copy_from_local_to_global_buffer_epilogue(
            use_local_acc, template_buffer_has_other_users, epilogue_creators
        ):

            def copy_from_local_to_global_buffer_epilogue(input_buffer: ir.Buffer):
                dtype = self.layout.dtype
                input_loader = input_buffer.make_loader()

                def copy_inner(index):
                    input = input_loader(index)
                    result = ops.to_dtype(input, dtype)
                    return result

                return ir.Pointwise(
                    device=input_buffer.get_device_or_error(),
                    dtype=self.layout.dtype,
                    inner_fn=copy_inner,
                    ranges=input_buffer.get_size(),
                )

            epilogue_creators.append(copy_from_local_to_global_buffer_epilogue)

        # NOTE [How CPP GEMM template epilogues are organized]
        #   gemm_output_buffer
        #     --> zero or more in-template epilogues (created by `epilogue_creators`) -->
        #   template_buffer
        #     --> zero or more out-of-template epilogues (`epilogue_nodes`) -->
        #   Y
        if epilogue_creators:
            gemm_output_name = f"{template_buffer.get_name()}_GemmOut"
            gemm_output_buffer = ir.Buffer(
                name=gemm_output_name, layout=template_buffer.layout
            )
            current_input_buffer = gemm_output_buffer
            for i, creator in enumerate(epilogue_creators):
                if i == len(epilogue_creators) - 1:
                    buffer_name = template_buffer.get_name()
                else:
                    buffer_name = f"{gemm_output_name}_epilogue_{i}"
                epilogues.append(
                    ir.ComputedBuffer(
                        name=buffer_name,
                        layout=template_buffer.layout,
                        data=creator(current_input_buffer),
                    )
                )
                fake_buffers.append(current_input_buffer)
                Y_aliases.add(current_input_buffer.get_name())
                reindexers.append(None)
                if i < len(epilogue_creators) - 1:
                    current_input_buffer = ir.Buffer(
                        name=buffer_name, layout=template_buffer.layout
                    )

        Y_2d: Union[ir.Buffer, ir.ReinterpretView] = Y

        if epilogue_nodes:
            if not template_buffer_has_other_users:
                Y_aliases.add(template_buffer.get_name())
            epilogues.extend(epilogue_nodes)
            assert Y.get_numel() == epilogues[-1].get_numel()
            Y = cast(ir.Buffer, epilogues[-1])
            Y_2d, reindexers = gen_2d_view_of_epilogue_buf(
                Y,
                template_buffer,
                epilogue_nodes,
                reindexers,
                default_reindexers=self.get_default_reindexers(epilogue_nodes),
            )

        output_dtype, compute_dtype = get_gemm_template_output_and_compute_dtype(
            X.get_dtype()
        )
        micro_gemm = create_micro_gemm(
            f"{kernel.kernel_name}_micro_gemm",
            self.m,
            self.n,
            self.k,
            input_dtype=X.get_dtype(),
            input2_dtype=W.get_dtype(),
            output_dtype=output_dtype,
            compute_dtype=compute_dtype,
            alpha=self.alpha,
            num_threads=self.num_threads,
        )
        assert micro_gemm is not None
        assert self.register_blocking == micro_gemm.register_blocking
        self.log_blockings()
        if isinstance(micro_gemm, CppMicroGemmAMX):
            counters["inductor"]["cpp_micro_gemm_amx_counter"] += 1
        if isinstance(micro_gemm, CppMicroBrgemm):
            counters["inductor"]["cpp_micro_brgemm_counter"] += 1

        L1_cache_size = torch._C._cpu._L1d_cache_size()  # per core cache size in Bytes
        assert L1_cache_size > 0, f"Expect L1_cache_size > 0 but got {L1_cache_size}"

        L2_cache_size = torch._C._cpu._L2_cache_size()  # per core cache size in Bytes
        assert L2_cache_size > 0, f"Expect L2_cache_size > 0 but got {L2_cache_size}"

        options = dict(
            X=X,
            W=W,
            inp=inp,
            Y=Y,
            N=self.n,
            K=self.k,
            PADDED_N=self.padded_n,
            GemmOut=gemm_output_buffer,
            aliases={alias: Y.get_name() for alias in Y_aliases},
            beta=self.beta,
            alpha=self.alpha,
            num_threads=self.num_threads,
            micro_gemm=micro_gemm,
            is_dynamic_M=self.is_dynamic_M,
            template=self,
            kernel=kernel,
            export_declaration=get_export_declaration(),
            epilogue_nodes=epilogues,
            reindexers=reindexers,
            Y_2d=Y_2d,
            use_local_acc=use_local_acc,
            maybe_k_slicing=self.maybe_k_slicing(),
            x_scale=x_scale,
            x_zp=x_zp,
            w_scale=w_scale,
            w_zp=w_zp,
            acc_buf_dtype=torch.int32 if int8_gemm else torch.float,
            DTYPE_TO_CPP=DTYPE_TO_CPP,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            config=config,
            fake_buffers=fake_buffers,
        )
        return options

    def render(  # type: ignore[override, return]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        flag_template_buffer_has_other_users: Optional[bool] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        options = self.get_options(
            kernel=kernel,
            template_buffer_node=template_buffer_node,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_nodes,
        )
        self.render_options = options

        with contextlib.ExitStack() as stack:
            for buf in options["fake_buffers"]:
                stack.enter_context(
                    patch.object(V.graph, "get_dtype", self._fake_get_dtype(buf))
                )
            return self._template_from_string(GEMM_TEMPLATE).render(**options)

    def codegen_blocks(
        self,
        num_threads,
        N,
        K,
        micro_gemm,
        is_dynamic_M,
        kernel,
        GemmOut,
        config,
        L1_cache_size,
        L2_cache_size,
        X,
        W,
    ):
        options = dict(
            num_threads=num_threads,
            N=N,
            K=K,
            micro_gemm=micro_gemm,
            is_dynamic_M=is_dynamic_M,
            kernel=kernel,
            GemmOut=GemmOut,
            config=config,
            L1_cache_size=L1_cache_size,
            L2_cache_size=L2_cache_size,
            template=self,
            X=X,
            W=W,
        )
        return self._template_from_string(GEMM_TEMPLATE_INIT_BLOCKING).render(options)

    def codegen_microkernel_def(self):
        return self._template_from_string(GEMM_TEMPLATE_MICROKERNEL_DEF).render(
            self.render_options
        )

    def codegen_gemm_stub_def(self):
        microkernel = self.codegen_microkernel_def()
        return microkernel + self._template_from_string(GEMM_TEMPLATE_STUB_DEF).render(
            self.render_options
        )

    def codegen_multi_threads_params(self):
        return self._template_from_string(GEMM_TEMPLATE_MULTI_THREADS_PARAMS).render()

    def codegen_single_thread_params(self, is_dynamic_M):
        options = dict(
            is_dynamic_M=is_dynamic_M,
        )
        return self._template_from_string(GEMM_TEMPLATE_SINGLE_THREAD_PARAMS).render(
            options
        )

    def codegen_m_loop_params(self):
        return self._template_from_string(GEMM_TEMPLATE_M_LOOP_PARAMS).render()

    def codegen_n_loop_params(self):
        return self._template_from_string(GEMM_TEMPLATE_N_LOOP_PARAMS).render()
