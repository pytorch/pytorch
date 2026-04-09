# mypy: allow-untyped-defs
"""Metal shader template for flex attention on MPS."""

import itertools
import logging
from typing import Any

import torch

from .. import ir
from ..virtualized import V


log = logging.getLogger(__name__)


METAL_DTYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16, "bfloat", #TODO check if this works
}


def _fx_graph_to_metal(
    graph_module: torch.fx.GraphModule,
    fixed_inputs: dict[str, str],
    captured_buffer_args: dict[str, str],
    output_var: str,
    var_prefix: str = "_sm",
) -> str:
    """Compile an FX GraphModule to inline Metal code.

    Args:
        graph_module: The traced score_mod or mask_mod FX graph
        fixed_inputs: Map from placeholder names to Metal variable names
            e.g. {"score": "score_val", "b": "b_idx", ...}
        captured_buffer_args: Map from captured buffer FX node names to
            Metal pointer variable names
        output_var: Metal variable name to assign the result to

    Returns:
        Metal code string
    """
    var_map: dict[str, str] = {}
    code_lines: list[str] = []
    tmp_counter = itertools.count()

    def _tmp() -> str:
        return f"{var_prefix}{next(tmp_counter)}"

    def _val(node) -> str:
        if isinstance(node, torch.fx.Node):
            return var_map[node.name]
        # Check bool BEFORE int since isinstance(True, int) is True
        if isinstance(node, bool):
            return "true" if node else "false"
        if isinstance(node, (int, float)):
            if isinstance(node, float):
                if node == float("inf"):
                    return "HUGE_VALF"
                if node == float("-inf"):
                    return "(-HUGE_VALF)"
                if node != node:  # NaN
                    return "NAN"
                return str(node) + "f"
            return str(node)
        return str(node)

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name in fixed_inputs:
                var_map[node.name] = fixed_inputs[node.name]
            elif node.name in captured_buffer_args:
                var_map[node.name] = captured_buffer_args[node.name]
            else:
                # Captured buffer passed as extra arg
                var_map[node.name] = f"/* unknown placeholder {node.name} */"

        elif node.op == "call_function":
            t = _tmp()
            target = node.target
            args = node.args
            kwargs = node.kwargs

            # Arithmetic
            if target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
                code_lines.append(f"float {t} = {_val(args[0])} + {_val(args[1])};")
            elif target in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar):
                code_lines.append(f"float {t} = {_val(args[0])} - {_val(args[1])};")
            elif target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar):
                code_lines.append(f"float {t} = {_val(args[0])} * {_val(args[1])};")
            elif target in (
                torch.ops.aten.div.Tensor,
                torch.ops.aten.div.Scalar,
            ) or (hasattr(target, "__name__") and "true_divide" in str(target)):
                code_lines.append(f"float {t} = {_val(args[0])} / {_val(args[1])};")
            elif target == torch.ops.aten.neg.default:
                code_lines.append(f"float {t} = -{_val(args[0])};")
            elif target in (torch.ops.aten.abs.default,):
                code_lines.append(f"float {t} = metal::abs({_val(args[0])});")
            elif target == torch.ops.aten.clamp.default:
                v = _val(args[0])
                lo = _val(args[1]) if len(args) > 1 and args[1] is not None else None
                hi = _val(args[2]) if len(args) > 2 and args[2] is not None else None
                if lo is not None and hi is not None:
                    code_lines.append(f"float {t} = metal::clamp({v}, {lo}, {hi});")
                elif lo is not None:
                    code_lines.append(f"float {t} = metal::max({v}, {lo});")
                elif hi is not None:
                    code_lines.append(f"float {t} = metal::min({v}, {hi});")
                else:
                    code_lines.append(f"float {t} = {v};")

            # Comparisons
            elif target in (torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Scalar):
                code_lines.append(f"bool {t} = {_val(args[0])} > {_val(args[1])};")
            elif target in (torch.ops.aten.ge.Tensor, torch.ops.aten.ge.Scalar):
                code_lines.append(f"bool {t} = {_val(args[0])} >= {_val(args[1])};")
            elif target in (torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar):
                code_lines.append(f"bool {t} = {_val(args[0])} < {_val(args[1])};")
            elif target in (torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar):
                code_lines.append(f"bool {t} = {_val(args[0])} <= {_val(args[1])};")
            elif target in (torch.ops.aten.eq.Tensor, torch.ops.aten.eq.Scalar):
                code_lines.append(f"bool {t} = {_val(args[0])} == {_val(args[1])};")
            elif target in (torch.ops.aten.ne.Tensor, torch.ops.aten.ne.Scalar):
                code_lines.append(f"bool {t} = {_val(args[0])} != {_val(args[1])};")
            elif target == torch.ops.aten.logical_and.default:
                code_lines.append(f"bool {t} = {_val(args[0])} && {_val(args[1])};")
            elif target == torch.ops.aten.logical_or.default:
                code_lines.append(f"bool {t} = {_val(args[0])} || {_val(args[1])};")
            elif target == torch.ops.aten.logical_not.default:
                code_lines.append(f"bool {t} = !{_val(args[0])};")

            # Math
            elif target == torch.ops.aten.exp.default:
                code_lines.append(f"float {t} = metal::precise::exp({_val(args[0])});")
            elif target == torch.ops.aten.log.default:
                code_lines.append(f"float {t} = metal::precise::log({_val(args[0])});")
            elif target == torch.ops.aten.sqrt.default:
                code_lines.append(f"float {t} = metal::precise::sqrt({_val(args[0])});")
            elif target == torch.ops.aten.rsqrt.default:
                code_lines.append(f"float {t} = metal::precise::rsqrt({_val(args[0])});")
            elif target == torch.ops.aten.tanh.default:
                code_lines.append(f"float {t} = metal::precise::tanh({_val(args[0])});")
            elif target == torch.ops.aten.sin.default:
                code_lines.append(f"float {t} = metal::precise::sin({_val(args[0])});")
            elif target == torch.ops.aten.cos.default:
                code_lines.append(f"float {t} = metal::precise::cos({_val(args[0])});")
            elif target in (
                torch.ops.aten.maximum.default,
                torch.ops.aten.max.other,
            ):
                code_lines.append(
                    f"float {t} = metal::max({_val(args[0])}, {_val(args[1])});"
                )
            elif target in (
                torch.ops.aten.minimum.default,
                torch.ops.aten.min.other,
            ):
                code_lines.append(
                    f"float {t} = metal::min({_val(args[0])}, {_val(args[1])});"
                )

            # Where / select
            elif target == torch.ops.aten.where.self:
                code_lines.append(
                    f"float {t} = {_val(args[0])} ? {_val(args[1])} : {_val(args[2])};"
                )

            # Constants
            elif target in (
                torch.ops.aten.full_like.default,
                torch.ops.aten.full.default,
            ):
                fill = args[1] if len(args) > 1 else kwargs.get("fill_value", 0)
                code_lines.append(f"float {t} = {_val(fill)};")
            elif target == torch.ops.aten.scalar_tensor.default:
                code_lines.append(f"float {t} = {_val(args[0])};")

            # Type conversion (no-op for float computation)
            elif target in (
                torch.ops.aten._to_copy.default,
                torch.ops.aten.to.dtype,
                torch.ops.prims.convert_element_type.default,
            ):
                code_lines.append(f"float {t} = static_cast<float>({_val(args[0])});")

            # Float cast
            elif target == torch.ops.aten.float.default:
                code_lines.append(f"float {t} = static_cast<float>({_val(args[0])});")

            else:
                # Fallback: try simple representation
                log.warning("Unsupported op in score_mod/mask_mod: %s", target)
                code_lines.append(f"float {t} = {_val(args[0])}; /* unsupported: {target} */")

            var_map[node.name] = t

        elif node.op == "get_attr":
            # Constant attribute from the module
            var_map[node.name] = f"/* get_attr {node.target} */"

        elif node.op == "output":
            out_node = node.args[0]
            if isinstance(out_node, (tuple, list)):
                out_node = out_node[0]
            code_lines.append(f"{output_var} = {_val(out_node)};")

    return "\n".join(code_lines)


def _compile_score_mod_to_metal(
    score_mod_graph: torch.fx.GraphModule,
    captured_buf_metal_names: dict[str, str],
    var_prefix: str = "_sm",
) -> str:
    """Compile score_mod FX graph to inline Metal code.

    Placeholders are matched by POSITION (not name):
      0=score, 1=b, 2=h, 3=q_idx, 4=kv_idx, 5+=captured buffers
    """
    metal_vars = ["score_val", "b_idx", "h_idx", "m_idx", "n_idx"]
    fixed = {}
    placeholder_idx = 0
    for node in score_mod_graph.graph.nodes:
        if node.op == "placeholder":
            if placeholder_idx < len(metal_vars):
                fixed[node.name] = metal_vars[placeholder_idx]
            placeholder_idx += 1

    return _fx_graph_to_metal(
        score_mod_graph, fixed, captured_buf_metal_names, "score_val",
        var_prefix=var_prefix,
    )


def _compile_mask_mod_to_metal(
    mask_mod_graph: torch.fx.GraphModule,
    captured_buf_metal_names: dict[str, str],
    var_prefix: str = "_mm",
) -> str:
    """Compile mask_mod FX graph to inline Metal code.

    Placeholders are matched by position:
      0=b, 1=h, 2=q_idx, 3=kv_idx, 4+=captured buffers
    """
    metal_vars = ["b_idx", "h_idx", "m_idx", "n_idx"]
    fixed = {}
    placeholder_idx = 0
    for node in mask_mod_graph.graph.nodes:
        if node.op == "placeholder":
            if placeholder_idx < len(metal_vars):
                fixed[node.name] = metal_vars[placeholder_idx]
            placeholder_idx += 1

    return _fx_graph_to_metal(
        mask_mod_graph, fixed, captured_buf_metal_names, "mask_result",
        var_prefix=var_prefix,
    )


def _generate_mma_shader(
    metal_dtype, d_qk, d_v, kv_dim, block_m, block_n,
    extra_buf_params, full_kv_params, scalar_params_str,
    unpack_code, score_mod_indented, mask_mod_indented,
    full_score_mod_indented, has_full_blocks,
):
    """Generate Metal shader using simdgroup_matrix for Q@K^T.

    Uses hardware 8x8 MMA for the dot-product-heavy Q@K^T, stores scores
    to threadgroup memory, then does per-thread online softmax + V
    accumulation (fused in one pass to minimize threadgroup traffic).
    """

    def _block_loop_body(score_code, mask_code, is_partial):
        """Generate the inner tiled loop body for one block type."""
        mask_section = ""
        if is_partial:
            mask_section = f"""
                    bool mask_result = true;
{mask_code}
                    if (!mask_result) s = -HUGE_VALF;
"""
        return f"""\
        for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_N) {{
            int tile_end = min(tile_start + BLOCK_N, kv_end);
            int tile_size = tile_end - tile_start;

            // Phase 1: Cooperative load K into KV_tg (zero-pad for MMA alignment)
            for (int i = (int)tid; i < BLOCK_N * D_QK; i += BLOCK_M) {{
                int n_local = i / D_QK;
                int d = i % D_QK;
                if (n_local < tile_size) {{
                    int n_global = tile_start + n_local;
                    long k_off = b_idx * stride_kz + hkv_idx * stride_kh + (long)n_global * stride_kn + (long)d * stride_kk;
                    KV_tg[n_local * D_QK + d] = K[k_off];
                }} else {{
                    KV_tg[n_local * D_QK + d] = ({metal_dtype})0;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2: simdgroup MMA — S = Q_tg @ KV_tg^T
            simdgroup_matrix<float, 8, 8> S_frag[BLOCK_M / 8][BLOCK_N / 8];
            for (int ii = 0; ii < BLOCK_M / 8; ii++)
                for (int jj = 0; jj < BLOCK_N / 8; jj++) {{
                    S_frag[ii][jj].thread_elements()[0] = 0.0f;
                    S_frag[ii][jj].thread_elements()[1] = 0.0f;
                }}
            for (int k = 0; k < D_QK / 8; k++) {{
                simdgroup_matrix<{metal_dtype}, 8, 8> K_frags[BLOCK_N / 8];
                for (int j = 0; j < BLOCK_N / 8; j++)
                    simdgroup_load(K_frags[j], &KV_tg[j * 8 * D_QK + k * 8], (ulong)D_QK, ulong2(0), true);
                for (int i = 0; i < BLOCK_M / 8; i++) {{
                    simdgroup_matrix<{metal_dtype}, 8, 8> Q_frag;
                    simdgroup_load(Q_frag, &Q_tg[i * 8 * D_QK + k * 8], (ulong)D_QK);
                    for (int j = 0; j < BLOCK_N / 8; j++)
                        simdgroup_multiply_accumulate(S_frag[i][j], Q_frag, K_frags[j], S_frag[i][j]);
                }}
            }}
            for (int i = 0; i < BLOCK_M / 8; i++)
                for (int j = 0; j < BLOCK_N / 8; j++)
                    simdgroup_store(S_frag[i][j], &S_tg[i * 8 * BLOCK_N + j * 8], (ulong)BLOCK_N);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3: Cooperative load V into KV_tg (overwrites K — safe after MMA)
            for (int i = (int)tid; i < tile_size * D_V; i += BLOCK_M) {{
                int n_local = i / D_V;
                int d = i % D_V;
                int n_global = tile_start + n_local;
                long v_off = b_idx * stride_vz + hkv_idx * stride_vh + (long)n_global * stride_vn + (long)d * stride_vk;
                KV_tg[n_local * D_V + d] = V[v_off];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 4: Fused per-thread softmax + V accumulation
            if (active) {{
                for (int n_local = 0; n_local < tile_size; n_local++) {{
                    int n_idx = tile_start + n_local;
                    float s = S_tg[(int)tid * BLOCK_N + n_local] * scale_val;
                    float score_val = s;
                    int b_idx_sm = b_idx; int h_idx_sm = h_idx;
                    (void)b_idx_sm; (void)h_idx_sm;
{score_code}
                    s = score_val;
{mask_section}\
                    float new_max = metal::max(row_max, s);
                    float old_scale_v = metal::precise::exp(row_max - new_max);
                    float p = metal::precise::exp(s - new_max);
                    row_sum = row_sum * old_scale_v + p;
                    for (int d = 0; d < D_V; d++) o_acc[d] *= old_scale_v;
                    for (int d = 0; d < D_V; d++)
                        o_acc[d] += p * float(KV_tg[n_local * D_V + d]);
                    row_max = new_max;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}"""

    partial_loop = _block_loop_body(score_mod_indented, mask_mod_indented, True)
    full_loop = _block_loop_body(full_score_mod_indented, None, False)

    shader = f"""\
#include <metal_stdlib>
using namespace metal;

constant int D_QK = {d_qk};
constant int D_V = {d_v};
constant int BLOCK_M = {block_m};
constant int BLOCK_N = {block_n};
constant int KV_DIM = {kv_dim};

kernel void flex_attn_fwd(
    device {metal_dtype}* out [[buffer(0)]],
    constant {metal_dtype}* Q [[buffer(1)]],
    constant {metal_dtype}* K [[buffer(2)]],
    constant {metal_dtype}* V [[buffer(3)]],
    constant int* kv_num_blocks [[buffer(4)]],
    constant int* kv_indices [[buffer(5)]],
    constant float* scale_buf [[buffer(6)]],
{extra_buf_params}{full_kv_params}{scalar_params_str},
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
{unpack_code}

    int q_block = tgpos.x;
    int h_idx = tgpos.y;
    int b_idx = tgpos.z;
    int m_base = q_block * BLOCK_M;
    int m_idx = m_base + (int)tid;
    bool active = (m_idx < N_Q);

    float scale_val = scale_buf[0];
    int hkv_idx = h_idx / (int)gqa_shared_heads;

    // Threadgroup memory: Q persistent, KV shared (K then V), S for MMA scores
    threadgroup {metal_dtype} Q_tg[BLOCK_M * D_QK];
    threadgroup {metal_dtype} KV_tg[BLOCK_N * KV_DIM];
    threadgroup float S_tg[BLOCK_M * BLOCK_N];

    // Cooperative load Q (persistent across all KV tiles)
    {{
        long q_base_global = b_idx * stride_qz + h_idx * stride_qh;
        for (int i = (int)tid; i < BLOCK_M * D_QK; i += BLOCK_M) {{
            int row = i / D_QK;
            int d = i % D_QK;
            int global_row = m_base + row;
            Q_tg[i] = (global_row < N_Q)
                ? Q[q_base_global + (long)global_row * stride_qm + (long)d * stride_qk]
                : ({metal_dtype})0;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_max = -HUGE_VALF;
    float row_sum = 0.0f;
    float o_acc[D_V];
    for (int d = 0; d < D_V; d++) o_acc[d] = 0.0f;

    int sparse_q_idx = m_base / (int)SPARSE_Q_BLOCK_SIZE;
    long sparse_idx_z = b_idx % SPARSE_Z;
    long sparse_idx_hq = h_idx % SPARSE_HQ;

    // Partial blocks (score_mod + mask_mod)
    int num_kv_blks = kv_num_blocks[sparse_idx_z * kv_nb_stride_z + sparse_idx_hq * kv_nb_stride_h + sparse_q_idx * kv_nb_stride_q];
    long kv_idx_base = sparse_idx_z * kv_idx_stride_z + sparse_idx_hq * kv_idx_stride_h + (long)sparse_q_idx * kv_idx_stride_q;

    for (int blk = 0; blk < num_kv_blks; blk++) {{
        int kv_block_idx = kv_indices[kv_idx_base + (long)blk * kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

{partial_loop}
    }}
"""

    if has_full_blocks:
        shader += f"""
    // Full blocks (score_mod only)
    int full_num_kv_blks = full_kv_num_blocks[sparse_idx_z * full_kv_nb_stride_z + sparse_idx_hq * full_kv_nb_stride_h + sparse_q_idx * full_kv_nb_stride_q];
    long full_kv_idx_base = sparse_idx_z * full_kv_idx_stride_z + sparse_idx_hq * full_kv_idx_stride_h + (long)sparse_q_idx * full_kv_idx_stride_q;

    for (int blk = 0; blk < full_num_kv_blks; blk++) {{
        int kv_block_idx = full_kv_indices[full_kv_idx_base + (long)blk * full_kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

{full_loop}
    }}
"""

    shader += f"""
    if (active) {{
        if (row_sum == 0.0f) row_sum = 1.0f;
        long out_base = b_idx * stride_oz + h_idx * stride_oh + (long)m_idx * stride_om;
        for (int d = 0; d < D_V; d++)
            out[out_base + (long)d * stride_ok] = {metal_dtype}(o_acc[d] / row_sum);
    }}
}}
"""
    return shader


def _generate_metal_shader(
    dtype: torch.dtype,
    d_qk: int,
    d_v: int,
    score_mod_graph: torch.fx.GraphModule,
    mask_mod_graph: torch.fx.GraphModule,
    num_extra_bufs: int,
    has_full_blocks: bool,
    block_m: int = 32,
) -> str:
    """Generate the complete Metal shader source for flex attention.

    Args:
        dtype: The data type (float32/float16/bfloat16)
        d_qk: Query/Key head dimension (compile-time constant)
        d_v: Value head dimension (compile-time constant)
        score_mod_graph: FX GraphModule for score modification
        mask_mod_graph: FX GraphModule for mask modification
        num_extra_bufs: Number of captured buffers from score_mod/mask_mod
        has_full_blocks: Whether the block mask has full (unmasked) blocks
        block_m: Number of query rows per threadgroup
    """
    metal_dtype = METAL_DTYPE_MAP[dtype]

    bytes_per_elem = 4 if dtype == torch.float32 else 2
    kv_dim = max(d_qk, d_v)

    # Use simdgroup MMA for Q@K^T when D dimensions are multiples of 8
    use_mma = (d_qk % 8 == 0) and (d_v % 8 == 0) and (block_m % 8 == 0)

    if use_mma:
        # MMA needs Q_tg + KV_tg + S_tg in 32KB threadgroup memory
        q_tg_bytes = block_m * d_qk * bytes_per_elem
        budget = 32768 - q_tg_bytes
        per_n = kv_dim * bytes_per_elem + block_m * 4  # KV_tg + S_tg
        max_block_n = budget // per_n
        block_n = min(32, max_block_n)
        block_n = (block_n // 8) * 8  # must be multiple of 8 for MMA tiles
        if block_n < 8:
            use_mma = False

    if not use_mma:
        # fallback: K_tile + V_tile only
        block_n = min(32, 32768 // ((d_qk + d_v) * bytes_per_elem))
        block_n = max(1, block_n)

    # Build extra buffer parameter declarations
    extra_buf_params = ""
    buf_idx = 7  # Start after out, Q, K, V, kv_num_blocks, kv_indices, scale
    for i in range(num_extra_bufs):
        extra_buf_params += (
            f"    constant float* extra_buf_{i} [[buffer({buf_idx})]],\n"
        )
        buf_idx += 1

    if has_full_blocks:
        full_kv_params = (
            f"    constant int* full_kv_num_blocks [[buffer({buf_idx})]],\n"
        )
        buf_idx += 1
        full_kv_params += (
            f"    constant int* full_kv_indices [[buffer({buf_idx})]],\n"
        )
        buf_idx += 1
    else:
        full_kv_params = ""

    # Pack all scalar parameters into a single struct buffer to stay within
    # Metal's 31-buffer limit.
    scalar_names = [
        "B", "Hq", "Hkv", "N_Q", "N_KV",
        "stride_qz", "stride_qh", "stride_qm", "stride_qk",
        "stride_kz", "stride_kh", "stride_kn", "stride_kk",
        "stride_vz", "stride_vh", "stride_vn", "stride_vk",
        "stride_oz", "stride_oh", "stride_om", "stride_ok",
        "SPARSE_KV_BLOCK_SIZE",
        "gqa_shared_heads",
        "SPARSE_Z", "SPARSE_HQ",
        "kv_nb_stride_z", "kv_nb_stride_h", "kv_nb_stride_q",
        "kv_idx_stride_z", "kv_idx_stride_h", "kv_idx_stride_q", "kv_idx_stride_b",
        "SPARSE_Q_BLOCK_SIZE",
    ]
    if has_full_blocks:
        scalar_names += [
            "full_kv_nb_stride_z", "full_kv_nb_stride_h", "full_kv_nb_stride_q",
            "full_kv_idx_stride_z", "full_kv_idx_stride_h", "full_kv_idx_stride_q", "full_kv_idx_stride_b",
        ]
    scalar_params_str = f"    constant long* _params [[buffer({buf_idx})]]"

    def _indent_code(code, spaces):
        prefix = " " * spaces
        return "\n".join(prefix + line for line in code.split("\n") if line.strip())

    # Compile score_mod/mask_mod with unique var prefixes for each usage site
    partial_score_code = _compile_score_mod_to_metal(score_mod_graph, {}, "_ps")
    partial_mask_code = _compile_mask_mod_to_metal(mask_mod_graph, {}, "_pm")
    full_score_code = _compile_score_mod_to_metal(score_mod_graph, {}, "_fs")

    score_mod_indented = _indent_code(partial_score_code, 24)
    mask_mod_indented = _indent_code(partial_mask_code, 24)
    full_score_mod_indented = _indent_code(full_score_code, 24)

    # Generate scalar unpacking code
    unpack_lines = []
    for i, name in enumerate(scalar_names):
        unpack_lines.append(f"    long {name} = _params[{i}];")
    unpack_code = "\n".join(unpack_lines)

    # MMA path: simdgroup_matrix for Q@K^T
    if use_mma:
        return _generate_mma_shader(
            metal_dtype, d_qk, d_v, kv_dim, block_m, block_n,
            extra_buf_params, full_kv_params, scalar_params_str,
            unpack_code, score_mod_indented, mask_mod_indented,
            full_score_mod_indented, has_full_blocks,
        )

    # fallback: per-thread dot products with cooperative K/V tiling
    shader = f"""\
#include <metal_stdlib>
using namespace metal;

constant int D_QK = {d_qk};
constant int D_V = {d_v};
constant int BLOCK_M = {block_m};
constant int BLOCK_N = {block_n};

kernel void flex_attn_fwd(
    device {metal_dtype}* out [[buffer(0)]],
    constant {metal_dtype}* Q [[buffer(1)]],
    constant {metal_dtype}* K [[buffer(2)]],
    constant {metal_dtype}* V [[buffer(3)]],
    constant int* kv_num_blocks [[buffer(4)]],
    constant int* kv_indices [[buffer(5)]],
    constant float* scale_buf [[buffer(6)]],
{extra_buf_params}{full_kv_params}{scalar_params_str},
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    // Unpack scalar parameters from packed buffer
{unpack_code}

    // Grid: (num_q_blocks, Hq, B)
    int q_block = tgpos.x;
    int h_idx = tgpos.y;
    int b_idx = tgpos.z;
    int m_idx = q_block * BLOCK_M + (int)tid;

    // All threads must participate in threadgroup barriers, so use activity flag
    bool active = (m_idx < N_Q);

    float scale_val = scale_buf[0];
    int hkv_idx = h_idx / (int)gqa_shared_heads;

    // Load Q row into registers (active threads only)
    float q_row[D_QK];
    if (active) {{
        long q_base = b_idx * stride_qz + h_idx * stride_qh + (long)m_idx * stride_qm;
        for (int d = 0; d < D_QK; d++) {{
            q_row[d] = float(Q[q_base + (long)d * stride_qk]);
        }}
    }}

    // Online softmax accumulators
    float row_max = -HUGE_VALF;
    float row_sum = 0.0f;
    float o_acc[D_V];
    for (int d = 0; d < D_V; d++) o_acc[d] = 0.0f;

    // Threadgroup shared memory for K and V tiles — cooperative loading
    // reduces global memory reads by ~BLOCK_M x vs per-thread loading
    threadgroup {metal_dtype} K_tile[BLOCK_N * D_QK];
    threadgroup {metal_dtype} V_tile[BLOCK_N * D_V];

    // Block mask: compute sparse Q index from threadgroup (consistent across all threads)
    int m_base = q_block * BLOCK_M;
    int sparse_q_idx = m_base / (int)SPARSE_Q_BLOCK_SIZE;
    long sparse_idx_z = b_idx % SPARSE_Z;
    long sparse_idx_hq = h_idx % SPARSE_HQ;

    // Process partial blocks (need both score_mod and mask_mod)
    int num_kv_blks = kv_num_blocks[sparse_idx_z * kv_nb_stride_z + sparse_idx_hq * kv_nb_stride_h + sparse_q_idx * kv_nb_stride_q];
    long kv_idx_base = sparse_idx_z * kv_idx_stride_z + sparse_idx_hq * kv_idx_stride_h + (long)sparse_q_idx * kv_idx_stride_q;

    for (int blk = 0; blk < num_kv_blks; blk++) {{
        int kv_block_idx = kv_indices[kv_idx_base + (long)blk * kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

        // Sub-tile over KV positions with cooperative threadgroup loading
        for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_N) {{
            int tile_end = min(tile_start + BLOCK_N, kv_end);
            int tile_size = tile_end - tile_start;

            // Cooperative load: all threads share the work of loading K tile
            for (int i = (int)tid; i < tile_size * D_QK; i += BLOCK_M) {{
                int n_local = i / D_QK;
                int d = i % D_QK;
                int n_global = tile_start + n_local;
                long k_off = b_idx * stride_kz + hkv_idx * stride_kh + (long)n_global * stride_kn + (long)d * stride_kk;
                K_tile[n_local * D_QK + d] = K[k_off];
            }}

            // Cooperative load V tile
            for (int i = (int)tid; i < tile_size * D_V; i += BLOCK_M) {{
                int n_local = i / D_V;
                int d = i % D_V;
                int n_global = tile_start + n_local;
                long v_off = b_idx * stride_vz + hkv_idx * stride_vh + (long)n_global * stride_vn + (long)d * stride_vk;
                V_tile[n_local * D_V + d] = V[v_off];
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {{
                for (int n_local = 0; n_local < tile_size; n_local++) {{
                    int n_idx = tile_start + n_local;

                    // Dot product Q[m_idx] . K_tile[n_local] (from threadgroup memory)
                    float s = 0.0f;
                    for (int d = 0; d < D_QK; d++) {{
                        s += q_row[d] * float(K_tile[n_local * D_QK + d]);
                    }}
                    s *= scale_val;

                    // Apply score_mod
                    float score_val = s;
                    int b_idx_sm = b_idx;
                    int h_idx_sm = h_idx;
                    (void)b_idx_sm; (void)h_idx_sm;
{score_mod_indented}
                    s = score_val;

                    // Apply mask_mod
                    bool mask_result = true;
{mask_mod_indented}
                    if (!mask_result) s = -HUGE_VALF;

                    // Online softmax update
                    float new_max = metal::max(row_max, s);
                    float old_scale = metal::precise::exp(row_max - new_max);
                    float p = metal::precise::exp(s - new_max);
                    row_sum = row_sum * old_scale + p;
                    for (int d = 0; d < D_V; d++) o_acc[d] *= old_scale;

                    // Accumulate V (from threadgroup memory)
                    for (int d = 0; d < D_V; d++) {{
                        o_acc[d] += p * float(V_tile[n_local * D_V + d]);
                    }}
                    row_max = new_max;
                }}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
"""

    if has_full_blocks:
        shader += f"""
    // Process full blocks (only score_mod, no mask_mod needed)
    int full_num_kv_blks = full_kv_num_blocks[sparse_idx_z * full_kv_nb_stride_z + sparse_idx_hq * full_kv_nb_stride_h + sparse_q_idx * full_kv_nb_stride_q];
    long full_kv_idx_base = sparse_idx_z * full_kv_idx_stride_z + sparse_idx_hq * full_kv_idx_stride_h + (long)sparse_q_idx * full_kv_idx_stride_q;

    for (int blk = 0; blk < full_num_kv_blks; blk++) {{
        int kv_block_idx = full_kv_indices[full_kv_idx_base + (long)blk * full_kv_idx_stride_b];
        int kv_start = kv_block_idx * (int)SPARSE_KV_BLOCK_SIZE;
        int kv_end = min(kv_start + (int)SPARSE_KV_BLOCK_SIZE, (int)N_KV);

        for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_N) {{
            int tile_end = min(tile_start + BLOCK_N, kv_end);
            int tile_size = tile_end - tile_start;

            for (int i = (int)tid; i < tile_size * D_QK; i += BLOCK_M) {{
                int n_local = i / D_QK;
                int d = i % D_QK;
                int n_global = tile_start + n_local;
                long k_off = b_idx * stride_kz + hkv_idx * stride_kh + (long)n_global * stride_kn + (long)d * stride_kk;
                K_tile[n_local * D_QK + d] = K[k_off];
            }}

            for (int i = (int)tid; i < tile_size * D_V; i += BLOCK_M) {{
                int n_local = i / D_V;
                int d = i % D_V;
                int n_global = tile_start + n_local;
                long v_off = b_idx * stride_vz + hkv_idx * stride_vh + (long)n_global * stride_vn + (long)d * stride_vk;
                V_tile[n_local * D_V + d] = V[v_off];
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {{
                for (int n_local = 0; n_local < tile_size; n_local++) {{
                    int n_idx = tile_start + n_local;

                    float s = 0.0f;
                    for (int d = 0; d < D_QK; d++) {{
                        s += q_row[d] * float(K_tile[n_local * D_QK + d]);
                    }}
                    s *= scale_val;

                    float score_val = s;
                    int b_idx_sm = b_idx;
                    int h_idx_sm = h_idx;
                    (void)b_idx_sm; (void)h_idx_sm;
{full_score_mod_indented}
                    s = score_val;

                    float new_max = metal::max(row_max, s);
                    float old_scale = metal::precise::exp(row_max - new_max);
                    float p = metal::precise::exp(s - new_max);
                    row_sum = row_sum * old_scale + p;
                    for (int d = 0; d < D_V; d++) o_acc[d] *= old_scale;

                    for (int d = 0; d < D_V; d++) {{
                        o_acc[d] += p * float(V_tile[n_local * D_V + d]);
                    }}
                    row_max = new_max;
                }}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}
"""

    shader += f"""
    // Normalize and write output
    if (active) {{
        if (row_sum == 0.0f) row_sum = 1.0f;
        long out_base = b_idx * stride_oz + h_idx * stride_oh + (long)m_idx * stride_om;
        for (int d = 0; d < D_V; d++) {{
            out[out_base + (long)d * stride_ok] = {metal_dtype}(o_acc[d] / row_sum);
        }}
    }}
}}
"""
    return shader


class MetalFlexAttentionNode(ir.ExternKernelAlloc):
    """IR node for MPS flex attention that generates a Metal shader at codegen time.

    Codegen emits: shader compilation, output allocation, and kernel dispatch.
    The Metal kernel writes directly to the output buffer (passed as first arg).
    Scalar arguments (sizes, strides) are emitted as int64 values after tensor args.
    """

    def __init__(
        self,
        layout: ir.Layout,
        inputs: list[ir.IRNode],
        shader_source: str,
        scalar_args: list[Any],
        grid: tuple[Any, ...],
        block_m: int,
    ):
        super().__init__(
            layout=layout,
            inputs=inputs,
            python_kernel_name="metal_flex_attention",
        )
        self.shader_source = shader_source
        self.scalar_args = scalar_args
        self.grid = grid
        self.block_m = block_m

    def codegen(self, wrapper) -> None:
        wrapper.add_import_once(
            "from torch._inductor.runtime.runtime_utils import compile_mps_shader"
        )
        wrapper.add_import_once("import torch")

        # Compile shader - use wrapper's src_to_kernel cache for dedup
        if not hasattr(wrapper, '_mps_flex_cache'):
            wrapper._mps_flex_cache = {}

        src_hash = hash(self.shader_source)
        if src_hash not in wrapper._mps_flex_cache:
            lib_name = f"_mps_flex_lib_{wrapper.next_kernel_suffix()}"
            wrapper._mps_flex_cache[src_hash] = lib_name
            wrapper.header.splice(
                f"{lib_name} = compile_mps_shader('''\n{self.shader_source}\n''')"
            )
        lib_name = wrapper._mps_flex_cache[src_hash]

        # Allocate output tensor
        name = self.get_name()
        sizes = ", ".join(wrapper.codegen_sizevar(s) for s in self.layout.size)
        strides = ", ".join(wrapper.codegen_sizevar(s) for s in self.layout.stride)
        wrapper.writeline(
            f"{name} = torch.empty_strided("
            f"({sizes},), ({strides},), "
            f"dtype={self.layout.dtype!r}, device='mps')"
        )

        # Pack scalar args into a single int64 tensor buffer
        num_scalars = len(self.scalar_args)
        scalar_strs = [wrapper.codegen_sizevar(s) for s in self.scalar_args]
        params_name = f"_flex_params_{name}"
        wrapper.writeline(
            f"{params_name} = torch.tensor("
            f"[{', '.join(scalar_strs)}], dtype=torch.int64, device='mps')"
        )

        # Build argument list: output, tensor inputs, then packed scalar buffer
        arg_parts = [name]
        for inp in self.inputs:
            arg_parts.append(inp.codegen_reference())
        arg_parts.append(params_name)
        args_str = ", ".join(arg_parts)

        # Grid: (ceil(N_Q/BLOCK_M), Hq, B)
        grid_x = wrapper.codegen_sizevar(self.grid[0])
        grid_y = wrapper.codegen_sizevar(self.grid[1])
        grid_z = wrapper.codegen_sizevar(self.grid[2])

        wrapper.writeline(
            f"{lib_name}.flex_attn_fwd("
            f"{args_str}, "
            f"threads=[{grid_x} * {self.block_m}, {grid_y}, {grid_z}], "
            f"group_size=[{self.block_m}, 1, 1])"
        )
