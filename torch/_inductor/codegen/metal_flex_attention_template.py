# mypy: allow-untyped-defs
"""Metal shader template for flex attention on MPS"""

import itertools
from collections import namedtuple
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils._ordered_set import OrderedSet

from .. import ir


METAL_DTYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
}

# Captured score_mod/mask_mod buffers may hold more types than q/k/v; elements
# are always loaded and cast to float for use in the (float-typed) op table.
# Integer captures (e.g. document ids) are therefore exact only below 2**24,
# which covers realistic id/index ranges.
CAPTURE_DTYPE_MAP = {
    torch.float32: "float",
    torch.float16: "half",
    torch.bfloat16: "bfloat",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "char",
    torch.uint8: "uchar",
    torch.bool: "bool",
}

# A captured buffer bound as a Metal kernel argument, with its static shape.
_CapturedBuf = namedtuple("_CapturedBuf", ["name", "sizes", "strides", "dtype"])


# FX aten target -> (output C type, Metal format string with positional {} placeholders).
aten = torch.ops.aten
prims = torch.ops.prims
_OP_TABLE: dict[Any, tuple[str, str]] = {
    # Arithmetic
    aten.add.Tensor: ("float", "{} + {}"),
    aten.add.Scalar: ("float", "{} + {}"),
    aten.sub.Tensor: ("float", "{} - {}"),
    aten.sub.Scalar: ("float", "{} - {}"),
    aten.mul.Tensor: ("float", "{} * {}"),
    aten.mul.Scalar: ("float", "{} * {}"),
    # Cast to float so Python truediv semantics hold even when both operands are ints
    aten.div.Tensor: ("float", "static_cast<float>({}) / static_cast<float>({})"),
    aten.div.Scalar: ("float", "static_cast<float>({}) / static_cast<float>({})"),
    aten.neg.default: ("float", "-{}"),
    aten.abs.default: ("float", "metal::abs({})"),
    # Comparisons
    aten.gt.Tensor: ("bool", "{} > {}"),
    aten.gt.Scalar: ("bool", "{} > {}"),
    aten.ge.Tensor: ("bool", "{} >= {}"),
    aten.ge.Scalar: ("bool", "{} >= {}"),
    aten.lt.Tensor: ("bool", "{} < {}"),
    aten.lt.Scalar: ("bool", "{} < {}"),
    aten.le.Tensor: ("bool", "{} <= {}"),
    aten.le.Scalar: ("bool", "{} <= {}"),
    aten.eq.Tensor: ("bool", "{} == {}"),
    aten.eq.Scalar: ("bool", "{} == {}"),
    aten.ne.Tensor: ("bool", "{} != {}"),
    aten.ne.Scalar: ("bool", "{} != {}"),
    aten.logical_and.default: ("bool", "{} && {}"),
    aten.logical_or.default: ("bool", "{} || {}"),
    aten.logical_not.default: ("bool", "!{}"),
    # Math
    aten.exp.default: ("float", "metal::precise::exp({})"),
    aten.log.default: ("float", "metal::precise::log({})"),
    aten.sqrt.default: ("float", "metal::precise::sqrt({})"),
    aten.rsqrt.default: ("float", "metal::precise::rsqrt({})"),
    aten.tanh.default: ("float", "metal::precise::tanh({})"),
    aten.sin.default: ("float", "metal::precise::sin({})"),
    aten.cos.default: ("float", "metal::precise::cos({})"),
    aten.exp2.default: ("float", "metal::precise::exp2({})"),
    # remainder
    aten.remainder.Scalar: ("float", "c10::metal::remainder({}, (float){})"),
    aten.remainder.Tensor: ("float", "c10::metal::remainder({}, {})"),
    aten.maximum.default: ("float", "metal::max({}, {})"),
    aten.max.other: ("float", "metal::max({}, {})"),
    aten.minimum.default: ("float", "metal::min({}, {})"),
    aten.min.other: ("float", "metal::min({}, {})"),
    # Bitwise - int output
    aten.bitwise_and.Tensor: ("int", "(int){} & (int){}"),
    aten.bitwise_and.Scalar: ("int", "(int){} & (int){}"),
    aten.bitwise_or.Tensor: ("int", "(int){} | (int){}"),
    aten.bitwise_or.Scalar: ("int", "(int){} | (int){}"),
    aten.where.self: ("float", "{} ? {} : {}"),
    aten._to_copy.default: ("float", "static_cast<float>({})"),
    aten.to.dtype: ("float", "static_cast<float>({})"),
    prims.convert_element_type.default: ("float", "static_cast<float>({})"),
    aten.scalar_tensor.default: ("float", "{}"),
}
del aten, prims


def _fx_graph_to_metal(
    graph_module: torch.fx.GraphModule,
    fixed_inputs: dict[str, str],
    captured_views: dict[str, _CapturedBuf],
    output_var: str,
    var_prefix: str,
) -> str:
    """Compile an FX GraphModule to inline Metal code.

    `fixed_inputs` maps scalar placeholder names (score/b/h/q/kv) to Metal
    variables. `captured_views` maps captured-buffer placeholder names to a
    `_CapturedBuf`; these are indexed by `aten.index.Tensor` down to a scalar
    element load. The output is assigned to `output_var`; `var_prefix`
    namespaces fresh temps.
    """
    var_map: dict[str, str] = {}
    views: dict[str, tuple] = {}
    code_lines: list[str] = []
    tmp_counter = itertools.count()

    # Only emit nodes the output depends on; this drops dead constant captures
    output_node = next(n for n in graph_module.graph.nodes if n.op == "output")
    needed: OrderedSet[str] = OrderedSet()
    stack = list(output_node.all_input_nodes)
    while stack:
        nd = stack.pop()
        if nd.name not in needed:
            needed.add(nd.name)
            stack.extend(nd.all_input_nodes)

    def _tmp() -> str:
        return f"{var_prefix}{next(tmp_counter)}"

    def _val(node) -> str:
        if isinstance(node, torch.fx.Node):
            if node.name in views:
                raise NotImplementedError(
                    "flex_attention on MPS: captured buffer used without being "
                    "fully indexed to a scalar"
                )
            return var_map[node.name]
        if isinstance(node, bool):
            return "true" if node else "false"
        if isinstance(node, float):
            if node == float("inf"):
                return "INFINITY"
            if node == float("-inf"):
                return "(-INFINITY)"
            if node != node:  # NaN
                return "NAN"
            return f"{node}f"
        if isinstance(node, int):
            return str(node)
        return str(node)

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if node.name in fixed_inputs:
                var_map[node.name] = fixed_inputs[node.name]
            elif node.name in captured_views:
                cap = captured_views[node.name]
                if len(cap.sizes) == 0:
                    var_map[node.name] = f"(float){cap.name}[0]"
                else:
                    views[node.name] = (
                        cap.name,
                        "0",
                        list(cap.sizes),
                        list(cap.strides),
                    )
            else:
                var_map[node.name] = f"/* unknown placeholder {node.name} */"
            continue

        if node.op == "output":
            out_node = node.args[0]
            if isinstance(out_node, (tuple, list)):
                out_node = out_node[0]
            code_lines.append(f"{output_var} = {_val(out_node)};")
            continue

        if node.name not in needed:
            continue

        if node.op == "get_attr":
            raise NotImplementedError(
                "flex_attention on MPS does not support constant tensor captures "
                "in score_mod/mask_mod yet"
            )

        if node.op != "call_function":
            continue

        target = node.target
        args = node.args

        # Index a captured buffer: accumulate a flat offset over the leading
        # dims, peeling them off; emit a scalar load when fully indexed.
        if target == torch.ops.aten.index.Tensor:
            base_node, idx_list = args[0], args[1]
            if not (isinstance(base_node, torch.fx.Node) and base_node.name in views):
                raise NotImplementedError(
                    "flex_attention on MPS only supports indexing captured buffers"
                )
            base, offset, sizes, strides = views[base_node.name]
            if any(ix is None for ix in idx_list) or len(idx_list) > len(sizes):
                raise NotImplementedError(
                    "flex_attention on MPS does not support this captured-buffer "
                    "indexing pattern"
                )
            for j, ix in enumerate(idx_list):
                term = f"(long)({_val(ix)}) * {strides[j]}"
                offset = term if offset == "0" else f"{offset} + {term}"
            k = len(idx_list)
            rem_sizes, rem_strides = sizes[k:], strides[k:]
            if rem_sizes:
                views[node.name] = (base, offset, rem_sizes, rem_strides)
            else:
                t = _tmp()
                code_lines.append(f"float {t} = (float){base}[{offset}];")
                var_map[node.name] = t
            continue

        t = _tmp()
        if target in _OP_TABLE:
            out_type, fmt = _OP_TABLE[target]
            code_lines.append(
                f"{out_type} {t} = {fmt.format(*(_val(a) for a in args))};"
            )
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
        elif target in (
            torch.ops.aten.div.Tensor_mode,
            torch.ops.aten.div.Scalar_mode,
        ):
            # Floor/trunc division (e.g. paged attention's kv_idx // page_size).
            mode = args[2] if len(args) > 2 else node.kwargs.get("rounding_mode")
            quot = f"static_cast<float>({_val(args[0])}) / static_cast<float>({_val(args[1])})"
            if mode == "floor":
                code_lines.append(f"float {t} = metal::floor({quot});")
            elif mode == "trunc":
                code_lines.append(f"float {t} = metal::trunc({quot});")
            else:
                code_lines.append(f"float {t} = {quot};")
        elif target in (
            torch.ops.aten.full_like.default,
            torch.ops.aten.full.default,
        ):
            fill = args[1] if len(args) > 1 else node.kwargs.get("fill_value", 0)
            code_lines.append(f"float {t} = {_val(fill)};")
        else:
            raise NotImplementedError(
                f"flex_attention on MPS does not support op {target} in "
                f"score_mod/mask_mod yet"
            )

        var_map[node.name] = t

    return "\n".join(code_lines)


def _compile_subgraph_to_metal(
    graph_module: torch.fx.GraphModule,
    placeholder_metal_names: list[str],
    captured_meta: list[_CapturedBuf],
    output_var: str,
    var_prefix: str,
) -> str:
    """
    Bind placeholders by position to Metal variable names and emit parts of Metal .

    score_mod placeholders are [score, b, h, q_idx, kv_idx, *captured];
    mask_mod's are [b, h, q_idx, kv_idx, *captured]. The captured placeholders
    map positionally to `captured_meta`.
    """
    fixed: dict[str, str] = {}
    captured_views: dict[str, _CapturedBuf] = {}
    n_fixed = len(placeholder_metal_names)
    for i, node in enumerate(
        n for n in graph_module.graph.nodes if n.op == "placeholder"
    ):
        if i < n_fixed:
            fixed[node.name] = placeholder_metal_names[i]
        else:
            captured_views[node.name] = captured_meta[i - n_fixed]
    return _fx_graph_to_metal(
        graph_module, fixed, captured_views, output_var, var_prefix
    )


def _generate_mma_shader(
    metal_dtype,
    d_qk,
    d_v,
    kv_dim,
    block_m,
    block_n,
    full_kv_params,
    captured_params,
    scalar_params_str,
    unpack_code,
    score_code,
    mask_code,
    has_full_blocks,
    scale,
    lse_param="",
    lse_write_code="",
):
    """Generate Metal shader using simdgroup_matrix for Q@K^T."""

    def _block_loop_body(is_partial):
        if is_partial:
            mask_section = f"""
                    bool mask_result = true;
{mask_code}
"""
        else:
            mask_section = """
                    bool mask_result = true;
"""
        return f"""\
        for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_N) {{
            int tile_end = min(tile_start + BLOCK_N, kv_end);
            int tile_size = tile_end - tile_start;

            // Load K into KV_tg, zero-padding rows past tile_size for MMA alignment.
            for (int i = (int)tid; i < BLOCK_N * D_QK; i += BLOCK_M) {{
                int n_local = i / D_QK;
                int d = i % D_QK;
                if (n_local < tile_size) {{
                    int n_global = tile_start + n_local;
                    long k_off = b_kv * stride_kz + hkv_idx * stride_kh + (long)n_global * stride_kn + (long)d * stride_kk;
                    KV_tg[n_local * D_QK + d] = K[k_off];
                }} else {{
                    KV_tg[n_local * D_QK + d] = ({metal_dtype})0;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // S = Q_tg @ KV_tg^T via simdgroup MMA.
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

            // Load V into KV_tg (reuses the K slot; safe past the MMA barrier).
            for (int i = (int)tid; i < tile_size * D_V; i += BLOCK_M) {{
                int n_local = i / D_V;
                int d = i % D_V;
                int n_global = tile_start + n_local;
                long v_off = b_kv * stride_vz + hkv_idx * stride_vh + (long)n_global * stride_vn + (long)d * stride_vk;
                KV_tg[n_local * D_V + d] = V[v_off];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Fused per-thread softmax + V accumulation.
            if (active) {{
                for (int n_local = 0; n_local < tile_size; n_local++) {{
                    int n_idx = tile_start + n_local;
                    float s = S_tg[(int)tid * BLOCK_N + n_local] * SCALE_VAL;
                    float score_val = s;
{score_code}
                    s = score_val;
{mask_section}\
                    // s > -inf guards exp(-inf - -inf) = NaN when an entire row is -inf so far.
                    if (mask_result && s > -INFINITY) {{
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
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}"""

    partial_loop = _block_loop_body(is_partial=True)
    full_loop = _block_loop_body(is_partial=False)

    shader = f"""\
#include <metal_stdlib>
#include <c10/metal/utils.h>
using namespace metal;

constant int D_QK = {d_qk};
constant int D_V = {d_v};
constant int BLOCK_M = {block_m};
constant int BLOCK_N = {block_n};
constant int KV_DIM = {kv_dim};
constant float SCALE_VAL = {scale!r}f;

kernel void flex_attn_fwd(
    device {metal_dtype}* out [[buffer(0)]],
    constant {metal_dtype}* Q [[buffer(1)]],
    constant {metal_dtype}* K [[buffer(2)]],
    constant {metal_dtype}* V [[buffer(3)]],
    constant int* kv_num_blocks [[buffer(4)]],
    constant int* kv_indices [[buffer(5)]],
{full_kv_params}{captured_params}{lse_param}{scalar_params_str},
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
{unpack_code}

    int q_block = tgpos.x;
    int h_idx = tgpos.y;
    int b_idx = tgpos.z;
    int b_kv = b_idx % (int)Bkv;
    int m_base = q_block * BLOCK_M;
    int m_idx = m_base + (int)tid;
    bool active = (m_idx < N_Q);

    int hkv_idx = h_idx / (int)gqa_shared_heads;

    threadgroup {metal_dtype} Q_tg[BLOCK_M * D_QK];
    threadgroup {metal_dtype} KV_tg[BLOCK_N * KV_DIM];
    threadgroup float S_tg[BLOCK_M * BLOCK_N];

    // Q stays resident across all KV tiles.
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

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[D_V];
    for (int d = 0; d < D_V; d++) o_acc[d] = 0.0f;

    int sparse_q_idx = m_base / (int)SPARSE_Q_BLOCK_SIZE;
    long sparse_idx_z = b_idx % SPARSE_Z;
    long sparse_idx_hq = h_idx % SPARSE_HQ;

    // Partial blocks: score_mod + mask_mod.
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
    // Full blocks: score_mod only.
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
{lse_write_code}        if (row_sum == 0.0f) row_sum = 1.0f;
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
    has_full_blocks: bool,
    block_m: int,
    scale: float,
    score_captured: Sequence[tuple] = (),
    mask_captured: Sequence[tuple] = (),
    write_lse: bool = False,
) -> str:
    """Generate the complete Metal shader source for flex attention.

    `score_captured` / `mask_captured` are the buffers captured by score_mod /
    mask_mod, as (sizes, strides, dtype) tuples in placeholder order.
    """
    metal_dtype = METAL_DTYPE_MAP[dtype]

    bytes_per_elem = 4 if dtype == torch.float32 else 2
    kv_dim = max(d_qk, d_v)

    # simdgroup MMA needs each tile dim a multiple of 8.
    use_mma = (d_qk % 8 == 0) and (d_v % 8 == 0) and (block_m % 8 == 0)
    block_n = 0

    if use_mma:
        # Q_tg + KV_tg + S_tg must fit in the 32KB threadgroup memory budget.
        q_tg_bytes = block_m * d_qk * bytes_per_elem
        budget = 32768 - q_tg_bytes
        per_n = kv_dim * bytes_per_elem + block_m * 4
        max_block_n = budget // per_n
        block_n = min(32, max_block_n)
        block_n = (block_n // 8) * 8
        if block_n < 8:
            use_mma = False

    if not use_mma:
        block_n = min(32, 32768 // ((d_qk + d_v) * bytes_per_elem))
        block_n = max(1, block_n)

    buf_idx = 6  # 0=out, 1=Q, 2=K, 3=V, 4=kv_num_blocks, 5=kv_indices
    if has_full_blocks:
        full_kv_params = (
            f"    constant int* full_kv_num_blocks [[buffer({buf_idx})]],\n"
        )
        buf_idx += 1
        full_kv_params += f"    constant int* full_kv_indices [[buffer({buf_idx})]],\n"
        buf_idx += 1
    else:
        full_kv_params = ""

    # Captured tensors follow the (full) kv buffers as extra Metal arguments, in
    # the order built in lower_mps: score captures first, then mask captures.
    captured_decls: list[str] = []
    score_meta: list[_CapturedBuf] = []
    mask_meta: list[_CapturedBuf] = []
    for meta_list, caps in ((score_meta, score_captured), (mask_meta, mask_captured)):
        for sizes, strides, cap_dtype in caps:
            mtype = CAPTURE_DTYPE_MAP.get(cap_dtype)
            if mtype is None:
                raise NotImplementedError(
                    f"flex_attention on MPS does not support captured buffer "
                    f"dtype {cap_dtype}"
                )
            name = f"capbuf{len(captured_decls)}"
            captured_decls.append(
                f"    constant {mtype}* {name} [[buffer({buf_idx})]],"
            )
            meta_list.append(_CapturedBuf(name, list(sizes), list(strides), cap_dtype))
            buf_idx += 1
    captured_params = "".join(d + "\n" for d in captured_decls)

    # Pack scalars into one buffer to stay under Metal's 31-buffer limit.
    scalar_names = [
        "B",
        "Bkv",
        "Hq",
        "Hkv",
        "N_Q",
        "N_KV",
        "stride_qz",
        "stride_qh",
        "stride_qm",
        "stride_qk",
        "stride_kz",
        "stride_kh",
        "stride_kn",
        "stride_kk",
        "stride_vz",
        "stride_vh",
        "stride_vn",
        "stride_vk",
        "stride_oz",
        "stride_oh",
        "stride_om",
        "stride_ok",
        "SPARSE_KV_BLOCK_SIZE",
        "gqa_shared_heads",
        "SPARSE_Z",
        "SPARSE_HQ",
        "kv_nb_stride_z",
        "kv_nb_stride_h",
        "kv_nb_stride_q",
        "kv_idx_stride_z",
        "kv_idx_stride_h",
        "kv_idx_stride_q",
        "kv_idx_stride_b",
        "SPARSE_Q_BLOCK_SIZE",
    ]
    if has_full_blocks:
        scalar_names += [
            "full_kv_nb_stride_z",
            "full_kv_nb_stride_h",
            "full_kv_nb_stride_q",
            "full_kv_idx_stride_z",
            "full_kv_idx_stride_h",
            "full_kv_idx_stride_q",
            "full_kv_idx_stride_b",
        ]
    lse_param = ""
    lse_write_code = ""
    if write_lse:
        lse_param = f"    device float* lse [[buffer({buf_idx})]],\n"
        buf_idx += 1
        lse_write_code = (
            "        lse[(long)b_idx * (Hq * N_Q)"
            " + (long)h_idx * N_Q + (long)m_idx]"
            " = (row_sum > 0.0f)"
            " ? (row_max + metal::precise::log(row_sum)) * 1.44269504088896340736f"
            " : -INFINITY;\n"
        )

    scalar_params_str = f"    constant long* _params [[buffer({buf_idx})]]"

    score_code = _compile_subgraph_to_metal(
        graph_module=score_mod_graph,
        placeholder_metal_names=["score_val", "b_idx", "h_idx", "m_idx", "n_idx"],
        captured_meta=score_meta,
        output_var="score_val",
        var_prefix="_sm",
    )
    mask_code = _compile_subgraph_to_metal(
        graph_module=mask_mod_graph,
        placeholder_metal_names=["b_idx", "h_idx", "m_idx", "n_idx"],
        captured_meta=mask_meta,
        output_var="mask_result",
        var_prefix="_mm",
    )

    unpack_code = "\n".join(
        f"    long {name} = _params[{i}];" for i, name in enumerate(scalar_names)
    )

    if use_mma:
        return _generate_mma_shader(
            metal_dtype,
            d_qk,
            d_v,
            kv_dim,
            block_m,
            block_n,
            full_kv_params,
            captured_params,
            scalar_params_str,
            unpack_code,
            score_code,
            mask_code,
            has_full_blocks,
            scale,
            lse_param=lse_param,
            lse_write_code=lse_write_code,
        )

    # Fallback: per-thread dot product with cooperative K/V tiling.
    def _fallback_block_loop_body(is_partial):
        if is_partial:
            mask_section = f"""
                bool mask_result = true;
                {mask_code}
            """
        else:
            mask_section = """
                bool mask_result = true;
            """
        return f"""\
        for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_N) {{
            int tile_end = min(tile_start + BLOCK_N, kv_end);
            int tile_size = tile_end - tile_start;

            for (int i = (int)tid; i < tile_size * D_QK; i += BLOCK_M) {{
                int n_local = i / D_QK;
                int d = i % D_QK;
                int n_global = tile_start + n_local;
                long k_off = b_kv * stride_kz + hkv_idx * stride_kh + (long)n_global * stride_kn + (long)d * stride_kk;
                K_tile[n_local * D_QK + d] = K[k_off];
            }}

            for (int i = (int)tid; i < tile_size * D_V; i += BLOCK_M) {{
                int n_local = i / D_V;
                int d = i % D_V;
                int n_global = tile_start + n_local;
                long v_off = b_kv * stride_vz + hkv_idx * stride_vh + (long)n_global * stride_vn + (long)d * stride_vk;
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
                    s *= SCALE_VAL;

                    float score_val = s;
{score_code}
                    s = score_val;
{mask_section}\
                    // s > -inf guards exp(-inf - -inf) = NaN when an entire row is -inf so far.
                    if (mask_result && s > -INFINITY) {{
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
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}"""

    partial_loop = _fallback_block_loop_body(is_partial=True)
    full_loop = _fallback_block_loop_body(is_partial=False)

    shader = f"""\
#include <metal_stdlib>
#include <c10/metal/utils.h>
using namespace metal;

constant int D_QK = {d_qk};
constant int D_V = {d_v};
constant int BLOCK_M = {block_m};
constant int BLOCK_N = {block_n};
constant float SCALE_VAL = {scale!r}f;

kernel void flex_attn_fwd(
    device {metal_dtype}* out [[buffer(0)]],
    constant {metal_dtype}* Q [[buffer(1)]],
    constant {metal_dtype}* K [[buffer(2)]],
    constant {metal_dtype}* V [[buffer(3)]],
    constant int* kv_num_blocks [[buffer(4)]],
    constant int* kv_indices [[buffer(5)]],
{full_kv_params}{captured_params}{lse_param}{scalar_params_str},
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
{unpack_code}

    int q_block = tgpos.x;
    int h_idx = tgpos.y;
    int b_idx = tgpos.z;
    int b_kv = b_idx % (int)Bkv;
    int m_idx = q_block * BLOCK_M + (int)tid;
    bool active = (m_idx < N_Q);

    int hkv_idx = h_idx / (int)gqa_shared_heads;

    float q_row[D_QK];
    if (active) {{
        long q_base = b_idx * stride_qz + h_idx * stride_qh + (long)m_idx * stride_qm;
        for (int d = 0; d < D_QK; d++) {{
            q_row[d] = float(Q[q_base + (long)d * stride_qk]);
        }}
    }}

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[D_V];
    for (int d = 0; d < D_V; d++) o_acc[d] = 0.0f;

    threadgroup {metal_dtype} K_tile[BLOCK_N * D_QK];
    threadgroup {metal_dtype} V_tile[BLOCK_N * D_V];

    int m_base = q_block * BLOCK_M;
    int sparse_q_idx = m_base / (int)SPARSE_Q_BLOCK_SIZE;
    long sparse_idx_z = b_idx % SPARSE_Z;
    long sparse_idx_hq = h_idx % SPARSE_HQ;

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
{lse_write_code}        if (row_sum == 0.0f) row_sum = 1.0f;
        long out_base = b_idx * stride_oz + h_idx * stride_oh + (long)m_idx * stride_om;
        for (int d = 0; d < D_V; d++) {{
            out[out_base + (long)d * stride_ok] = {metal_dtype}(o_acc[d] / row_sum);
        }}
    }}
}}
"""
    return shader


class MetalFlexAttentionNode(ir.ExternKernelAlloc):
    """IR node for MPS flex attention; emits a compiled Metal shader at codegen."""

    def __init__(
        self,
        layout: ir.Layout,
        inputs: list[ir.IRNode],
        shader_source: str,
        scalar_args: list[Any],
        grid: tuple[Any, ...],
        block_m: int,
        mutates_lse: bool = False,
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
        if mutates_lse:
            lse_buf = self.inputs[-1]
            self.mutation_outputs = [
                # pyrefly: ignore [bad-argument-type]
                ir.MutationOutput(ir.NoneLayout(device=layout.device), lse_buf, self)
            ]

    def codegen(self, wrapper) -> None:
        wrapper.add_import_once(
            "from torch._inductor.runtime.runtime_utils import compile_mps_shader"
        )
        wrapper.add_import_once("import torch")

        src_code = self.shader_source
        if src_code in wrapper.src_to_kernel:
            lib_name = wrapper.src_to_kernel[src_code]
        else:
            lib_name = f"_mps_flex_lib_{wrapper.next_kernel_suffix()}"
            wrapper.src_to_kernel[src_code] = lib_name
            wrapper.header.splice(
                f"{lib_name} = compile_mps_shader('''\n{src_code}\n''')"
            )

        name = self.get_name()
        layout: ir.Layout = self.layout  # type: ignore[assignment]
        sizes = ", ".join(wrapper.codegen_sizevar(s) for s in layout.size)
        strides = ", ".join(wrapper.codegen_sizevar(s) for s in layout.stride)
        device_str = f"{layout.device.type!r}"
        wrapper.writeline(
            f"{name} = torch.empty_strided("
            f"({sizes},), ({strides},), "
            f"dtype={layout.dtype!r}, device={device_str})"
        )

        # One packed int64 buffer keeps us under Metal's 31-buffer limit.
        scalar_strs = [wrapper.codegen_sizevar(s) for s in self.scalar_args]
        params_name = f"_flex_params_{name}"
        wrapper.writeline(
            f"{params_name} = torch.tensor("
            f"[{', '.join(scalar_strs)}], dtype=torch.int64, device={device_str})"
        )

        arg_parts = [name]
        for inp in self.inputs:
            # pyrefly: ignore [missing-attribute]
            arg_parts.append(inp.codegen_reference())
        arg_parts.append(params_name)
        args_str = ", ".join(arg_parts)

        grid_x = wrapper.codegen_sizevar(self.grid[0])
        grid_y = wrapper.codegen_sizevar(self.grid[1])
        grid_z = wrapper.codegen_sizevar(self.grid[2])

        wrapper.writeline(
            f"{lib_name}.flex_attn_fwd("
            f"{args_str}, "
            f"threads=[{grid_x} * {self.block_m}, {grid_y}, {grid_z}], "
            f"group_size=[{self.block_m}, 1, 1])"
        )
