// Largely influeneced by
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
#include <ATen/native/mps/kernels/DecodeAttention.h>
#include <ATen/native/mps/kernels/PrefillAttention.h>
#include <ATen/native/mps/kernels/Attention.h>

// ── forward ──────────────────────────────────────────────────────────────────

template<typename T, int D>
[[kernel]] void flash_attn_fwd(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    device       T*       O   [[buffer(3)]],
    device       float*   LSE [[buffer(4)]],
    const constant uint&  qL  [[buffer(5)]],
    const constant uint&  kL  [[buffer(6)]],
    const constant uint&  gqa [[buffer(7)]],
    const constant uint&  nh  [[buffer(8)]],
    const constant float& sc  [[buffer(9)]],
    const constant bool&  ic  [[buffer(10)]],
    const constant uint4& qs  [[buffer(11)]],
    const constant uint4& ks  [[buffer(12)]],
    const constant uint4& vs  [[buffer(13)]],
    const constant uint4& os  [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D == 64) ? 64 : 32;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;

    const uint b    = bh / nh;
    const uint h    = bh % nh;
    const uint kv_h = h / gqa;

    const device T* Q_ptr = Q + b * qs[0] + h    * qs[1];
    const device T* K_ptr = K + b * ks[0] + kv_h * ks[1];
    const device T* V_ptr = V + b * vs[0] + kv_h * vs[1];
    device       T* O_ptr = O + b * os[0] + h    * os[1];

    const bool valid_q = (q_row < qL);

    float q_reg[EPL];
    for (int e = 0; e < EPL; e++)
        q_reg[e] = valid_q ? float(Q_ptr[q_row * qs[2] + lane * EPL + e]) : 0.0f;

    float acc[EPL] = {};
    float m = -INFINITY, l = 0.0f;

    const uint tg_size = 32 * BQ;  // 1024

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * ks[2] + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * vs[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool cv = !ic || (k_row <= q_row);
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < EPL; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];
            float score = cv ? (simd_sum(partial) * sc) : -INFINITY;

            float m_new = max(m, score);
            float alpha = metal::precise::exp(m - m_new);
            float p_j   = metal::precise::exp(score - m_new);
            m = m_new;
            l = l * alpha + p_j;

            for (int e = 0; e < EPL; e++)
                acc[e] = acc[e] * alpha + p_j * V_smem[j * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int e = 0; e < EPL; e++)
        O_ptr[q_row * os[2] + lane * EPL + e] = T(acc[e] * inv_l);
    if (lane == 0)
        LSE[bh * qL + q_row] = m + log(l);
}

// ── backward preprocess ───────────────────────────────────────────────────────
// D_vec[i] = rowsum(dO_i * O_i).  Pure register reduction via simd_sum().
// Grid  : (B*H, ceil(qL/BQ), 1),  TG : (32, BQ, 1)

template<typename T, int D>
[[kernel]] void flash_attn_bwd_preprocess(
    const device T*       dO  [[buffer(0)]],
    const device T*       O   [[buffer(1)]],
    device       float*   Dv  [[buffer(2)]],
    const constant uint&  qL  [[buffer(3)]],
    const constant uint&  nh  [[buffer(4)]],
    const constant uint4& dos [[buffer(5)]],
    const constant uint4& os  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;

    if (q_row >= qL) return;

    const uint b = bh / nh;
    const uint h = bh % nh;

    const device T* dO_ptr = dO + b * dos[0] + h * dos[1];
    const device T* O_ptr  = O  + b * os[0]  + h * os[1];

    float partial = 0.0f;
    for (int e = 0; e < EPL; e++)
        partial += float(dO_ptr[q_row * dos[2] + lane * EPL + e])
                 * float(O_ptr[q_row *  os[2]  + lane * EPL + e]);

    float total = simd_sum(partial);
    if (lane == 0)
        Dv[bh * qL + q_row] = total;
}

// ── backward dQ ──────────────────────────────────────────────────────────────
// Recomputes attention weights from saved LSE; accumulates dQ.
// Same smem strategy as forward: K+V both in threadgroup memory.
// Grid  : (B*H, ceil(qL/BQ), 1),  TG : (32, BQ, 1)

template<typename T, int D>
[[kernel]] void flash_attn_bwd_dq(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    const device T*       O   [[buffer(3)]],
    const device T*       dO  [[buffer(4)]],
    const device float*   LSE [[buffer(5)]],
    const device float*   Dv  [[buffer(6)]],
    device       T*       dQ  [[buffer(7)]],
    const constant uint&  qL  [[buffer(8)]],
    const constant uint&  kL  [[buffer(9)]],
    const constant uint&  gqa [[buffer(10)]],
    const constant uint&  nh  [[buffer(11)]],
    const constant float& sc  [[buffer(12)]],
    const constant bool&  ic  [[buffer(13)]],
    const constant uint4& qs  [[buffer(14)]],
    const constant uint4& ks  [[buffer(15)]],
    const constant uint4& vs  [[buffer(16)]],
    const constant uint4& os  [[buffer(17)]],
    const constant uint4& dos [[buffer(18)]],
    const constant uint4& dqs [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D == 64) ? 64 : 32;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;

    const uint b    = bh / nh;
    const uint h    = bh % nh;
    const uint kv_h = h / gqa;

    const device T*  Q_ptr  = Q  + b * qs[0]  + h    * qs[1];
    const device T*  K_ptr  = K  + b * ks[0]  + kv_h * ks[1];
    const device T*  V_ptr  = V  + b * vs[0]  + kv_h * vs[1];
    const device T*  dO_ptr = dO + b * dos[0] + h    * dos[1];
    device       T*  dQ_ptr = dQ + b * dqs[0] + h    * dqs[1];

    const bool valid_q = (q_row < qL);

    float q_reg[EPL]  = {};
    float do_reg[EPL] = {};
    float dq_acc[EPL] = {};
    float lse_val = 0.0f, d_vec = 0.0f;

    if (valid_q) {
        for (int e = 0; e < EPL; e++) {
            q_reg[e]  = float(Q_ptr[q_row  * qs[2]  + lane * EPL + e]);
            do_reg[e] = float(dO_ptr[q_row * dos[2] + lane * EPL + e]);
        }
        lse_val = LSE[bh * qL + q_row];
        d_vec   = Dv[bh * qL + q_row];
    }

    const uint tg_size = 32 * BQ;

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * ks[2] + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * vs[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool cv = !ic || (k_row <= q_row);
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < EPL; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];
            float score = cv ? (simd_sum(partial) * sc) : -INFINITY;
            float p_ij  = metal::precise::exp(score - lse_val);
            if (!valid_q) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < EPL; e++)
                dov += do_reg[e] * V_smem[j * D + lane * EPL + e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec);

            for (int e = 0; e < EPL; e++)
                dq_acc[e] += ds_ij * K_smem[j * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    for (int e = 0; e < EPL; e++)
        dQ_ptr[q_row * dqs[2] + lane * EPL + e] = T(dq_acc[e] * sc);
}

// ── backward dK + dV ─────────────────────────────────────────────────────────
// K and V live in per-simdgroup registers.  Q and dO are tiled through smem.
// Grid  : (B*H, ceil(kL/BK), 1),  TG : (32, BK, 1)
//
// Smem budget: Q_smem + dO_smem = BQS * D * sizeof(float) * 2 ≤ 32 KB
// → BQS = (D == 64) ? 64 : 32

template<typename T, int D>
[[kernel]] void flash_attn_bwd_dkdv(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    const device T*       O   [[buffer(3)]],   // unused, kept for dispatch compat
    const device T*       dO  [[buffer(4)]],
    const device float*   LSE [[buffer(5)]],
    const device float*   Dv  [[buffer(6)]],
    device       T*       dK  [[buffer(7)]],
    device       T*       dV  [[buffer(8)]],
    const constant uint&  qL  [[buffer(9)]],
    const constant uint&  kL  [[buffer(10)]],
    const constant uint&  gqa [[buffer(11)]],
    const constant uint&  nh  [[buffer(12)]],
    const constant float& sc  [[buffer(13)]],
    const constant bool&  ic  [[buffer(14)]],
    const constant uint4& qs  [[buffer(15)]],
    const constant uint4& ks  [[buffer(16)]],
    const constant uint4& vs  [[buffer(17)]],
    const constant uint4& os  [[buffer(18)]],
    const constant uint4& dos [[buffer(19)]],
    const constant uint4& dks [[buffer(20)]],
    const constant uint4& dvs [[buffer(21)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL  = (D + 31) / 32;
    constexpr int BK   = 32;
    constexpr int BQS  = (D == 64) ? 64 : 32;

    threadgroup float  Q_smem[BQS * D];
    threadgroup float dO_smem[BQS * D];

    const uint lane    = tid % 32;
    const uint k_local = tid / 32;
    const uint bh      = tgid.x;
    const uint k_row   = tgid.y * BK + k_local;
    const uint k_min   = tgid.y * BK;

    // bh encodes (batch, kv_head): bh = b * nh + kv_h  (nh == kvH here)
    const uint kv_h = bh % nh;
    const uint b    = bh / nh;

    const device T*  K_ptr  = K  + b * ks[0]  + kv_h * ks[1];
    const device T*  V_ptr  = V  + b * vs[0]  + kv_h * vs[1];
    device       T*  dK_ptr = dK + b * dks[0] + kv_h * dks[1];
    device       T*  dV_ptr = dV + b * dvs[0] + kv_h * dvs[1];

    const bool valid_k = (k_row < kL);

    float k_reg[EPL] = {};
    float v_reg[EPL] = {};
    if (valid_k) {
        for (int e = 0; e < EPL; e++) {
            k_reg[e] = float(K_ptr[k_row * ks[2] + lane * EPL + e]);
            v_reg[e] = float(V_ptr[k_row * vs[2] + lane * EPL + e]);
        }
    }

    float dk_acc[EPL] = {};
    float dv_acc[EPL] = {};

    const uint tg_size = 32 * BK;

    for (uint g = 0; g < gqa; g++) {
        const uint q_head = kv_h * gqa + g;
        const device T* Q_ptr  = Q  + b * qs[0]  + q_head * qs[1];
        const device T* dO_ptr = dO + b * dos[0] + q_head * dos[1];
        const uint bh_lse = b * (nh * gqa) + q_head;

    for (uint qb = 0; qb < qL; qb += BQS) {
        if (ic && qb + (uint)BQS - 1 < k_min) continue;

        for (uint i = tid; i < (uint)(BQS * D); i += tg_size) {
            uint r = qb + i / D;
            uint d = i % D;
            bool in = (r < qL);
            Q_smem[i]  = in ? float( Q_ptr[r * qs[2]  + d]) : 0.0f;
            dO_smem[i] = in ? float(dO_ptr[r * dos[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(qb + (uint)BQS, qL);

        for (uint q_row = qb; q_row < tile_end; ++q_row) {
            if (ic && k_row > q_row) continue;

            float lse_i   = LSE[bh_lse * qL + q_row];
            float d_vec_i = Dv[bh_lse * qL + q_row];
            int i = (int)(q_row - qb);

            float qk = 0.0f;
            for (int e = 0; e < EPL; e++)
                qk += Q_smem[i * D + lane * EPL + e] * k_reg[e];
            float p_ij = metal::precise::exp(simd_sum(qk) * sc - lse_i);
            if (!valid_k) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < EPL; e++)
                dov += dO_smem[i * D + lane * EPL + e] * v_reg[e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec_i);

            for (int e = 0; e < EPL; e++)
                dv_acc[e] += p_ij * dO_smem[i * D + lane * EPL + e];

            for (int e = 0; e < EPL; e++)
                dk_acc[e] += ds_ij * Q_smem[i * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    } // end gqa g-loop

    if (!valid_k) return;

    for (int e = 0; e < EPL; e++) {
        dK_ptr[k_row * dks[2] + lane * EPL + e] = T(dk_acc[e] * sc);
        dV_ptr[k_row * dvs[2] + lane * EPL + e] = T(dv_acc[e]);
    }
}

// ── explicit instantiation macros ────────────────────────────────────────────

#define INST_FLASH_FWD(T, D) \
  template [[host_name("flash_attn_fwd_" #T "_" #D)]] [[kernel]] \
  void flash_attn_fwd<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      device       T*        O   [[buffer(3)]],  \
      device       float*    LSE [[buffer(4)]],  \
      const constant uint&   qL  [[buffer(5)]],  \
      const constant uint&   kL  [[buffer(6)]],  \
      const constant uint&   gqa [[buffer(7)]],  \
      const constant uint&   nh  [[buffer(8)]],  \
      const constant float&  sc  [[buffer(9)]],  \
      const constant bool&   ic  [[buffer(10)]], \
      const constant uint4&  qs  [[buffer(11)]], \
      const constant uint4&  ks  [[buffer(12)]], \
      const constant uint4&  vs  [[buffer(13)]], \
      const constant uint4&  os  [[buffer(14)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_PRE(T, D) \
  template [[host_name("flash_attn_bwd_pre_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_preprocess<T, D>( \
      const device T*        dO  [[buffer(0)]], \
      const device T*        O   [[buffer(1)]], \
      device       float*    Dv  [[buffer(2)]], \
      const constant uint&   qL  [[buffer(3)]], \
      const constant uint&   nh  [[buffer(4)]], \
      const constant uint4&  dos [[buffer(5)]], \
      const constant uint4&  os  [[buffer(6)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DQ(T, D) \
  template [[host_name("flash_attn_bwd_dq_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dq<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      const device T*        O   [[buffer(3)]],  \
      const device T*        dO  [[buffer(4)]],  \
      const device float*    LSE [[buffer(5)]],  \
      const device float*    Dv  [[buffer(6)]],  \
      device       T*        dQ  [[buffer(7)]],  \
      const constant uint&   qL  [[buffer(8)]],  \
      const constant uint&   kL  [[buffer(9)]],  \
      const constant uint&   gqa [[buffer(10)]], \
      const constant uint&   nh  [[buffer(11)]], \
      const constant float&  sc  [[buffer(12)]], \
      const constant bool&   ic  [[buffer(13)]], \
      const constant uint4&  qs  [[buffer(14)]], \
      const constant uint4&  ks  [[buffer(15)]], \
      const constant uint4&  vs  [[buffer(16)]], \
      const constant uint4&  os  [[buffer(17)]], \
      const constant uint4&  dos [[buffer(18)]], \
      const constant uint4&  dqs [[buffer(19)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DKDV(T, D) \
  template [[host_name("flash_attn_bwd_dkdv_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dkdv<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      const device T*        O   [[buffer(3)]],  \
      const device T*        dO  [[buffer(4)]],  \
      const device float*    LSE [[buffer(5)]],  \
      const device float*    Dv  [[buffer(6)]],  \
      device       T*        dK  [[buffer(7)]],  \
      device       T*        dV  [[buffer(8)]],  \
      const constant uint&   qL  [[buffer(9)]],  \
      const constant uint&   kL  [[buffer(10)]], \
      const constant uint&   gqa [[buffer(11)]], \
      const constant uint&   nh  [[buffer(12)]], \
      const constant float&  sc  [[buffer(13)]], \
      const constant bool&   ic  [[buffer(14)]], \
      const constant uint4&  qs  [[buffer(15)]], \
      const constant uint4&  ks  [[buffer(16)]], \
      const constant uint4&  vs  [[buffer(17)]], \
      const constant uint4&  os  [[buffer(18)]], \
      const constant uint4&  dos [[buffer(19)]], \
      const constant uint4&  dks [[buffer(20)]], \
      const constant uint4&  dvs [[buffer(21)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_ALL(T) \
  INST_FLASH_FWD(T, 64)      \
  INST_FLASH_FWD(T, 128)     \
  INST_FLASH_BWD_PRE(T, 64)  \
  INST_FLASH_BWD_PRE(T, 128) \
  INST_FLASH_BWD_DQ(T, 64)   \
  INST_FLASH_BWD_DQ(T, 128)  \
  INST_FLASH_BWD_DKDV(T, 64) \
  INST_FLASH_BWD_DKDV(T, 128)

INST_FLASH_ALL(float)
INST_FLASH_ALL(half)
INST_FLASH_ALL(bfloat)
// ═══════════════════════════════════════════════════════════════════════════════
// Variable-length FlashAttention-2  (forward + backward)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Sequences of different lengths are packed end-to-end without padding.
//
// Layout  : Q  [H,   total_q, D]   K/V [kvH, total_k, D]
//           O  [H,   total_q, D]   LSE [H,   total_q]    (float)
//           Dv [H,   total_q]      (backward scratch, float)
//
// cu_seqlens : [B+1] cumulative token counts (int32, compatible with PyG batch.ptr)
// gqa        : H / kvH  (= 1 for standard MHA)
// wnd_left   : left window size  (-1 = unlimited; 0 = attend only to current token from left)
// wnd_right  : right window size (-1 = unlimited; 0 = causal, same as is_causal=true)
// alibi      : [H] ALiBi slopes (positive; bias = slope * position_delta, subtracted from score)
//              For causal:       delta = k_pos - q_pos  (<= 0)
//              For bidirectional: delta = -|k_pos - q_pos|  (<= 0)
//
// Template parameter EPL = elements per SIMD lane = ceil(D/32).
// D is passed at runtime via buffer, so ANY head_dim is supported without
// explicit per-D instantiation.  Dpad = EPL * 32 is the shared-memory stride
// (zero-padded when D is not a multiple of 32).
//
// Grid      : (H,   ceil(max_seqlen_q / BQ), B)   for forward, preprocess, dQ
//           : (kvH, ceil(max_seqlen_k / BK), B)   for dK+dV
// Threadgroup: 1024 flat threads  (32 lanes x 32 rows)
// ═══════════════════════════════════════════════════════════════════════════════

// ── varlen forward ────────────────────────────────────────────────────────────

template<typename T, int EPL>
[[kernel]] void flash_attn_varlen_fwd(
    const device T*       Q            [[buffer(0)]],
    const device T*       K            [[buffer(1)]],
    const device T*       V            [[buffer(2)]],
    device       T*       O            [[buffer(3)]],
    device       float*   LSE          [[buffer(4)]],
    const device uint*    cu_seqlens_q [[buffer(5)]],
    const device uint*    cu_seqlens_k [[buffer(6)]],
    const constant uint&  total_q      [[buffer(7)]],
    const constant uint&  total_k      [[buffer(8)]],
    const constant float& sc           [[buffer(9)]],
    const constant bool&  ic           [[buffer(10)]],
    const constant uint&  gqa          [[buffer(11)]],
    const constant int&   wnd_left     [[buffer(12)]],
    const constant int&   wnd_right    [[buffer(13)]],
    const device float*   alibi        [[buffer(14)]],
    const constant bool&  has_alibi    [[buffer(15)]],
    const constant uint&  D            [[buffer(16)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int Dpad = EPL * 32;
    constexpr int BQ   = 32;
    constexpr int BKV  = (Dpad <= 64) ? 64 : (Dpad <= 128) ? 32 : (Dpad <= 256) ? 16 : 8;

    threadgroup float K_smem[BKV * Dpad];
    threadgroup float V_smem[BKV * Dpad];

    const uint h    = tgid.x;
    const uint b    = tgid.z;
    const uint kv_h = h / gqa;
    const uint lane    = tid % 32;
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint q_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint k_start = cu_seqlens_k[b];
    const uint k_end   = cu_seqlens_k[b + 1];
    const uint qL = q_end - q_start;
    const uint kL = k_end - k_start;

    if (tgid.y * BQ >= qL) return;

    const uint q_row     = tgid.y * BQ + q_local;
    const uint q_row_max = tgid.y * BQ + BQ - 1;
    const bool valid_q   = (q_row < qL);

    const device T* Q_ptr = Q + h    * total_q * D + q_start * D;
    const device T* K_ptr = K + kv_h * total_k * D + k_start * D;
    const device T* V_ptr = V + kv_h * total_k * D + k_start * D;
    device       T* O_ptr = O + h    * total_q * D + q_start * D;

    float q_reg[EPL] = {};
    for (int e = 0; e < epl; e++)
        q_reg[e] = valid_q ? float(Q_ptr[q_row * D + lane * EPL + e]) : 0.0f;

    float acc[EPL] = {};
    float m = -INFINITY, l = 0.0f;

    const uint tg_size = 32 * BQ;

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_row_max) break;
        if (wnd_right >= 0 && (int)kb > (int)q_row_max + wnd_right) break;
        if (wnd_left >= 0 && (int)(kb + (uint)BKV - 1) + wnd_left < (int)(tgid.y * BQ)) continue;

        for (uint i = tid; i < (uint)(BKV * Dpad); i += tg_size) {
            uint r = kb + i / Dpad;
            uint d = i % Dpad;
            bool valid = (r < kL) && (d < (uint)D);
            K_smem[i] = valid ? float(K_ptr[r * D + d]) : 0.0f;
            V_smem[i] = valid ? float(V_ptr[r * D + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool in_causal = !ic || (k_row <= q_row);
            const bool in_wl = (wnd_left  < 0) || (int(k_row) + wnd_left  >= int(q_row));
            const bool in_wr = (wnd_right < 0) || (int(k_row) <= int(q_row) + wnd_right);
            const bool mask_ok = in_causal && in_wl && in_wr;
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < epl; e++)
                partial += q_reg[e] * K_smem[j * Dpad + lane * EPL + e];

            float score = mask_ok ? (simd_sum(partial) * sc) : -INFINITY;
            if (has_alibi && mask_ok) {
                int rel = int(k_row) - int(q_row);
                score += alibi[h] * float(ic ? rel : -abs(rel));
            }

            float m_new = max(m, score);
            float alpha = (m_new == -INFINITY) ? 1.0f : metal::precise::exp(m - m_new);
            float p_j   = (score  == -INFINITY) ? 0.0f : metal::precise::exp(score - m_new);
            m = m_new;
            l = l * alpha + p_j;

            for (int e = 0; e < epl; e++)
                acc[e] = acc[e] * alpha + p_j * V_smem[j * Dpad + lane * EPL + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int e = 0; e < epl; e++)
        O_ptr[q_row * D + lane * EPL + e] = T(acc[e] * inv_l);
    if (lane == 0)
        LSE[h * total_q + q_start + q_row] = m + metal::precise::log(l);
}

// ── varlen backward preprocess ────────────────────────────────────────────────
// Dv[h * total_q + q_start + q_row] = rowsum(dO * O)
// No GQA or window/ALiBi dependency: pure Q-side operation.
// Grid : (H, ceil(max_qL/BQ), B),  TG : 1024 flat threads

template<typename T, int EPL>
[[kernel]] void flash_attn_varlen_bwd_preprocess(
    const device T*       dO           [[buffer(0)]],
    const device T*       O            [[buffer(1)]],
    device       float*   Dv           [[buffer(2)]],
    const device uint*    cu_seqlens_q [[buffer(3)]],
    const constant uint&  total_q      [[buffer(4)]],
    const constant uint&  D            [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int BQ  = 32;

    const uint h       = tgid.x;
    const uint b       = tgid.z;
    const uint lane    = tid % 32;
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint q_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint qL      = q_end - q_start;

    if (tgid.y * BQ >= qL) return;

    const uint q_row   = tgid.y * BQ + q_local;
    const bool valid_q = (q_row < qL);

    const device T* dO_ptr = dO + h * total_q * D + q_start * D;
    const device T* O_ptr  = O  + h * total_q * D + q_start * D;

    float partial = 0.0f;
    if (valid_q) {
        for (int e = 0; e < epl; e++)
            partial += float(dO_ptr[q_row * D + lane * EPL + e])
                     * float( O_ptr[q_row * D + lane * EPL + e]);
    }
    float total = simd_sum(partial);
    if (lane == 0 && valid_q)
        Dv[h * total_q + q_start + q_row] = total;
}

// ── varlen backward dQ ────────────────────────────────────────────────────────
// Grid : (H, ceil(max_qL/BQ), B),  TG : 1024 flat threads


// ── varlen backward dQ SGMMA ──────────────────────────────────────────────────
// Replaces scalar simd_sum inner loop with tile_matmad.
// Natural-scale exp (no exp2 trick) — LSE is in natural log space.
// Grid: (H, ceil(max_q/BQ), B)   TG: WM*WN*32 flat threads

template<typename T, int BQ, int BK, int BD, int WM, int WN, bool DO_CAUSAL>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void flash_attn_varlen_bwd_dq_sgmma(
    const device T*       Q            [[buffer(0)]],
    const device T*       K            [[buffer(1)]],
    const device T*       V            [[buffer(2)]],
    const device T*       dO           [[buffer(3)]],
    const device float*   LSE          [[buffer(4)]],
    const device float*   Dv           [[buffer(5)]],
    device       T*       dQ           [[buffer(6)]],
    const device uint*    cu_seqlens_q [[buffer(7)]],
    const device uint*    cu_seqlens_k [[buffer(8)]],
    const device float*               alibi  [[buffer(9)]],
    const constant VarlenAttnParams&  params [[buffer(10)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    const uint  total_q   = params.total_q;
    const uint  total_k   = params.total_k;
    const float sc        = params.sc;
    const uint  gqa       = params.gqa;
    const int   wnd_left  = params.wnd_left;
    const int   wnd_right = params.wnd_right;
    const bool  has_alibi = (bool)params.has_alibi;
    using AccumType = c10::metal::accum_t<T>;
    constexpr int kFragSize = 8;
    using MMAFrag_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

    // Grid: tgid.x=H, tgid.y=q_block, tgid.z=B (same as scalar dQ)
    const int block_q = (int)tgid.y;
    const int h       = (int)tgid.x;
    const int b       = (int)tgid.z;
    const int kv_h    = h / (int)gqa;

    const uint q_start = cu_seqlens_q[b];
    const uint k_start = cu_seqlens_k[b];
    const int  qL = (int)(cu_seqlens_q[b+1] - q_start);
    const int  kL = (int)(cu_seqlens_k[b+1] - k_start);

    if (block_q * BQ >= qL) return;

    Q  += h    * (int)total_q * BD + ((int)q_start + block_q * BQ) * BD;
    K  += kv_h * (int)total_k * BD + (int)k_start * BD;
    V  += kv_h * (int)total_k * BD + (int)k_start * BD;
    dO += h    * (int)total_q * BD + ((int)q_start + block_q * BQ) * BD;
    dQ += h    * (int)total_q * BD + ((int)q_start + block_q * BQ) * BD;

    // Smem layouts
    constexpr short padQ   = 16 / sizeof(T);
    constexpr short padKc  = 16 / sizeof(T);
    constexpr short padKr  = 16 / sizeof(T);
    constexpr short LDQ    = BD + padQ;    // row-major Q/dO  (q_row*LDQ + d_col)
    constexpr short LDKc   = BK + padKc;  // col-major K^T   (k_row + d_col*LDKc)
    constexpr short LDKr   = BD + padKr;  // row-major K     (k_row*LDKr + d_col)

    // Q and dO: row-major BQ*(BD+pad), loaded once
    threadgroup T  Q_smem[BQ * LDQ];
    threadgroup T dO_smem[BQ * LDQ];
    // KV_col: col-major BD*(BK+pad), reused for K^T then V^T each K-block
    threadgroup T KV_col_smem[BD * LDKc];
    // K_row: row-major BK*(BD+pad), loaded alongside K_col each K-block
    threadgroup T K_row_smem[BK * LDKr];

    using QLoader    = BlockLoaderT<T, BQ, BD, LDQ,  1,    1, WM*WN*32>;
    using KColLoader = BlockLoaderT<T, BK, BD, 1,    LDKc, 0, WM*WN*32>;
    using KRowLoader = BlockLoaderT<T, BK, BD, LDKr, 1,    0, WM*WN*32>;
    // VColLoader same type as KColLoader, different src ptr
    using VColLoader = KColLoader;

    const int q_end  = min(block_q * BQ + BQ, qL);
    const int q_size = q_end - block_q * BQ;

    // Load Q and dO once (before K loop)
    QLoader loader_q(Q, BD, Q_smem, simd_group_id, simd_lane_id);
    QLoader loader_do(dO, BD, dO_smem, simd_group_id, simd_lane_id);
    if (q_size < BQ) {
        loader_q.load_safe(short2(BD, q_size));
        loader_do.load_safe(short2(BD, q_size));
    } else {
        loader_q.load_unsafe();
        loader_do.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SGMMA tile dimensions
    constexpr int TQ = BQ / (WM * WN * kFragSize);  // = 1
    constexpr int TK = BK / kFragSize;
    constexpr int TD = BD / kFragSize;
    static_assert(TQ == 1, "TQ must be 1");

    const short2 coord = MMAFrag_t::get_coord(simd_lane_id);
    const short sm = coord.y;
    const short sn = coord.x;
    const short tm = (short)(kFragSize * TQ * simd_group_id);

    // Per-thread row: lse and delta (one per Q-row owned by this thread)
    const int my_local_row  = (int)(tm + sm);
    const int my_global_row = (int)q_start + block_q * BQ + my_local_row;
    const bool valid_q = (my_local_row < q_size);

    AccumType lse_val   = valid_q ? AccumType(LSE[h * (int)total_q + my_global_row]) : AccumType(0);
    AccumType delta_val = valid_q ? AccumType(Dv[h  * (int)total_q + my_global_row]) : AccumType(0);

    // SGMMA tile declarations
    MMATile<AccumType, TQ, 1,  MMAFrag_t> Qtile;
    MMATile<AccumType, 1,  TK, MMAFrag_t> KCtile;  // K^T (col-major)
    MMATile<AccumType, TQ, TK, MMAFrag_t> Stile;   // scores and P/dS
    MMATile<AccumType, TQ, 1,  MMAFrag_t> dOtile;
    MMATile<AccumType, 1,  TK, MMAFrag_t> VCtile;  // V^T (col-major)
    MMATile<AccumType, TQ, TK, MMAFrag_t> dov_tile;
    MMATile<AccumType, 1,  1,  MMAFrag_t> KRtile;  // K row (for dQ+=dS*K)
    MMATile<AccumType, TQ, TD, MMAFrag_t> dQtile;
    dQtile.clear();

    // Smem access offsets
    const short Qs_off  = (tm + sm) * LDQ  + sn;  // row-major Q/dO
    const short Kcs_off = sm * LDKc + sn;          // col-major K^T/V^T
    const short Krs_off = sm * LDKr + sn;          // row-major K (for dQ step)
    constexpr short Qs_stride  = kFragSize;
    constexpr short Kcs_stride = kFragSize * LDKc;

    // Window/causal precompute
    int kb_lim = (kL + BK - 1) / BK;
    if constexpr (DO_CAUSAL)
        kb_lim = min(kb_lim, (block_q * BQ + q_size + BK - 1) / BK);
    if (wnd_right >= 0)
        kb_lim = min(kb_lim, (block_q * BQ + q_size + wnd_right + BK - 1) / BK);

    KColLoader loader_kc(K, BD, KV_col_smem, simd_group_id, simd_lane_id);
    KRowLoader loader_kr(K, BD, K_row_smem,  simd_group_id, simd_lane_id);
    VColLoader loader_vc(V, BD, KV_col_smem, simd_group_id, simd_lane_id);

    for (int kb = 0; kb < kb_lim; kb++) {
        // Sliding-window early-skip
        if (wnd_left >= 0 && (kb + 1) * BK <= block_q * BQ - wnd_left) {
            loader_kc.next(); loader_kr.next(); loader_vc.next(); continue;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int k_end  = min(kb * BK + BK, kL);
        const int k_size = k_end - kb * BK;

        // Load K in both col-major and row-major simultaneously
        if (k_size < BK) {
            loader_kc.load_safe(short2(BD, k_size));
            loader_kr.load_safe(short2(BD, k_size));
        } else {
            loader_kc.load_unsafe();
            loader_kr.load_unsafe();
        }
        Stile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q * K^T  (natural-scale, no exp2 pre-scaling)
        PREFILL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.template load<T, 1, 1, LDQ, 1>(&Q_smem[Qs_off + dd * Qs_stride]);
            KCtile.template load<T, 1, 1, LDKc, 1>(&KV_col_smem[Kcs_off + dd * Kcs_stride]);
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(Stile, Qtile, KCtile, Stile);
        }

        // Mask partial K block
        if (k_size < BK) {
            PREFILL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                PREFILL_PRAGMA_UNROLL
                for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                    if ((int)sn + ik * kFragSize + jj >= k_size)
                        Stile.frag_at(0, ik)[jj] = -INFINITY;
                }
            }
        }

        // Causal mask
        if constexpr (DO_CAUSAL) {
            int q_row = block_q * BQ + (int)(tm + sm);
            PREFILL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                int col_base = kb * BK + (int)sn + ik * kFragSize;
                PREFILL_PRAGMA_UNROLL
                for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++)
                    if (q_row < col_base + jj) Stile.frag_at(0, ik)[jj] = -INFINITY;
            }
        }

        // Window mask
        if (wnd_left >= 0 || wnd_right >= 0) {
            int q_row = block_q * BQ + (int)(tm + sm);
            PREFILL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                int col_base = kb * BK + (int)sn + ik * kFragSize;
                PREFILL_PRAGMA_UNROLL
                for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                    int col = col_base + jj;
                    bool ok = (wnd_left  < 0 || col + wnd_left  >= q_row)
                           && (wnd_right < 0 || col             <= q_row + wnd_right);
                    if (!ok) Stile.frag_at(0, ik)[jj] = -INFINITY;
                }
            }
        }

        // P = exp(S*sc + alibi_bias - lse): ALiBi applied after sc multiplication
        {
            int q_row_p = block_q * BQ + (int)(tm + sm);
            PREFILL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                int col_base = kb * BK + (int)sn + ik * kFragSize;
                PREFILL_PRAGMA_UNROLL
                for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                    AccumType s = Stile.frag_at(0, ik)[jj];
                    if (valid_q && s > -INFINITY) {
                        AccumType score = s * sc;
                        if (has_alibi) {
                            int rel = (col_base + jj) - q_row_p;
                            score += AccumType(alibi[h]) *
                                AccumType(DO_CAUSAL ? rel : -abs(rel));
                        }
                        Stile.frag_at(0, ik)[jj] = metal::precise::exp(score - lse_val);
                    } else {
                        Stile.frag_at(0, ik)[jj] = AccumType(0);
                    }
                }
            }
        }

        // Load V into KV_col_smem (reuse, col-major for V^T)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_size < BK) loader_vc.load_safe(short2(BD, k_size));
        else             loader_vc.load_unsafe();
        dov_tile.clear();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dov = dO * V^T  (same structure as S = Q * K^T)
        PREFILL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            dOtile.template load<T, 1, 1, LDQ, 1>(&dO_smem[Qs_off + dd * Qs_stride]);
            VCtile.template load<T, 1, 1, LDKc, 1>(&KV_col_smem[Kcs_off + dd * Kcs_stride]);
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(dov_tile, dOtile, VCtile, dov_tile);
        }

        // dS = P * (dov - delta)  element-wise, per owned row
        PREFILL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            PREFILL_PRAGMA_UNROLL
            for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                Stile.frag_at(0, ik)[jj] *=
                    (dov_tile.frag_at(0, ik)[jj] - delta_val);
            }
        }

        // dQ += dS * K  (same structure as O += P * V in forward)
        // K_row_smem was loaded alongside K_col_smem in step 1 above
        PREFILL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++)
        PREFILL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++)
        PREFILL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
            KRtile.template load<T, 1, 1, LDKr, 1>(
                &K_row_smem[Krs_off + ik * kFragSize * LDKr + id * kFragSize]);
            if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
            MMAFrag_t::mma(dQtile.frag_at(iq, id),
                           Stile.frag_at(iq, ik),
                           KRtile.frag_at(0, 0),
                           dQtile.frag_at(iq, id));
        }

        loader_kc.next();
        loader_kr.next();
        loader_vc.next();
    }

    // dQ *= sc, then write out
    PREFILL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++)
        PREFILL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++)
            dQtile.frag_at(0, id)[jj] *= sc;

    device T* dQ_tile = dQ + (int)(tm + sm) * BD + sn;
    if (q_size < BQ) {
        if (my_local_row < q_size && sn < BD)
            dQtile.template store_safe<T, 1, 1>(dQ_tile, BD,
                short2(BD - sn, q_size - my_local_row));
    } else {
        dQtile.template store<T, 1, 1>(dQ_tile, BD);
    }
}

#define INST_VARLEN_BWD_DQ_SGMMA(T, BQ, BK, BD, WM, WN) \
  template [[host_name("flash_attn_varlen_bwd_dq_sgmma_" #T \
    "_bq" #BQ "_bk" #BK "_bd" #BD "_wm" #WM "_wn" #WN "_causal0")]] \
  [[kernel]] void flash_attn_varlen_bwd_dq_sgmma<T,BQ,BK,BD,WM,WN,false>( \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], \
      const device T* V [[buffer(2)]], const device T* dO [[buffer(3)]], \
      const device float* LSE [[buffer(4)]], const device float* Dv [[buffer(5)]], \
      device T* dQ [[buffer(6)]], const device uint* cu_seqlens_q [[buffer(7)]], \
      const device uint* cu_seqlens_k [[buffer(8)]], \
      const device float* alibi [[buffer(9)]], \
      const constant VarlenAttnParams& params [[buffer(10)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]); \
  template [[host_name("flash_attn_varlen_bwd_dq_sgmma_" #T \
    "_bq" #BQ "_bk" #BK "_bd" #BD "_wm" #WM "_wn" #WN "_causal1")]] \
  [[kernel]] void flash_attn_varlen_bwd_dq_sgmma<T,BQ,BK,BD,WM,WN,true>( \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], \
      const device T* V [[buffer(2)]], const device T* dO [[buffer(3)]], \
      const device float* LSE [[buffer(4)]], const device float* Dv [[buffer(5)]], \
      device T* dQ [[buffer(6)]], const device uint* cu_seqlens_q [[buffer(7)]], \
      const device uint* cu_seqlens_k [[buffer(8)]], \
      const device float* alibi [[buffer(9)]], \
      const constant VarlenAttnParams& params [[buffer(10)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]);

// D=64: BQ=32 BK=32 BD=64  WM=4 WN=1
// D=128: BQ=32 BK=16 BD=128 WM=4 WN=1
INST_VARLEN_BWD_DQ_SGMMA(half,   32, 32, 64,  4, 1)
INST_VARLEN_BWD_DQ_SGMMA(bfloat, 32, 32, 64,  4, 1)
INST_VARLEN_BWD_DQ_SGMMA(float,  32, 32, 64,  4, 1)
INST_VARLEN_BWD_DQ_SGMMA(half,   32, 16, 128, 4, 1)
INST_VARLEN_BWD_DQ_SGMMA(bfloat, 32, 16, 128, 4, 1)
INST_VARLEN_BWD_DQ_SGMMA(float,  32, 16, 128, 4, 1)

// ── varlen backward dK/dV SGMMA ──────────────────────────────────────────────
// K-outer: one TG per K-tile, inner loop over all GQA Q-heads and Q-blocks.
// WM=4 (128 threads) for BK=32 (D=64); WM=2 (64 threads) for BK=16 (D=128).
// Grid: (kvH, k_tiles, B)

template<typename T, int BQ, int BK, int BD, int WM, int WN, bool DO_CAUSAL>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void flash_attn_varlen_bwd_dkdv_sgmma(
    const device T*       Q            [[buffer(0)]],
    const device T*       K            [[buffer(1)]],
    const device T*       V            [[buffer(2)]],
    const device T*       dO           [[buffer(3)]],
    const device float*   LSE          [[buffer(4)]],
    const device float*   Dv           [[buffer(5)]],
    device       T*       dK           [[buffer(6)]],
    device       T*       dV           [[buffer(7)]],
    const device uint*    cu_seqlens_q [[buffer(8)]],
    const device uint*    cu_seqlens_k [[buffer(9)]],
    const device float*               alibi  [[buffer(10)]],
    const constant VarlenAttnParams&  params [[buffer(11)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    const uint  total_q   = params.total_q;
    const uint  total_k   = params.total_k;
    const float sc        = params.sc;
    const uint  gqa       = params.gqa;
    const int   wnd_left  = params.wnd_left;
    const int   wnd_right = params.wnd_right;
    const bool  has_alibi = (bool)params.has_alibi;
    using AccumType = c10::metal::accum_t<T>;
    constexpr int kFragSize = 8;
    using MMAFrag_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

    // Grid: (kvH, k_block, B)
    const int k_block = (int)tgid.y;
    const int kv_h    = (int)tgid.x;
    const int b       = (int)tgid.z;

    const uint k_start = cu_seqlens_k[b];
    const uint q_start = cu_seqlens_q[b];
    const int  kL = (int)(cu_seqlens_k[b+1] - k_start);
    const int  qL = (int)(cu_seqlens_q[b+1] - q_start);

    if (k_block * BK >= kL) return;

    K  += kv_h * (int)total_k * BD + ((int)k_start + k_block * BK) * BD;
    V  += kv_h * (int)total_k * BD + ((int)k_start + k_block * BK) * BD;
    dK += kv_h * (int)total_k * BD + ((int)k_start + k_block * BK) * BD;
    dV += kv_h * (int)total_k * BD + ((int)k_start + k_block * BK) * BD;

    // Smem strides with bank-conflict padding
    constexpr short padK  = 16 / (short)sizeof(T);
    constexpr short padQ  = 16 / (short)sizeof(T);
    constexpr short LDKr  = BD + padK;   // row-major K/V:   K[k*LDKr + d]
    constexpr short LDQc  = BQ + padQ;   // col-major Q/dO:  Q[q + d*LDQc]
    constexpr short LDQr  = BD + padQ;   // row-major Q/dO:  Q[q*LDQr + d]

    // Shared memory allocations
    threadgroup T     K_smem[BK * LDKr];   // K row-major, permanent
    threadgroup T     V_smem[BK * LDKr];   // V row-major, permanent
    threadgroup T     QC_smem[BD * LDQc];  // col-major Q or dO (alternating per Q-tile)
    threadgroup T     QR_smem[BQ * LDQr];  // row-major dO then Q (alternating per Q-tile)
    threadgroup float lse_smem[BQ];
    threadgroup float delta_smem[BQ];

    using KVLoader   = BlockLoaderT<T, BK, BD, LDKr, 1,    0, WM*WN*32>;
    using QColLoader = BlockLoaderT<T, BQ, BD, 1,    LDQc, 0, WM*WN*32>;
    using QRowLoader = BlockLoaderT<T, BQ, BD, LDQr, 1,    0, WM*WN*32>;

    const int k_end  = min(k_block * BK + BK, kL);
    const int k_size = k_end - k_block * BK;

    // Load K and V once before GQA/Q loop
    {
        KVLoader lk(K, BD, K_smem, simd_group_id, simd_lane_id);
        KVLoader lv(V, BD, V_smem, simd_group_id, simd_lane_id);
        if (k_size < BK) {
            lk.load_safe(short2(BD, k_size));
            lv.load_safe(short2(BD, k_size));
        } else {
            lk.load_unsafe();
            lv.load_unsafe();
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SGMMA tile dimensions
    constexpr int TK_out = BK / (WM * WN * kFragSize);  // = 1
    constexpr int TQ     = BQ / kFragSize;               // = 4 (BQ=32)
    constexpr int TD     = BD / kFragSize;               // = 8 or 16
    static_assert(TK_out == 1, "TK_out must be 1 for this kernel");

    const short2 coord = MMAFrag_t::get_coord(simd_lane_id);
    const short sm = coord.y;   // row within 8x8 fragment (K dim)
    const short sn = coord.x;   // col within 8x8 fragment
    const short tk = (short)(kFragSize * TK_out * (int)simd_group_id);

    const int my_k_local = (int)(tk + sm);
    const bool valid_k   = (my_k_local < k_size);

    // Smem access offsets
    const short Ks_off        = (tk + sm) * LDKr + sn;   // K/V row-major: row tk+sm
    const short Qcs_off       = sm * LDQc + sn;           // Q/dO col-major: D-row sm
    const short QRs_off       = sm * LDQr + sn;           // Q/dO row-major: row sm
    constexpr short Qcs_dstride = kFragSize * LDQc;        // D-tile stride in col-major buf

    // Accumulators (across all GQA heads and Q-blocks)
    MMATile<AccumType, TK_out, TD, MMAFrag_t> dKtile;
    MMATile<AccumType, TK_out, TD, MMAFrag_t> dVtile;
    dKtile.clear();
    dVtile.clear();

    // Working tiles (reused per Q-tile)
    MMATile<AccumType, TK_out, TQ, MMAFrag_t> Stile;
    MMATile<AccumType, TK_out, TQ, MMAFrag_t> dov_tile;
    MMATile<AccumType, TK_out, 1,  MMAFrag_t> KRfrag;
    MMATile<AccumType, TK_out, 1,  MMAFrag_t> VRfrag;
    MMATile<AccumType, 1,      TQ, MMAFrag_t> QCtile;
    MMATile<AccumType, 1,      TQ, MMAFrag_t> dOCtile;
    MMATile<AccumType, 1,      TD, MMAFrag_t> dORtile;
    MMATile<AccumType, 1,      TD, MMAFrag_t> QRtile;
    MMATile<AccumType, TK_out, 1,  MMAFrag_t> pfrag;   // single P or dS fragment

    // First 32 threads handle LSE/delta smem loads (BQ=32 values)
    const uint linear_tid = (uint)simd_group_id * 32 + (uint)simd_lane_id;

    for (uint g = 0; g < gqa; g++) {
        const uint q_h = kv_h * gqa + g;
        const device T* Q_h  = Q  + q_h * (int)total_q * BD + (int)q_start * BD;
        const device T* dO_h = dO + q_h * (int)total_q * BD + (int)q_start * BD;

        const int qb_lim = (qL + BQ - 1) / BQ;

        for (int qb = 0; qb < qb_lim; qb++) {
            // Skip Q-blocks that have no overlap with this K-block
            if constexpr (DO_CAUSAL) {
                // Causal: K <= Q. Skip if all Q in this block < K_min.
                if ((qb + 1) * BQ <= k_block * BK) continue;
            }
            if (wnd_right >= 0 && qb * BQ > k_block * BK + k_size - 1 + wnd_right) continue;
            if (wnd_left  >= 0 && (qb + 1) * BQ <= k_block * BK - wnd_left)         continue;

            const int q_end  = min(qb * BQ + BQ, qL);
            const int q_size = q_end - qb * BQ;

            // ── Phase 1: load Q col-major, dO row-major, lse, delta ──────────
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                QColLoader lqc(Q_h  + qb * BQ * BD, BD, QC_smem, simd_group_id, simd_lane_id);
                QRowLoader ldor(dO_h + qb * BQ * BD, BD, QR_smem, simd_group_id, simd_lane_id);
                if (q_size < BQ) {
                    lqc.load_safe(short2(BD, q_size));
                    ldor.load_safe(short2(BD, q_size));
                } else {
                    lqc.load_unsafe();
                    ldor.load_unsafe();
                }
            }
            if (linear_tid < (uint)BQ) {
                int q_gbl = (int)q_start + qb * BQ + (int)linear_tid;
                bool vq   = ((int)linear_tid < q_size);
                lse_smem  [(int)linear_tid] = vq ? LSE[q_h * (int)total_q + q_gbl] : 0.0f;
                delta_smem[(int)linear_tid] = vq ? Dv [q_h * (int)total_q + q_gbl] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── S = K * Q^T ───────────────────────────────────────────────────
            Stile.clear();
            PREFILL_PRAGMA_UNROLL
            for (short dd = 0; dd < TD; dd++) {
                simdgroup_barrier(mem_flags::mem_none);
                KRfrag.template load<T, 1, 1, LDKr, 1>(&K_smem[Ks_off + dd * kFragSize]);
                QCtile.template load<T, 1, 1, LDQc, 1>(&QC_smem[Qcs_off + dd * Qcs_dstride]);
                simdgroup_barrier(mem_flags::mem_none);
                tile_matmad(Stile, KRfrag, QCtile, Stile);
            }

            // ── Masks ─────────────────────────────────────────────────────────
            if (q_size < BQ) {
                PREFILL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++)
                    PREFILL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++)
                        if (iq * kFragSize + (int)sn + jj >= q_size)
                            Stile.frag_at(0, iq)[jj] = -INFINITY;
            }
            if constexpr (DO_CAUSAL) {
                const int k_grow = k_block * BK + my_k_local;
                PREFILL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    const int q_base = qb * BQ + (int)sn + iq * kFragSize;
                    PREFILL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++)
                        if (k_grow > q_base + jj)
                            Stile.frag_at(0, iq)[jj] = -INFINITY;
                }
            }
            if (wnd_left >= 0 || wnd_right >= 0) {
                const int k_grow = k_block * BK + my_k_local;
                PREFILL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    const int q_base = qb * BQ + (int)sn + iq * kFragSize;
                    PREFILL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                        int q_col = q_base + jj;
                        bool ok = (wnd_left  < 0 || k_grow <= q_col + wnd_left)
                               && (wnd_right < 0 || k_grow >= q_col - wnd_right);
                        if (!ok) Stile.frag_at(0, iq)[jj] = -INFINITY;
                    }
                }
            }

            // ── Softmax: P = exp(S*sc + alibi - lse) ─────────────────────────
            {
                const int k_grow = k_block * BK + my_k_local;
                PREFILL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    const int q_base = qb * BQ + (int)sn + iq * kFragSize;
                    PREFILL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                        const int  q_local = iq * kFragSize + (int)sn + jj;
                        const AccumType s  = Stile.frag_at(0, iq)[jj];
                        if (valid_k && s > -INFINITY && q_local < q_size) {
                            AccumType score = s * AccumType(sc);
                            if (has_alibi) {
                                int rel = k_grow - (q_base + jj);
                                score += AccumType(alibi[q_h]) *
                                    AccumType(DO_CAUSAL ? rel : -abs(rel));
                            }
                            Stile.frag_at(0, iq)[jj] = AccumType(
                                metal::precise::exp(float(score) - lse_smem[q_local]));
                        } else {
                            Stile.frag_at(0, iq)[jj] = AccumType(0);
                        }
                    }
                }
            }

            // ── dV += P * dO_row ──────────────────────────────────────────────
            PREFILL_PRAGMA_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
                PREFILL_PRAGMA_UNROLL
                for (short j = 0; j < MMAFrag_t::kElemCols; j++)
                    pfrag.frag_at(0, 0)[j] = Stile.frag_at(0, iq)[j];
                simdgroup_barrier(mem_flags::mem_none);
                dORtile.template load<T, 1, 1, LDQr, 1>(&QR_smem[QRs_off + iq * kFragSize * LDQr]);
                simdgroup_barrier(mem_flags::mem_none);
                tile_matmad(dVtile, pfrag, dORtile, dVtile);
            }

            // ── Reload: dO col-major + Q row-major ────────────────────────────
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                QColLoader ldoc(dO_h + qb * BQ * BD, BD, QC_smem, simd_group_id, simd_lane_id);
                QRowLoader lqr (Q_h  + qb * BQ * BD, BD, QR_smem, simd_group_id, simd_lane_id);
                if (q_size < BQ) {
                    ldoc.load_safe(short2(BD, q_size));
                    lqr.load_safe(short2(BD, q_size));
                } else {
                    ldoc.load_unsafe();
                    lqr.load_unsafe();
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── dov = V * dO^T ────────────────────────────────────────────────
            dov_tile.clear();
            PREFILL_PRAGMA_UNROLL
            for (short dd = 0; dd < TD; dd++) {
                simdgroup_barrier(mem_flags::mem_none);
                VRfrag.template load<T, 1, 1, LDKr, 1>(&V_smem[Ks_off + dd * kFragSize]);
                dOCtile.template load<T, 1, 1, LDQc, 1>(&QC_smem[Qcs_off + dd * Qcs_dstride]);
                simdgroup_barrier(mem_flags::mem_none);
                tile_matmad(dov_tile, VRfrag, dOCtile, dov_tile);
            }

            // ── dS = P * (dov - delta) element-wise ──────────────────────────
            PREFILL_PRAGMA_UNROLL
            for (short iq = 0; iq < TQ; iq++)
                PREFILL_PRAGMA_UNROLL
                for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                    const int q_local = iq * kFragSize + (int)sn + jj;
                    const float dv    = (q_local < q_size) ? delta_smem[q_local] : 0.0f;
                    Stile.frag_at(0, iq)[jj] *=
                        (dov_tile.frag_at(0, iq)[jj] - AccumType(dv));
                }

            // ── dK += dS * Q_row ──────────────────────────────────────────────
            PREFILL_PRAGMA_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
                PREFILL_PRAGMA_UNROLL
                for (short j = 0; j < MMAFrag_t::kElemCols; j++)
                    pfrag.frag_at(0, 0)[j] = Stile.frag_at(0, iq)[j];
                simdgroup_barrier(mem_flags::mem_none);
                QRtile.template load<T, 1, 1, LDQr, 1>(&QR_smem[QRs_off + iq * kFragSize * LDQr]);
                simdgroup_barrier(mem_flags::mem_none);
                tile_matmad(dKtile, pfrag, QRtile, dKtile);
            }
        }  // qb loop
    }  // gqa loop

    // ── Scale dK by sc and write dK, dV ──────────────────────────────────────
    PREFILL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++)
        PREFILL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++)
            dKtile.frag_at(0, id)[jj] *= AccumType(sc);

    if (valid_k) {
        device T* dK_tile = dK + my_k_local * BD + (int)sn;
        device T* dV_tile = dV + my_k_local * BD + (int)sn;
        dKtile.template store<T, 1, 1>(dK_tile, BD);
        dVtile.template store<T, 1, 1>(dV_tile, BD);
    }
}

#define INST_VARLEN_BWD_DKDV_SGMMA(T, BQ, BK, BD, WM, WN) \
  template [[host_name("flash_attn_varlen_bwd_dkdv_sgmma_" #T \
    "_bq" #BQ "_bk" #BK "_bd" #BD "_wm" #WM "_wn" #WN "_causal0")]] \
  [[kernel]] void flash_attn_varlen_bwd_dkdv_sgmma<T,BQ,BK,BD,WM,WN,false>( \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], \
      const device T* V [[buffer(2)]], const device T* dO [[buffer(3)]], \
      const device float* LSE [[buffer(4)]], const device float* Dv [[buffer(5)]], \
      device T* dK [[buffer(6)]], device T* dV [[buffer(7)]], \
      const device uint* cu_seqlens_q [[buffer(8)]], \
      const device uint* cu_seqlens_k [[buffer(9)]], \
      const device float* alibi [[buffer(10)]], \
      const constant VarlenAttnParams& params [[buffer(11)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]); \
  template [[host_name("flash_attn_varlen_bwd_dkdv_sgmma_" #T \
    "_bq" #BQ "_bk" #BK "_bd" #BD "_wm" #WM "_wn" #WN "_causal1")]] \
  [[kernel]] void flash_attn_varlen_bwd_dkdv_sgmma<T,BQ,BK,BD,WM,WN,true>( \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], \
      const device T* V [[buffer(2)]], const device T* dO [[buffer(3)]], \
      const device float* LSE [[buffer(4)]], const device float* Dv [[buffer(5)]], \
      device T* dK [[buffer(6)]], device T* dV [[buffer(7)]], \
      const device uint* cu_seqlens_q [[buffer(8)]], \
      const device uint* cu_seqlens_k [[buffer(9)]], \
      const device float* alibi [[buffer(10)]], \
      const constant VarlenAttnParams& params [[buffer(11)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]);

// D=64:  BK=32 WM=4 -> 128 threads/TG;  D=128: BK=16 WM=2 -> 64 threads/TG
INST_VARLEN_BWD_DKDV_SGMMA(half,   32, 32, 64,  4, 1)
INST_VARLEN_BWD_DKDV_SGMMA(bfloat, 32, 32, 64,  4, 1)
INST_VARLEN_BWD_DKDV_SGMMA(float,  32, 32, 64,  4, 1)
INST_VARLEN_BWD_DKDV_SGMMA(half,   32, 16, 128, 2, 1)
INST_VARLEN_BWD_DKDV_SGMMA(bfloat, 32, 16, 128, 2, 1)
INST_VARLEN_BWD_DKDV_SGMMA(float,  32, 16, 128, 2, 1)



template<typename T, int EPL>
[[kernel]] void flash_attn_varlen_bwd_dq(
    const device T*       Q            [[buffer(0)]],
    const device T*       K            [[buffer(1)]],
    const device T*       V            [[buffer(2)]],
    const device T*       dO           [[buffer(3)]],
    const device float*   LSE          [[buffer(4)]],
    const device float*   Dv           [[buffer(5)]],
    device       T*       dQ           [[buffer(6)]],
    const device uint*    cu_seqlens_q [[buffer(7)]],
    const device uint*    cu_seqlens_k [[buffer(8)]],
    const constant uint&  total_q      [[buffer(9)]],
    const constant uint&  total_k      [[buffer(10)]],
    const constant float& sc           [[buffer(11)]],
    const constant bool&  ic           [[buffer(12)]],
    const constant uint&  gqa          [[buffer(13)]],
    const constant int&   wnd_left     [[buffer(14)]],
    const constant int&   wnd_right    [[buffer(15)]],
    const device float*   alibi        [[buffer(16)]],
    const constant bool&  has_alibi    [[buffer(17)]],
    const constant uint&  D            [[buffer(18)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int Dpad = EPL * 32;
    constexpr int BQ   = 32;
    constexpr int BKV  = (Dpad <= 64) ? 64 : (Dpad <= 128) ? 32 : (Dpad <= 256) ? 16 : 8;

    threadgroup float K_smem[BKV * Dpad];
    threadgroup float V_smem[BKV * Dpad];

    const uint h    = tgid.x;
    const uint b    = tgid.z;
    const uint kv_h = h / gqa;
    const uint lane    = tid % 32;
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint q_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint k_start = cu_seqlens_k[b];
    const uint k_end   = cu_seqlens_k[b + 1];
    const uint qL = q_end - q_start;
    const uint kL = k_end - k_start;

    if (tgid.y * BQ >= qL) return;

    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;
    const bool valid_q = (q_row < qL);

    const device T* Q_ptr  = Q  + h    * total_q * D + q_start * D;
    const device T* K_ptr  = K  + kv_h * total_k * D + k_start * D;
    const device T* V_ptr  = V  + kv_h * total_k * D + k_start * D;
    const device T* dO_ptr = dO + h    * total_q * D + q_start * D;
    device       T* dQ_ptr = dQ + h    * total_q * D + q_start * D;

    float q_reg[EPL]  = {};
    float do_reg[EPL] = {};
    float dq_acc[EPL] = {};
    float lse_val = 0.0f, d_vec = 0.0f;

    if (valid_q) {
        for (int e = 0; e < epl; e++) {
            q_reg[e]  = float(Q_ptr[q_row * D + lane * EPL + e]);
            do_reg[e] = float(dO_ptr[q_row * D + lane * EPL + e]);
        }
        lse_val = LSE[h * total_q + q_start + q_row];
        d_vec   = Dv[h * total_q + q_start + q_row];
    }

    const uint tg_size = 32 * BQ;

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;
        if (wnd_right >= 0 && (int)kb > (int)q_max + wnd_right) break;
        if (wnd_left  >= 0 && (int)(kb + (uint)BKV - 1) + wnd_left < (int)(tgid.y * BQ)) continue;

        for (uint i = tid; i < (uint)(BKV * Dpad); i += tg_size) {
            uint r = kb + i / Dpad;
            uint d = i % Dpad;
            bool valid = (r < kL) && (d < (uint)D);
            K_smem[i] = valid ? float(K_ptr[r * D + d]) : 0.0f;
            V_smem[i] = valid ? float(V_ptr[r * D + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool in_causal = !ic || (k_row <= q_row);
            const bool in_wl = (wnd_left  < 0) || (int(k_row) + wnd_left  >= int(q_row));
            const bool in_wr = (wnd_right < 0) || (int(k_row) <= int(q_row) + wnd_right);
            const bool mask_ok = in_causal && in_wl && in_wr;
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < epl; e++)
                partial += q_reg[e] * K_smem[j * Dpad + lane * EPL + e];

            float score = mask_ok ? (simd_sum(partial) * sc) : -INFINITY;
            if (has_alibi && mask_ok) {
                int rel = int(k_row) - int(q_row);
                score += alibi[h] * float(ic ? rel : -abs(rel));
            }
            float p_ij  = metal::precise::exp(score - lse_val);
            if (!valid_q || !mask_ok) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < epl; e++)
                dov += do_reg[e] * V_smem[j * Dpad + lane * EPL + e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec);

            for (int e = 0; e < epl; e++)
                dq_acc[e] += ds_ij * K_smem[j * Dpad + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;
    for (int e = 0; e < epl; e++)
        dQ_ptr[q_row * D + lane * EPL + e] = T(dq_acc[e] * sc);
}

// ── varlen backward dK + dV ───────────────────────────────────────────────────
// K/V stay in per-simdgroup registers; Q+dO are tiled through smem.
// Grid : (kvH, ceil(max_kL/BK), B),  TG : 1024 flat threads
//
// GQA: tgid.x is the kv-head index. For each kv-head we loop over all
// gqa query heads that share it, accumulating dK and dV.
//
// Window + ALiBi: applied when recomputing p_ij from Q·K to match forward pass.

template<typename T, int EPL>
[[kernel]] void flash_attn_varlen_bwd_dkdv(
    const device T*       Q            [[buffer(0)]],
    const device T*       K            [[buffer(1)]],
    const device T*       V            [[buffer(2)]],
    const device T*       dO           [[buffer(3)]],
    const device float*   LSE          [[buffer(4)]],
    const device float*   Dv           [[buffer(5)]],
    device       T*       dK           [[buffer(6)]],
    device       T*       dV           [[buffer(7)]],
    const device uint*    cu_seqlens_q [[buffer(8)]],
    const device uint*    cu_seqlens_k [[buffer(9)]],
    const constant uint&  total_q      [[buffer(10)]],
    const constant uint&  total_k      [[buffer(11)]],
    const constant float& sc           [[buffer(12)]],
    const constant bool&  ic           [[buffer(13)]],
    const constant uint&  gqa          [[buffer(14)]],
    const constant int&   wnd_left     [[buffer(15)]],
    const constant int&   wnd_right    [[buffer(16)]],
    const device float*   alibi        [[buffer(17)]],
    const constant bool&  has_alibi    [[buffer(18)]],
    const constant uint&  D            [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int Dpad = EPL * 32;
    constexpr int BK   = 32;
    constexpr int BQS  = (Dpad <= 64) ? 64 : (Dpad <= 128) ? 32 : (Dpad <= 256) ? 16 : 8;

    threadgroup float  Q_smem[BQS * Dpad];
    threadgroup float dO_smem[BQS * Dpad];

    const uint kv_h    = tgid.x;
    const uint b       = tgid.z;
    const uint lane    = tid % 32;
    const int epl = min(EPL, max(0, (int)D - (int)lane * EPL));
    const uint k_local = tid / 32;

    const uint q_start = cu_seqlens_q[b];
    const uint q_end   = cu_seqlens_q[b + 1];
    const uint k_start = cu_seqlens_k[b];
    const uint k_end   = cu_seqlens_k[b + 1];
    const uint qL = q_end - q_start;
    const uint kL = k_end - k_start;

    if (tgid.y * BK >= kL) return;

    const uint k_row = tgid.y * BK + k_local;
    const uint k_min = tgid.y * BK;
    const bool valid_k = (k_row < kL);

    const device T* K_ptr  = K  + kv_h * total_k * D + k_start * D;
    const device T* V_ptr  = V  + kv_h * total_k * D + k_start * D;
    device       T* dK_ptr = dK + kv_h * total_k * D + k_start * D;
    device       T* dV_ptr = dV + kv_h * total_k * D + k_start * D;

    float k_reg[EPL] = {};
    float v_reg[EPL] = {};
    if (valid_k) {
        for (int e = 0; e < epl; e++) {
            k_reg[e] = float(K_ptr[k_row * D + lane * EPL + e]);
            v_reg[e] = float(V_ptr[k_row * D + lane * EPL + e]);
        }
    }

    float dk_acc[EPL] = {};
    float dv_acc[EPL] = {};

    const uint tg_size = 32 * BK;

    for (uint g = 0; g < gqa; g++) {
        const uint q_head = kv_h * gqa + g;

        const device T* Q_ptr  = Q  + q_head * total_q * D + q_start * D;
        const device T* dO_ptr = dO + q_head * total_q * D + q_start * D;

        for (uint qb = 0; qb < qL; qb += BQS) {
            if (ic && qb + (uint)BQS - 1 < k_min) continue;
            if (wnd_right >= 0 && (int)(qb + (uint)BQS - 1) + wnd_right < (int)k_row) continue;
            if (wnd_left >= 0 && (int)qb > (int)k_row + wnd_left) continue;

            for (uint i = tid; i < (uint)(BQS * Dpad); i += tg_size) {
                uint r = qb + i / Dpad;
                uint d = i % Dpad;
                bool valid = (r < qL) && (d < (uint)D);
                Q_smem[i]  = valid ? float( Q_ptr[r * D + d]) : 0.0f;
                dO_smem[i] = valid ? float(dO_ptr[r * D + d]) : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint tile_end = min(qb + (uint)BQS, qL);

            for (uint q_row = qb; q_row < tile_end; ++q_row) {
                const bool in_causal = !ic || (k_row <= q_row);
                const bool in_wl = (wnd_left  < 0) || (int(k_row) + wnd_left  >= int(q_row));
                const bool in_wr = (wnd_right < 0) || (int(k_row) <= int(q_row) + wnd_right);
                if (!in_causal || !in_wl || !in_wr) continue;

                float lse_i   = LSE[q_head * total_q + q_start + q_row];
                float d_vec_i = Dv[q_head * total_q + q_start + q_row];
                int i = (int)(q_row - qb);

                float qk = 0.0f;
                for (int e = 0; e < epl; e++)
                    qk += Q_smem[i * Dpad + lane * EPL + e] * k_reg[e];

                float raw_score = simd_sum(qk) * sc;
                if (has_alibi) {
                    int rel = int(k_row) - int(q_row);
                    raw_score += alibi[q_head] * float(ic ? rel : -abs(rel));
                }
                float p_ij = metal::precise::exp(raw_score - lse_i);
                if (!valid_k) p_ij = 0.0f;

                float dov = 0.0f;
                for (int e = 0; e < epl; e++)
                    dov += dO_smem[i * Dpad + lane * EPL + e] * v_reg[e];
                float ds_ij = p_ij * (simd_sum(dov) - d_vec_i);

                for (int e = 0; e < epl; e++)
                    dv_acc[e] += p_ij * dO_smem[i * Dpad + lane * EPL + e];
                for (int e = 0; e < epl; e++)
                    dk_acc[e] += ds_ij * Q_smem[i * Dpad + lane * EPL + e];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } // end gqa loop

    if (!valid_k) return;
    for (int e = 0; e < epl; e++) {
        dK_ptr[k_row * D + lane * EPL + e] = T(dk_acc[e] * sc);
        dV_ptr[k_row * D + lane * EPL + e] = T(dv_acc[e]);
    }
}


// ── varlen forward SGMMA ─────────────────────────────────────────────────────
// Replaces flash_attn_varlen_fwd scalar inner loop with tile_matmad (SGMMA).
// Reuses BlockLoaderT/BaseMMAFrag/MMATile/tile_matmad from PrefillAttention.h.
// Layout: Q/K/V/O [H, total, D] packed; LSE [H, total_q] natural log-sum-exp.
// Grid: (ceil(max_seqlen_q/BQ), H, B)   TG: WM*WN*32 flat threads

template<typename T, int BQ, int BK, int BD, int WM, int WN, bool DO_CAUSAL>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void flash_attn_varlen_fwd_sgmma(
    const device T*       Q            [[buffer(0)]],
    const device T*       K            [[buffer(1)]],
    const device T*       V            [[buffer(2)]],
    device       T*       O            [[buffer(3)]],
    device       float*   LSE          [[buffer(4)]],
    const device uint*    cu_seqlens_q [[buffer(5)]],
    const device uint*    cu_seqlens_k [[buffer(6)]],
    const device float*               alibi  [[buffer(7)]],
    const constant VarlenAttnParams&  params [[buffer(8)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    const uint  total_q   = params.total_q;
    const uint  total_k   = params.total_k;
    const float sc        = params.sc;
    const uint  gqa       = params.gqa;
    const int   wnd_left  = params.wnd_left;
    const int   wnd_right = params.wnd_right;
    const bool  has_alibi = (bool)params.has_alibi;
    using AccumType = c10::metal::accum_t<T>;
    constexpr int kFragSize = 8;
    using MMAFrag_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

    const int block_q = (int)tgid.x;
    const int h       = (int)tgid.y;
    const int b       = (int)tgid.z;
    const int kv_h    = h / (int)gqa;

    const uint q_start = cu_seqlens_q[b];
    const uint k_start = cu_seqlens_k[b];
    const int  qL = (int)(cu_seqlens_q[b+1] - q_start);
    const int  kL = (int)(cu_seqlens_k[b+1] - k_start);

    if (block_q * BQ >= qL) return;

    Q += h    * (int)total_q * BD + ((int)q_start + block_q * BQ) * BD;
    K += kv_h * (int)total_k * BD + (int)k_start * BD;
    V += kv_h * (int)total_k * BD + (int)k_start * BD;
    O += h    * (int)total_q * BD + ((int)q_start + block_q * BQ) * BD;

    constexpr short padQ = 16 / sizeof(T);
    constexpr short padK = 16 / sizeof(T);
    constexpr short padV = 16 / sizeof(T);
    constexpr short LDQ  = BD + padQ;
    constexpr short LDKr = BD + padK;  // row-major K (== LDV; padK==padV)
    constexpr short LDV  = BD + padV;

    threadgroup T Q_smem[BQ * (BD + padQ)];
    threadgroup T KV_smem[BK * LDV];       // K+V share single buffer (sync load)
    threadgroup T* Ks = KV_smem;
    threadgroup T* Vs = KV_smem;

    using QLoader = BlockLoaderT<T, BQ, BD, LDQ, 1,   1, WM*WN*32>;
    using KLoader = BlockLoaderT<T, BK, BD, LDKr, 1,  0, WM*WN*32>;
    using VLoader = BlockLoaderT<T, BK, BD, LDV, 1,   0, WM*WN*32>;

    // Scale Q by sc/ln(2) so we can use exp2 in softmax
    TransformScale<T> ts(static_cast<T>(sc * 1.44269504089f));

    QLoader loader_q(Q, BD, Q_smem, simd_group_id, simd_lane_id);
    // K loaded via per-simdgroup simdgroup_async_copy; V constructed per-iteration

    constexpr int TQ = BQ / (WM * WN * kFragSize);
    constexpr int TK = BK / kFragSize;
    constexpr int TD = BD / kFragSize;
    static_assert(TQ == 1, "TQ must equal 1");

    MMATile<AccumType, TQ, 1,  MMAFrag_t> Qtile;
    MMATile<AccumType, 1,  TK, MMAFrag_t> Ktile;
    MMATile<AccumType, TQ, TK, MMAFrag_t> Stile;
    MMATile<AccumType, 1,  1,  MMAFrag_t> Vtile;
    MMATile<AccumType, TQ, TD, MMAFrag_t> Otile;
    Otile.clear();

    const short2 coord = MMAFrag_t::get_coord(simd_lane_id);
    const short sm = coord.y;  // row within 8x8 fragment
    const short sn = coord.x;  // col within 8x8 fragment (even: 0,2,4,6)
    const short tm = (short)(kFragSize * TQ * simd_group_id);

    const short Qs_off    = (tm + sm) * LDQ + sn;
    const short Ks_off    = sn * LDKr + sm;   // row-major K^T: kSrcStrCol=LDKr
    const short Vs_off    = sm * LDV + sn;
    constexpr short Qs_stride = kFragSize;
    constexpr short Ks_stride = kFragSize;     // d-dimension advance in row-major K

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int q_end  = min(block_q * BQ + BQ, qL);
    const int q_size = q_end - block_q * BQ;
    if (q_size < BQ) loader_q.load_safe(short2(BD, q_size));
    else             loader_q.load_unsafe();
    loader_q.apply_inplace_op(ts);

    // kRowsPT=1 because TQ=1, kElemRows=1: each thread owns 1 row's max/sum
    constexpr short kRowsPT = 1;
    AccumType max_score[kRowsPT], sum_score[kRowsPT] = {0};
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) max_score[i] = -INFINITY;

    int kb_lim = (kL + BK - 1) / BK;
    if constexpr (DO_CAUSAL)
        kb_lim = min(kb_lim, (block_q * BQ + q_size + BK - 1) / BK);
    if (wnd_right >= 0)
        kb_lim = min(kb_lim, (block_q * BQ + q_size + wnd_right + BK - 1) / BK);

    for (int kb = 0; kb < kb_lim; kb++) {
        // Sliding window: skip K blocks entirely before window
        if (wnd_left >= 0 && (kb + 1) * BK <= block_q * BQ - wnd_left)
            continue;

        Ks = KV_smem;
        Vs = KV_smem;

        const int k_end  = min(kb * BK + BK, kL);
        const int k_size = k_end - kb * BK;

        // Load K[kb] synchronously into shared memory
        {
            KLoader loader_k(K + kb * BK * BD, BD, KV_smem, simd_group_id, simd_lane_id);
            if (k_size < BK) loader_k.load_safe(short2(BD, k_size));
            else             loader_k.load_unsafe();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);


        Stile.clear();

        // Stile = Q * K^T via SGMMA (row-major K; K^T loaded with kSrcStrCol=LDKr)
        PREFILL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
            simdgroup_barrier(mem_flags::mem_none);
            Qtile.template load<T, 1, 1, LDQ, 1>(&Q_smem[Qs_off + dd * Qs_stride]);
            Ktile.template load<T, 1, 1, 1, LDKr>(&Ks[Ks_off    + dd * Ks_stride]);
            simdgroup_barrier(mem_flags::mem_none);
            tile_matmad(Stile, Qtile, Ktile, Stile);
        }

        // Mask partial K block
        if (k_size < BK) {
            PREFILL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Stile)::kTileRows; i++)
            PREFILL_PRAGMA_UNROLL
            for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
                short col = sn + j * kFragSize;
                PREFILL_PRAGMA_UNROLL
                for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++)
                    if ((col + jj) >= k_size) Stile.frag_at(i,j)[jj] = -INFINITY;
            }
        }

        // Causal mask
        if constexpr (DO_CAUSAL) {
            PREFILL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
                int row = block_q * BQ + (int)(tm + sm) + i * kFragSize;
                PREFILL_PRAGMA_UNROLL
                for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
                    int col = kb * BK + (int)sn + j * kFragSize;
                    PREFILL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++)
                        if (row < (col + jj)) Stile.frag_at(i,j)[jj] = -INFINITY;
                }
            }
        }

        // Sliding window mask
        if (wnd_left >= 0 || wnd_right >= 0) {
            PREFILL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
                int row = block_q * BQ + (int)(tm + sm) + i * kFragSize;
                PREFILL_PRAGMA_UNROLL
                for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
                    int col_base = kb * BK + (int)sn + j * kFragSize;
                    PREFILL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                        int col = col_base + jj;
                        bool ok = (wnd_left  < 0 || col + wnd_left  >= row)
                               && (wnd_right < 0 || col             <= row + wnd_right);
                        if (!ok) Stile.frag_at(i,j)[jj] = -INFINITY;
                    }
                }
            }
        }

        // ALiBi (bias in log2 space to match exp2 softmax)
        if (has_alibi) {
            AccumType slope_log2 = AccumType(alibi[h] * 1.4426950408889634f);
            PREFILL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
                int row = block_q * BQ + (int)(tm + sm) + i * kFragSize;
                PREFILL_PRAGMA_UNROLL
                for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
                    int col_base = kb * BK + (int)sn + j * kFragSize;
                    PREFILL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
                        int rel = (col_base + jj) - row;
                        Stile.frag_at(i,j)[jj] += slope_log2 * AccumType(DO_CAUSAL ? rel : -abs(rel));
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            VLoader loader_v(V + kb * BK * BD, BD, KV_smem, simd_group_id, simd_lane_id);
            if (k_size < BK) loader_v.load_safe(short2(BD, k_size));
            else             loader_v.load_unsafe();
        }

        // Online softmax (in log2 space)
        AccumType new_max[kRowsPT], factor[kRowsPT];
        PREFILL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];

        Stile.template row_reduce<MaxOp>(new_max);
        Stile.template row_bin_op<ExpSubOp>(new_max);

        PREFILL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i]    = (new_max[i] == -INFINITY)
                         ? AccumType(1) : fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
        }

        AccumType sum_tmp[kRowsPT] = {0};
        Stile.template row_reduce<SumOp>(sum_tmp);
        PREFILL_PRAGMA_UNROLL
        for (short i = 0; i < kRowsPT; ++i)
            sum_score[i] = sum_score[i] * factor[i] + sum_tmp[i];

        Otile.template row_bin_op<MulOp>(factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += softmax(S) * V  via SGMMA
        PREFILL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++)
        PREFILL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++)
        PREFILL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
            Vtile.template load<T, 1, 1, LDV, 1>(
                &Vs[Vs_off + (ik * kFragSize) * LDV + id * kFragSize]);
            if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
            MMAFrag_t::mma(Otile.frag_at(iq,id),
                           Stile.frag_at(iq,ik),
                           Vtile.frag_at(0,0),
                           Otile.frag_at(iq,id));
        }

    }  // end K-block loop

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i)
        if (max_score[i] == -INFINITY) sum_score[i] = AccumType(1);

    Otile.template row_bin_op<DivOp>(sum_score);
    threadgroup_barrier(mem_flags::mem_none);

    device T* O_tile = O + (int)(tm + sm) * BD + sn;
    if (q_size < BQ) {
        if ((tm + sm) < q_size && sn < BD)
            Otile.template store_safe<T, 1, 1>(O_tile, BD,
                short2(BD - sn, q_size - (tm + sm)));
    } else {
        Otile.template store<T, 1, 1>(O_tile, BD);
    }

    // Write LSE per row (lane with sn==0 only); convert log2 -> natural lse
    // natural_lse = max_log2 * ln(2) + log(sum_exp2)
    if (sn == 0) {
        int local_row  = (int)(tm + sm);
        int global_row = (int)q_start + block_q * BQ + local_row;
        if (local_row < q_size && global_row < (int)total_q) {
            float nat_lse = (max_score[0] == -INFINITY)
                ? -INFINITY
                : max_score[0] * 0.6931471805599453f
                  + metal::precise::log(sum_score[0]);
            LSE[h * (int)total_q + global_row] = nat_lse;
        }
    }
}

#define INST_VARLEN_SGMMA_FWD(T, BQ, BK, BD, WM, WN) \
  template [[host_name("flash_attn_varlen_fwd_sgmma_" #T \
    "_bq" #BQ "_bk" #BK "_bd" #BD "_wm" #WM "_wn" #WN "_causal0")]] \
  [[kernel]] void flash_attn_varlen_fwd_sgmma<T,BQ,BK,BD,WM,WN,false>( \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], \
      const device T* V [[buffer(2)]], device T* O [[buffer(3)]], \
      device float* LSE [[buffer(4)]], const device uint* cu_seqlens_q [[buffer(5)]], \
      const device uint* cu_seqlens_k [[buffer(6)]], \
      const device float* alibi [[buffer(7)]], \
      const constant VarlenAttnParams& params [[buffer(8)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]); \
  template [[host_name("flash_attn_varlen_fwd_sgmma_" #T \
    "_bq" #BQ "_bk" #BK "_bd" #BD "_wm" #WM "_wn" #WN "_causal1")]] \
  [[kernel]] void flash_attn_varlen_fwd_sgmma<T,BQ,BK,BD,WM,WN,true>( \
      const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], \
      const device T* V [[buffer(2)]], device T* O [[buffer(3)]], \
      device float* LSE [[buffer(4)]], const device uint* cu_seqlens_q [[buffer(5)]], \
      const device uint* cu_seqlens_k [[buffer(6)]], \
      const device float* alibi [[buffer(7)]], \
      const constant VarlenAttnParams& params [[buffer(8)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]);

// D=64:  BQ=32 BK=32 BD=64  WM=4 WN=1 -> 128 threads/TG, TQ=1 TK=4 TD=8
// D=128: BQ=32 BK=16 BD=128 WM=4 WN=1 -> 128 threads/TG, TQ=1 TK=2 TD=16
INST_VARLEN_SGMMA_FWD(half,   32, 32, 64,  4, 1)
INST_VARLEN_SGMMA_FWD(bfloat, 32, 32, 64,  4, 1)
INST_VARLEN_SGMMA_FWD(float,  32, 32, 64,  4, 1)
INST_VARLEN_SGMMA_FWD(half,   32, 16, 128, 4, 1)
INST_VARLEN_SGMMA_FWD(bfloat, 32, 16, 128, 4, 1)
INST_VARLEN_SGMMA_FWD(float,  32, 16, 128, 4, 1)


// ── varlen explicit instantiation ─────────────────────────────────────────────

#define INSTANTIATE_FLASH_VARLEN_FWD(T, E) \
  template [[host_name("flash_attn_varlen_fwd_" #T "_epl" #E)]] [[kernel]] \
  void flash_attn_varlen_fwd<T, E>( \
      const device T*       Q            [[buffer(0)]],   \
      const device T*       K            [[buffer(1)]],   \
      const device T*       V            [[buffer(2)]],   \
      device       T*       O            [[buffer(3)]],   \
      device       float*   LSE          [[buffer(4)]],   \
      const device uint*    cu_seqlens_q [[buffer(5)]],   \
      const device uint*    cu_seqlens_k [[buffer(6)]],   \
      const constant uint&  total_q      [[buffer(7)]],   \
      const constant uint&  total_k      [[buffer(8)]],   \
      const constant float& sc           [[buffer(9)]],   \
      const constant bool&  ic           [[buffer(10)]],  \
      const constant uint&  gqa          [[buffer(11)]],  \
      const constant int&   wnd_left     [[buffer(12)]],  \
      const constant int&   wnd_right    [[buffer(13)]],  \
      const device float*   alibi        [[buffer(14)]],  \
      const constant bool&  has_alibi    [[buffer(15)]],  \
      const constant uint&  D            [[buffer(16)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],        \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, E) \
  template [[host_name("flash_attn_varlen_bwd_pre_" #T "_epl" #E)]] [[kernel]] \
  void flash_attn_varlen_bwd_preprocess<T, E>( \
      const device T*       dO           [[buffer(0)]],  \
      const device T*       O            [[buffer(1)]],  \
      device       float*   Dv           [[buffer(2)]],  \
      const device uint*    cu_seqlens_q [[buffer(3)]],  \
      const constant uint&  total_q      [[buffer(4)]],  \
      const constant uint&  D            [[buffer(5)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],       \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, E) \
  template [[host_name("flash_attn_varlen_bwd_dq_" #T "_epl" #E)]] [[kernel]] \
  void flash_attn_varlen_bwd_dq<T, E>( \
      const device T*       Q            [[buffer(0)]],   \
      const device T*       K            [[buffer(1)]],   \
      const device T*       V            [[buffer(2)]],   \
      const device T*       dO           [[buffer(3)]],   \
      const device float*   LSE          [[buffer(4)]],   \
      const device float*   Dv           [[buffer(5)]],   \
      device       T*       dQ           [[buffer(6)]],   \
      const device uint*    cu_seqlens_q [[buffer(7)]],   \
      const device uint*    cu_seqlens_k [[buffer(8)]],   \
      const constant uint&  total_q      [[buffer(9)]],   \
      const constant uint&  total_k      [[buffer(10)]],  \
      const constant float& sc           [[buffer(11)]],  \
      const constant bool&  ic           [[buffer(12)]],  \
      const constant uint&  gqa          [[buffer(13)]],  \
      const constant int&   wnd_left     [[buffer(14)]],  \
      const constant int&   wnd_right    [[buffer(15)]],  \
      const device float*   alibi        [[buffer(16)]],  \
      const constant bool&  has_alibi    [[buffer(17)]],  \
      const constant uint&  D            [[buffer(18)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],        \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, E) \
  template [[host_name("flash_attn_varlen_bwd_dkdv_" #T "_epl" #E)]] [[kernel]] \
  void flash_attn_varlen_bwd_dkdv<T, E>( \
      const device T*       Q            [[buffer(0)]],   \
      const device T*       K            [[buffer(1)]],   \
      const device T*       V            [[buffer(2)]],   \
      const device T*       dO           [[buffer(3)]],   \
      const device float*   LSE          [[buffer(4)]],   \
      const device float*   Dv           [[buffer(5)]],   \
      device       T*       dK           [[buffer(6)]],   \
      device       T*       dV           [[buffer(7)]],   \
      const device uint*    cu_seqlens_q [[buffer(8)]],   \
      const device uint*    cu_seqlens_k [[buffer(9)]],   \
      const constant uint&  total_q      [[buffer(10)]],  \
      const constant uint&  total_k      [[buffer(11)]],  \
      const constant float& sc           [[buffer(12)]],  \
      const constant bool&  ic           [[buffer(13)]],  \
      const constant uint&  gqa          [[buffer(14)]],  \
      const constant int&   wnd_left     [[buffer(15)]],  \
      const constant int&   wnd_right    [[buffer(16)]],  \
      const device float*   alibi        [[buffer(17)]],  \
      const constant bool&  has_alibi    [[buffer(18)]],  \
      const constant uint&  D            [[buffer(19)]],  \
      uint3 tgid [[threadgroup_position_in_grid]],        \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INSTANTIATE_FLASH_VARLEN_EPL(T, E) \
  INSTANTIATE_FLASH_VARLEN_FWD(T, E)      \
  INSTANTIATE_FLASH_VARLEN_BWD_PRE(T, E)  \
  INSTANTIATE_FLASH_VARLEN_BWD_DQ(T, E)   \
  INSTANTIATE_FLASH_VARLEN_BWD_DKDV(T, E)

#define INSTANTIATE_FLASH_VARLEN_ALL(T) \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 1)   \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 2)   \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 3)   \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 4)   \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 5)   \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 6)   \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 7)   \
  INSTANTIATE_FLASH_VARLEN_EPL(T, 8)

INSTANTIATE_FLASH_VARLEN_ALL(float)
INSTANTIATE_FLASH_VARLEN_ALL(half)
INSTANTIATE_FLASH_VARLEN_ALL(bfloat)
