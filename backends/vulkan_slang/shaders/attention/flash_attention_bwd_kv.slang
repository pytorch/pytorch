// Flash Attention backward: compute grad_K and grad_V
// One workgroup per (b, kv_h, k) — iterates over all query positions.
//
// grad_V[k,d] = sum_q P[q,k] * grad_out[q,d]
// grad_K[k,d] = scale * sum_q P[q,k] * (dot(go[q], V[k]) - D_i[q]) * Q[q,d]
//   where D_i[q] = dot(go[q], fwd_out[q])  (computed inline per q)
//
// Dispatch: B * KV_H * S workgroups
// q_seq_major=1: Q/K/V in [B,S,H,D]/[B,S,KV_H,D] layout (avoids contiguous copy)
// go_seq_major=1: grad_out + fwd_out in [B,N,H,D] seq-major layout.

struct Params {
    uint B;
    uint H;
    uint KV_H;
    uint N;
    uint S;
    uint D;
    float scale;
    uint is_causal;
    uint q_seq_major;  // Q/K/V + grad_K/grad_V layout
    uint go_seq_major; // grad_out layout
};

[[vk::binding(0)]] StructuredBuffer<float> grad_out; // [B,H,N,D] head-major OR [B,N,H,D] seq-major
[[vk::binding(1)]] StructuredBuffer<float> Q;        // [B, H, N, D] or [B, N, H, D]
[[vk::binding(2)]] StructuredBuffer<float> K;        // [B, KV_H, S, D] or [B, S, KV_H, D]
[[vk::binding(3)]] StructuredBuffer<float> V;        // [B, KV_H, S, D] or [B, S, KV_H, D]
[[vk::binding(4)]] StructuredBuffer<float> lse;      // [B, H, N]
[[vk::binding(5)]] StructuredBuffer<float> fwd_out;  // [B, H, N, D] OR [B, N, H, D] (same layout as grad_out)
[[vk::binding(6)]] RWStructuredBuffer<float> grad_K; // [B, KV_H, S, D] seq-major or head-major
[[vk::binding(7)]] RWStructuredBuffer<float> grad_V; // [B, KV_H, S, D] seq-major or head-major
[[vk::push_constant]] Params params;

// smem[0]: reduced scalar (dot product or go_dot_v)
// smem[1]: p_k broadcasted to all threads
// smem[2]: ds_k broadcasted to all threads
// smem[3..258]: reduction workspace (256 elements starting at smem[3])
groupshared float smem[259];

// Parallel reduction helper: reduces smem[off..off+255] into smem[off]
void reduce_sum(uint tid, uint off) {
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[off + tid] += smem[off + tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint flat = gid.x;
    uint k     = flat % params.S;
    uint kv_h  = (flat / params.S) % params.KV_H;
    uint b     = flat / (params.S * params.KV_H);
    if (b >= params.B) return;

    uint tid   = lid.x;
    uint n_rep = params.H / params.KV_H;

    // K/V base for position k — head-major or seq-major
    uint k_base;
    if (params.q_seq_major != 0u) {
        k_base = b * params.S * params.KV_H * params.D + k * params.KV_H * params.D + kv_h * params.D;
    } else {
        k_base = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D + k * params.D;
    }

    float gK = 0.0f;
    float gV = 0.0f;

    for (uint rep = 0; rep < n_rep; rep++) {
        uint h = kv_h * n_rep + rep;

        for (uint q_pos = 0; q_pos < params.N; q_pos++) {
            uint masked = (params.is_causal != 0 && k > q_pos) ? 1u : 0u;

            // grad_out: head-major when go_seq_major=0, seq-major [B,N,H,D] when go_seq_major=1
            uint go_base;
            if (params.go_seq_major != 0u) {
                go_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
            } else {
                go_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
            }
            uint lse_idx = b * params.H * params.N + h * params.N + q_pos;

            // Q base: head-major or seq-major
            uint q_base;
            if (params.q_seq_major != 0u) {
                q_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
            } else {
                q_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
            }

            // --- Compute score = dot(Q[q], K[k]) * scale ---
            float dot_qk = 0.0f;
            for (uint d = tid; d < params.D; d += 256) {
                dot_qk += Q[q_base + d] * K[k_base + d];
            }
            smem[3 + tid] = dot_qk;
            GroupMemoryBarrierWithGroupSync();
            reduce_sum(tid, 3);
            // smem[3] = full dot product, visible to all after reduce_sum's final barrier

            // Thread 0 computes p_k and broadcasts it
            if (tid == 0) {
                float score = smem[3] * params.scale;
                float lse_val = lse[lse_idx];
                smem[1] = masked == 0u ? exp(score - lse_val) : 0.0f;
            }
            GroupMemoryBarrierWithGroupSync();
            float p_k = smem[1]; // all threads read same p_k

            // --- Compute go_dot_v = dot(go[q], V[k]) and go_dot_out = dot(go[q], out[q]) ---
            float go_dot_v = 0.0f;
            float go_dot_out = 0.0f;
            for (uint d = tid; d < params.D; d += 256) {
                float go_val = grad_out[go_base + d];
                go_dot_v   += go_val * V[k_base + d];
                go_dot_out += go_val * fwd_out[go_base + d];
            }
            smem[3 + tid] = go_dot_v;
            GroupMemoryBarrierWithGroupSync();
            reduce_sum(tid, 3);
            float go_dot_v_total = smem[3]; // after reduce_sum's final barrier
            smem[3 + tid] = go_dot_out;
            GroupMemoryBarrierWithGroupSync();
            reduce_sum(tid, 3);
            // smem[3] = di_val = dot(go[q], out[q])

            // Thread 0 computes ds_k and broadcasts it
            if (tid == 0) {
                smem[2] = go_dot_v_total - smem[3]; // go_dot_v - di_val
            }
            GroupMemoryBarrierWithGroupSync();
            float ds_k = smem[2];

            // --- Accumulate ---
            if (tid < params.D) {
                gV += p_k * grad_out[go_base + tid];
                gK += p_k * params.scale * ds_k * Q[q_base + tid];
            }
        }
    }

    // Write grad_K/grad_V: head-major [B,KV_H,S,D] when q_seq_major=0,
    //   seq-major [B,S,KV_H,D] when q_seq_major=1 (matches input K/V storage).
    uint kv_out_base;
    if (params.q_seq_major != 0u) {
        // seq-major [B, S, KV_H, D]: index = b*S*KV_H*D + k*KV_H*D + kv_h*D
        kv_out_base = b * params.S * params.KV_H * params.D + k * params.KV_H * params.D + kv_h * params.D;
    } else {
        // head-major [B, KV_H, S, D]
        kv_out_base = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D + k * params.D;
    }
    if (tid < params.D) {
        grad_K[kv_out_base + tid] = gK;
        grad_V[kv_out_base + tid] = gV;
    }
}
