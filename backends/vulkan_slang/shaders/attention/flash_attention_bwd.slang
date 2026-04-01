// Flash Attention backward: compute grad_Q
// One workgroup per (b, h, q) — iterates over all key positions.
//
// grad_Q[q,d] = scale * sum_k P[q,k] * (dot(go[q], V[k]) - D_i[q]) * K[k,d]
//   where P[q,k] = exp(score[q,k] - lse[q])
//         D_i[q] = dot(grad_out[q], fwd_out[q])  (computed inline — no separate di pass)
//
// Dispatch: B * H * N workgroups
// q_seq_major=1: Q/K/V in [B,S,H,D]/[B,S,KV_H,D] layout; grad_Q in [B,N,H,D] seq-major.
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
    uint q_seq_major;  // Q/K/V + grad_Q layout
    uint go_seq_major; // grad_out layout
};

[[vk::binding(0)]] StructuredBuffer<float> grad_out; // [B,H,N,D] head-major OR [B,N,H,D] seq-major
[[vk::binding(1)]] StructuredBuffer<float> Q;        // [B, H, N, D] or [B, N, H, D]
[[vk::binding(2)]] StructuredBuffer<float> K;        // [B, KV_H, S, D] or [B, S, KV_H, D]
[[vk::binding(3)]] StructuredBuffer<float> V;        // [B, KV_H, S, D] or [B, S, KV_H, D]
[[vk::binding(4)]] StructuredBuffer<float> lse;      // [B, H, N]
[[vk::binding(5)]] StructuredBuffer<float> fwd_out;  // [B, H, N, D] OR [B, N, H, D] (same layout as grad_out)
[[vk::binding(6)]] RWStructuredBuffer<float> grad_Q; // [B, H, N, D] head-major OR [B, N, H, D] seq-major
[[vk::push_constant]] Params params;

// smem layout:
//   smem[0]: reduction result (scalar broadcast)
//   smem[1]: p_k broadcasted
//   smem[2]: ds_k broadcasted
//   smem[3..258]: 256-element reduction workspace
groupshared float smem[259];

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
    uint flat  = gid.x;
    uint q_pos = flat % params.N;
    uint h     = (flat / params.N) % params.H;
    uint b     = flat / (params.N * params.H);
    if (b >= params.B) return;

    uint tid  = lid.x;
    uint kv_h = h / (params.H / params.KV_H);

    // grad_out: head-major when go_seq_major=0, seq-major [B,N,H,D] storage when go_seq_major=1
    uint go_base;
    if (params.go_seq_major != 0u) {
        go_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        go_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
    }

    // Q/K/V: head-major or seq-major based on q_seq_major flag
    uint q_base, k_base0;
    if (params.q_seq_major != 0u) {
        // [B, S, H, D]: index = b*S*H*D + s*H*D + h*D
        q_base  = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
        k_base0 = b * params.S * params.KV_H * params.D + kv_h * params.D; // + k*KV_H*D
    } else {
        // [B, H, N, D]: index = b*H*N*D + h*N*D + q*D
        q_base  = b * params.H    * params.N * params.D + h    * params.N * params.D + q_pos * params.D;
        k_base0 = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D;
    }

    uint lse_idx = b * params.H * params.N + h * params.N + q_pos;
    float lse_val = lse[lse_idx];

    // Compute di_val = dot(go[q], fwd_out[q]) inline — same base index as go_base (same layout)
    float di_partial = 0.0f;
    for (uint d = tid; d < params.D; d += 256) {
        di_partial += grad_out[go_base + d] * fwd_out[go_base + d];
    }
    smem[3 + tid] = di_partial;
    GroupMemoryBarrierWithGroupSync();
    reduce_sum(tid, 3);
    float di_val = smem[3]; // all threads read after reduce_sum's final barrier

    float gQ = 0.0f;

    for (uint k = 0; k < params.S; k++) {
        uint masked = (params.is_causal != 0 && k > q_pos) ? 1u : 0u;
        uint k_base = params.q_seq_major != 0u
            ? k_base0 + k * params.KV_H * params.D
            : k_base0 + k * params.D;

        // --- Compute score = dot(Q[q], K[k]) * scale ---
        float dot_qk = 0.0f;
        for (uint d = tid; d < params.D; d += 256) {
            dot_qk += Q[q_base + d] * K[k_base + d];
        }
        smem[3 + tid] = dot_qk;
        GroupMemoryBarrierWithGroupSync();
        reduce_sum(tid, 3);

        // Thread 0 computes and broadcasts p_k
        if (tid == 0) {
            float score = smem[3] * params.scale;
            smem[1] = masked == 0u ? exp(score - lse_val) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        float p_k = smem[1];

        // --- Compute go_dot_v = dot(go[q], V[k]) ---
        float go_dot_v = 0.0f;
        for (uint d = tid; d < params.D; d += 256) {
            go_dot_v += grad_out[go_base + d] * V[k_base + d];
        }
        smem[3 + tid] = go_dot_v;
        GroupMemoryBarrierWithGroupSync();
        reduce_sum(tid, 3);

        // Thread 0 computes and broadcasts ds_k
        if (tid == 0) {
            smem[2] = smem[3] - di_val;
        }
        GroupMemoryBarrierWithGroupSync();
        float ds_k = smem[2];

        // --- Accumulate grad_Q ---
        if (tid < params.D) {
            gQ += p_k * params.scale * ds_k * K[k_base + tid];
        }
    }

    // Write grad_Q: head-major [B,H,N,D] when q_seq_major=0, seq-major [B,N,H,D] when q_seq_major=1
    if (tid < params.D) {
        uint gq_base;
        if (params.q_seq_major != 0u) {
            // seq-major [B, N, H, D]: index = b*N*H*D + q_pos*H*D + h*D
            gq_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
        } else {
            // head-major [B, H, N, D] = same as go_base
            gq_base = go_base;
        }
        grad_Q[gq_base + tid] = gQ;
    }
}
