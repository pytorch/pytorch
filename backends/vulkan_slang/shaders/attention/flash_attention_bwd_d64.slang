// Flash Attention backward: compute grad_Q — 64-thread variant for D <= 64.
// Uses 64 threads with 5-step shared-memory reduction (vs 256 threads / 8 steps).
// One workgroup per (b, h, q) — iterates over all key positions.
//
// grad_Q[q,d] = scale * sum_k P[q,k] * (dot(go[q], V[k]) - D_i[q]) * K[k,d]
//   where P[q,k] = exp(score[q,k] - lse[q])
//         D_i[q] = dot(grad_out[q], fwd_out[q])  (computed inline)

static const uint THREADS = 64;

struct Params {
    uint B;
    uint H;
    uint KV_H;
    uint N;
    uint S;
    uint D;
    float scale;
    uint is_causal;
    uint q_seq_major;
    uint go_seq_major;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_out;
[[vk::binding(1)]] StructuredBuffer<float> Q;
[[vk::binding(2)]] StructuredBuffer<float> K;
[[vk::binding(3)]] StructuredBuffer<float> V;
[[vk::binding(4)]] StructuredBuffer<float> lse;
[[vk::binding(5)]] StructuredBuffer<float> fwd_out;
[[vk::binding(6)]] RWStructuredBuffer<float> grad_Q;
[[vk::push_constant]] Params params;

// smem[0]: broadcast result, smem[1]: p_k, smem[2]: ds_k, smem[3..66]: reduction workspace
groupshared float smem[3 + THREADS];

void reduce_sum(uint tid, uint off) {
    for (uint s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem[off + tid] += smem[off + tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint flat  = gid.x;
    uint q_pos = flat % params.N;
    uint h     = (flat / params.N) % params.H;
    uint b     = flat / (params.N * params.H);
    if (b >= params.B) return;

    uint tid  = lid.x;
    uint kv_h = h / (params.H / params.KV_H);

    uint go_base;
    if (params.go_seq_major != 0u) {
        go_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        go_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
    }

    uint q_base, k_base0;
    if (params.q_seq_major != 0u) {
        q_base  = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
        k_base0 = b * params.S * params.KV_H * params.D + kv_h * params.D;
    } else {
        q_base  = b * params.H    * params.N * params.D + h    * params.N * params.D + q_pos * params.D;
        k_base0 = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D;
    }

    float lse_val = lse[b * params.H * params.N + h * params.N + q_pos];

    // D_i = dot(go[q], fwd_out[q]) via shared reduction
    float di_partial = 0.0f;
    for (uint d = tid; d < params.D; d += THREADS) {
        di_partial += grad_out[go_base + d] * fwd_out[go_base + d];
    }
    smem[3 + tid] = di_partial;
    GroupMemoryBarrierWithGroupSync();
    reduce_sum(tid, 3);
    float di_val = smem[3];

    float gQ = 0.0f;

    for (uint k = 0; k < params.S; k++) {
        uint masked = (params.is_causal != 0 && k > q_pos) ? 1u : 0u;
        uint k_base = params.q_seq_major != 0u
            ? k_base0 + k * params.KV_H * params.D
            : k_base0 + k * params.D;

        // score = dot(Q[q], K[k]) * scale
        float dot_qk = 0.0f;
        for (uint d = tid; d < params.D; d += THREADS) {
            dot_qk += Q[q_base + d] * K[k_base + d];
        }
        smem[3 + tid] = dot_qk;
        GroupMemoryBarrierWithGroupSync();
        reduce_sum(tid, 3);

        if (tid == 0) {
            float score = smem[3] * params.scale;
            smem[1] = masked == 0u ? exp(score - lse_val) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        float p_k = smem[1];

        // go_dot_v = dot(go[q], V[k])
        float go_dot_v = 0.0f;
        for (uint d = tid; d < params.D; d += THREADS) {
            go_dot_v += grad_out[go_base + d] * V[k_base + d];
        }
        smem[3 + tid] = go_dot_v;
        GroupMemoryBarrierWithGroupSync();
        reduce_sum(tid, 3);

        if (tid == 0) {
            smem[2] = smem[3] - di_val;
        }
        GroupMemoryBarrierWithGroupSync();
        float ds_k = smem[2];

        if (tid < params.D) {
            gQ += p_k * params.scale * ds_k * K[k_base + tid];
        }
    }

    if (tid < params.D) {
        uint gq_base;
        if (params.q_seq_major != 0u) {
            gq_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
        } else {
            gq_base = go_base;
        }
        grad_Q[gq_base + tid] = gQ;
    }
}
