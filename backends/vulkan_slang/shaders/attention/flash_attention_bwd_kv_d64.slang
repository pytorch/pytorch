// Flash Attention backward: compute grad_K and grad_V — 64-thread variant for D <= 64.
// Uses 64 threads with 5-step shared-memory reduction (vs 256 threads / 8 steps).
// One workgroup per (b, kv_h, k) — iterates over all query positions.
//
// grad_V[k,d] = sum_q P[q,k] * grad_out[q,d]
// grad_K[k,d] = scale * sum_q P[q,k] * (dot(go[q], V[k]) - D_i[q]) * Q[q,d]
//   where D_i[q] = dot(go[q], fwd_out[q])  (computed inline per q)

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
[[vk::binding(6)]] RWStructuredBuffer<float> grad_K;
[[vk::binding(7)]] RWStructuredBuffer<float> grad_V;
[[vk::push_constant]] Params params;

// smem[0]: (unused), smem[1]: p_k, smem[2]: ds_k, smem[3..66]: reduction workspace
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
    uint flat = gid.x;
    uint k     = flat % params.S;
    uint kv_h  = (flat / params.S) % params.KV_H;
    uint b     = flat / (params.S * params.KV_H);
    if (b >= params.B) return;

    uint tid   = lid.x;
    uint n_rep = params.H / params.KV_H;

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

            uint go_base;
            if (params.go_seq_major != 0u) {
                go_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
            } else {
                go_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
            }
            uint lse_idx = b * params.H * params.N + h * params.N + q_pos;

            uint q_base;
            if (params.q_seq_major != 0u) {
                q_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
            } else {
                q_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
            }

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
                float lse_val = lse[lse_idx];
                smem[1] = masked == 0u ? exp(score - lse_val) : 0.0f;
            }
            GroupMemoryBarrierWithGroupSync();
            float p_k = smem[1];

            // go_dot_v and di_val in two separate reductions
            float go_dot_v = 0.0f;
            float go_dot_out = 0.0f;
            for (uint d = tid; d < params.D; d += THREADS) {
                float go_val = grad_out[go_base + d];
                go_dot_v   += go_val * V[k_base + d];
                go_dot_out += go_val * fwd_out[go_base + d];
            }
            smem[3 + tid] = go_dot_v;
            GroupMemoryBarrierWithGroupSync();
            reduce_sum(tid, 3);
            float go_dot_v_total = smem[3];
            smem[3 + tid] = go_dot_out;
            GroupMemoryBarrierWithGroupSync();
            reduce_sum(tid, 3);

            if (tid == 0) {
                smem[2] = go_dot_v_total - smem[3];
            }
            GroupMemoryBarrierWithGroupSync();
            float ds_k = smem[2];

            if (tid < params.D) {
                gV += p_k * grad_out[go_base + tid];
                gK += p_k * params.scale * ds_k * Q[q_base + tid];
            }
        }
    }

    uint kv_out_base;
    if (params.q_seq_major != 0u) {
        kv_out_base = b * params.S * params.KV_H * params.D + k * params.KV_H * params.D + kv_h * params.D;
    } else {
        kv_out_base = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D + k * params.D;
    }
    if (tid < params.D) {
        grad_K[kv_out_base + tid] = gK;
        grad_V[kv_out_base + tid] = gV;
    }
}
