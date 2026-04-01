// Flash Attention backward: compute grad_K and grad_V — wave-intrinsic variant for D <= 32.
// Uses WaveActiveSum for all dot-product reductions (zero barrier syncs per (q,k) pair).
// One workgroup per (b, kv_h, k_pos) — iterates over all (rep, q) pairs.
//
// grad_V[k,d] = sum_q P[q,k] * grad_out[q,d]
// grad_K[k,d] = scale * sum_q P[q,k] * (dot(go[q], V[k]) - D_i[q]) * Q[q,d]
//   where D_i[q] = dot(go[q], fwd_out[q])  (computed inline)

static const uint THREADS = 32;

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

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint flat  = gid.x;
    uint k_pos = flat % params.S;
    uint kv_h  = (flat / params.S) % params.KV_H;
    uint b     = flat / (params.S * params.KV_H);
    if (b >= params.B) return;

    uint tid   = lid.x;
    uint n_rep = params.H / params.KV_H;

    // K/V base for position k_pos
    uint k_base;
    if (params.q_seq_major != 0u) {
        k_base = b * params.S * params.KV_H * params.D + k_pos * params.KV_H * params.D + kv_h * params.D;
    } else {
        k_base = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D + k_pos * params.D;
    }

    float k_val = (tid < params.D) ? K[k_base + tid] : 0.0f;
    float v_val = (tid < params.D) ? V[k_base + tid] : 0.0f;

    float gK = 0.0f;
    float gV = 0.0f;

    for (uint rep = 0; rep < n_rep; rep++) {
        uint h = kv_h * n_rep + rep;

        for (uint q_pos = 0; q_pos < params.N; q_pos++) {
            if (params.is_causal != 0 && k_pos > q_pos) continue;

            uint go_base;
            if (params.go_seq_major != 0u) {
                go_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
            } else {
                go_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
            }

            uint q_base;
            if (params.q_seq_major != 0u) {
                q_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
            } else {
                q_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
            }

            float lse_val = lse[b * params.H * params.N + h * params.N + q_pos];

            float q_val  = (tid < params.D) ? Q[q_base + tid]        : 0.0f;
            float go_val = (tid < params.D) ? grad_out[go_base + tid] : 0.0f;
            float fo_val = (tid < params.D) ? fwd_out[go_base + tid]  : 0.0f;

            // All reductions via WaveActiveSum (zero barriers)
            float score      = WaveActiveSum(q_val * k_val) * params.scale;
            float go_dot_v   = WaveActiveSum(go_val * v_val);
            float di         = WaveActiveSum(go_val * fo_val);

            float p_k  = exp(score - lse_val);
            float ds_k = go_dot_v - di;

            if (tid < params.D) {
                gV += p_k * go_val;
                gK += p_k * params.scale * ds_k * q_val;
            }
        }
    }

    // Write grad_K/grad_V
    uint kv_out_base;
    if (params.q_seq_major != 0u) {
        kv_out_base = b * params.S * params.KV_H * params.D + k_pos * params.KV_H * params.D + kv_h * params.D;
    } else {
        kv_out_base = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D + k_pos * params.D;
    }
    if (tid < params.D) {
        grad_K[kv_out_base + tid] = gK;
        grad_V[kv_out_base + tid] = gV;
    }
}
