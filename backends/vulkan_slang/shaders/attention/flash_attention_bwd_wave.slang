// Flash Attention backward: compute grad_Q — wave-intrinsic variant for D <= 32.
// Uses WaveActiveSum for all dot-product reductions (zero barriers for score/di computation).
// One workgroup per (b, h, q) — iterates over all key positions.
//
// grad_Q[q,d] = scale * sum_k P[q,k] * (dot(go[q], V[k]) - D_i[q]) * K[k,d]
//   where P[q,k] = exp(score[q,k] - lse[q])
//         D_i[q] = dot(grad_out[q], fwd_out[q])  (computed inline)

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
[[vk::binding(6)]] RWStructuredBuffer<float> grad_Q;
[[vk::push_constant]] Params params;

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

    uint go_base, q_base, k_base0, v_base0, gq_base;
    if (params.go_seq_major != 0u) {
        go_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        go_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
    }

    if (params.q_seq_major != 0u) {
        q_base  = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
        k_base0 = b * params.S * params.KV_H * params.D + kv_h * params.D;
        v_base0 = b * params.S * params.KV_H * params.D + kv_h * params.D;
        gq_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        q_base  = b * params.H    * params.N * params.D + h    * params.N * params.D + q_pos * params.D;
        k_base0 = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D;
        v_base0 = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D;
        gq_base = b * params.H    * params.N * params.D + h    * params.N * params.D + q_pos * params.D;
    }

    uint fwd_base;
    if (params.go_seq_major != 0u) {
        fwd_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        fwd_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
    }

    float lse_val = lse[b * params.H * params.N + h * params.N + q_pos];

    // Compute D_i = dot(grad_out[q], fwd_out[q]) via wave reduction
    float go_val  = (tid < params.D) ? grad_out[go_base + tid] : 0.0f;
    float fo_val  = (tid < params.D) ? fwd_out[fwd_base + tid] : 0.0f;
    float di = WaveActiveSum(go_val * fo_val);

    // Each thread accumulates its element of grad_Q
    float gq_acc = 0.0f;

    for (uint k = 0; k < params.S; k++) {
        if (params.is_causal != 0 && k > q_pos) continue;

        uint k_base = params.q_seq_major != 0u
            ? k_base0 + k * params.KV_H * params.D
            : k_base0 + k * params.D;

        // score = Q[q] . K[k] * scale
        float q_val = (tid < params.D) ? Q[q_base + tid] : 0.0f;
        float k_val = (tid < params.D) ? K[k_base + tid] : 0.0f;
        float score = WaveActiveSum(q_val * k_val) * params.scale;

        float p_k = exp(score - lse_val);

        // ds_k = dot(go[q], V[k]) - D_i
        uint v_base = params.q_seq_major != 0u
            ? v_base0 + k * params.KV_H * params.D
            : v_base0 + k * params.D;
        float v_val = (tid < params.D) ? V[v_base + tid] : 0.0f;
        float ds_k = WaveActiveSum(go_val * v_val) - di;

        // grad_Q[q, tid] += scale * p_k * ds_k * K[k, tid]
        if (tid < params.D) {
            gq_acc += params.scale * p_k * ds_k * K[k_base + tid];
        }
    }

    if (tid < params.D) {
        grad_Q[gq_base + tid] = gq_acc;
    }
}
