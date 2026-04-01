// Flash Attention forward pass — small D variant (D <= 64, 64 threads/workgroup).
// Same algorithm as flash_attention_fwd.slang but tuned for small head dimensions.
// Dispatch when D <= 64 to reduce barrier overhead (6 vs 8 reduction steps).
//
// Layout:
//   Q: [B, H, N, D]  — query
//   K: [B, KV_H, S, D]  — key (KV_H may be < H for GQA)
//   V: [B, KV_H, S, D]  — value
//   out: [B, H, N, D]

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
};

[[vk::binding(0)]] StructuredBuffer<float> Q;
[[vk::binding(1)]] StructuredBuffer<float> K;
[[vk::binding(2)]] StructuredBuffer<float> V;
[[vk::binding(3)]] RWStructuredBuffer<float> out;
[[vk::binding(4)]] RWStructuredBuffer<float> lse;
[[vk::push_constant]] Params params;

groupshared float smem[THREADS];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint flat  = gid.x;
    uint q_pos = flat % params.N;
    uint h     = (flat / params.N) % params.H;
    uint b     = flat / (params.N * params.H);

    if (b >= params.B) return;

    uint tid   = lid.x;
    uint kv_h  = h / (params.H / params.KV_H);

    uint q_base, k_base0, o_base;
    if (params.q_seq_major != 0u) {
        q_base  = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
        k_base0 = b * params.S * params.KV_H * params.D + kv_h * params.D;
        o_base  = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        q_base  = b * params.H   * params.N * params.D + h   * params.N * params.D + q_pos * params.D;
        k_base0 = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D;
        o_base  = b * params.H   * params.N * params.D + h   * params.N * params.D + q_pos * params.D;
    }

    float m = -1e38f;
    float l = 0.0f;
    float o_acc = 0.0f;

    for (uint k = 0; k < params.S; k++) {
        float causal_bias = 0.0f;
        if (params.is_causal != 0 && k > q_pos) {
            causal_bias = -1e38f;
        }

        uint k_base = params.q_seq_major != 0u
            ? k_base0 + k * params.KV_H * params.D
            : k_base0 + k * params.D;

        // Parallel dot product with THREADS threads
        float dot_part = 0.0f;
        for (uint d = tid; d < params.D; d += THREADS) {
            dot_part += Q[q_base + d] * K[k_base + d];
        }
        smem[tid] = dot_part;
        GroupMemoryBarrierWithGroupSync();

        // Reduction: THREADS/2 steps (for THREADS=64: 5 steps)
        for (uint s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }
        float score = smem[0] * params.scale + causal_bias;

        float m_new = max(m, score);
        float exp_score = exp(score - m_new);
        float scale_old = exp(m - m_new);

        float v_val = 0.0f;
        if (tid < params.D) {
            v_val = V[k_base + tid];
        }
        o_acc = o_acc * scale_old + exp_score * v_val;

        l = l * scale_old + exp_score;
        m = m_new;

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < params.D) {
        out[o_base + tid] = o_acc / l;
    }
    if (tid == 0) {
        uint lse_idx = b * params.H * params.N + h * params.N + q_pos;
        lse[lse_idx] = m + log(l);
    }
}
