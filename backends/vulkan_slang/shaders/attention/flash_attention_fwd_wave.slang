// Flash Attention forward — wave-intrinsic variant for D <= 32.
// Uses WaveActiveSum for dot-product reduction (zero barrier syncs for score computation).
// One workgroup per (b, h, q). Threads = 32 (= wave size on NVIDIA/AMD).
//
// For D=32: each thread owns exactly 1 element — WaveActiveSum reduces in hardware.
// For D<32:  threads d>=D contribute 0, result still correct.
// For D>32: falls back to strided accumulation (still no shared mem needed for D<=64 with 2 waves).
//           But correctness requires that WaveActiveSum sums all 32 threads — fine since D<=32 always
//           with this variant.
//
// Layout identical to flash_attention_fwd.slang (supports both head-major and seq-major).

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
};

[[vk::binding(0)]] StructuredBuffer<float> Q;
[[vk::binding(1)]] StructuredBuffer<float> K;
[[vk::binding(2)]] StructuredBuffer<float> V;
[[vk::binding(3)]] RWStructuredBuffer<float> out;
[[vk::binding(4)]] RWStructuredBuffer<float> lse;
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

    float m    = -1e38f;
    float l    = 0.0f;
    float o_acc = 0.0f;

    for (uint k = 0; k < params.S; k++) {
        float causal_bias = 0.0f;
        if (params.is_causal != 0 && k > q_pos) {
            causal_bias = -1e38f;
        }

        uint k_base = params.q_seq_major != 0u
            ? k_base0 + k * params.KV_H * params.D
            : k_base0 + k * params.D;

        // Parallel dot product via wave reduction (no shared memory, no barrier)
        float q_val = (tid < params.D) ? Q[q_base + tid] : 0.0f;
        float k_val = (tid < params.D) ? K[k_base + tid] : 0.0f;
        float score = WaveActiveSum(q_val * k_val) * params.scale + causal_bias;

        float m_new    = max(m, score);
        float exp_score = exp(score - m_new);
        float scale_old = exp(m - m_new);

        float v_val = (tid < params.D) ? V[k_base + tid] : 0.0f;
        o_acc = o_acc * scale_old + exp_score * v_val;

        l = l * scale_old + exp_score;
        m = m_new;
    }

    // Normalize and write output
    if (tid < params.D) {
        out[o_base + tid] = o_acc / l;
    }
    if (tid == 0) {
        uint lse_idx = b * params.H * params.N + h * params.N + q_pos;
        lse[lse_idx] = m + log(l);
    }
}
