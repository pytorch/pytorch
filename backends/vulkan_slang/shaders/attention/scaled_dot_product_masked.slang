// Scaled Dot-Product Attention with additive mask
// Q, K, V: [B, H, N, D] flattened
// Mask: [B, H, N, S] flattened (additive: 0 = attend, -inf = block)
// Output: [B, H, N, D]

struct Params {
    uint B;      // batch size
    uint H;      // num heads
    uint N;      // query sequence length
    uint D;      // head dimension
    float scale; // 1/sqrt(D)
    uint S;      // key sequence length (may differ from N)
};

[[vk::binding(0)]] StructuredBuffer<float> Q;
[[vk::binding(1)]] StructuredBuffer<float> K;
[[vk::binding(2)]] StructuredBuffer<float> V;
[[vk::binding(3)]] StructuredBuffer<float> mask;
[[vk::binding(4)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.B * params.H * params.N * params.D;
    if (tid.x >= total) return;

    uint idx = tid.x;
    uint d = idx % params.D; idx /= params.D;
    uint n = idx % params.N; idx /= params.N;
    uint h = idx % params.H; idx /= params.H;
    uint b = idx;

    uint qkv_head_base = (b * params.H + h) * params.N * params.D;
    uint mask_head_base = (b * params.H + h) * params.N * params.S;

    // Step 1: Find max score for numerical stability
    float max_score = -3.402823466e+38f;
    for (uint m = 0; m < params.S; m++) {
        float score = 0.0f;
        uint q_base = qkv_head_base + n * params.D;
        uint k_base = (b * params.H + h) * params.S * params.D + m * params.D;
        for (uint dd = 0; dd < params.D; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score = score * params.scale + mask[mask_head_base + n * params.S + m];
        max_score = max(max_score, score);
    }

    // Step 2: Compute exp(score - max) and sum, accumulate weighted V
    float exp_sum = 0.0f;
    float result = 0.0f;

    for (uint m = 0; m < params.S; m++) {
        float score = 0.0f;
        uint q_base = qkv_head_base + n * params.D;
        uint k_base = (b * params.H + h) * params.S * params.D + m * params.D;
        for (uint dd = 0; dd < params.D; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score = score * params.scale + mask[mask_head_base + n * params.S + m];
        float w = exp(score - max_score);
        exp_sum += w;
        uint v_base = (b * params.H + h) * params.S * params.D + m * params.D;
        result += w * V[v_base + d];
    }

    out[tid.x] = result / max(exp_sum, 1e-9f);
}
