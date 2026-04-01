// Scaled Dot-Product Attention (naive implementation)
// Q, K, V: [B, H, N, D] flattened
// Output: [B, H, N, D]
// Each thread computes one element of the output.
// This is a naive O(N^2 D) implementation. Flash attention would tile.

struct Params {
    uint B;      // batch size
    uint H;      // num heads
    uint N;      // sequence length
    uint D;      // head dimension
    float scale; // 1/sqrt(D)
};

[[vk::binding(0)]] StructuredBuffer<float> Q;
[[vk::binding(1)]] StructuredBuffer<float> K;
[[vk::binding(2)]] StructuredBuffer<float> V;
[[vk::binding(3)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.B * params.H * params.N * params.D;
    if (tid.x >= total) return;

    // Decode indices: [b, h, n, d]
    uint idx = tid.x;
    uint d = idx % params.D; idx /= params.D;
    uint n = idx % params.N; idx /= params.N;
    uint h = idx % params.H; idx /= params.H;
    uint b = idx;

    uint head_base = (b * params.H + h) * params.N * params.D;

    // Compute attention scores for position n: score[m] = Q[n] . K[m] * scale
    // Then softmax over m, then weighted sum of V

    // Step 1: Find max score for numerical stability
    float max_score = -3.402823466e+38f;
    for (uint m = 0; m < params.N; m++) {
        float score = 0.0f;
        uint q_base = head_base + n * params.D;
        uint k_base = head_base + m * params.D;
        for (uint dd = 0; dd < params.D; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score *= params.scale;
        max_score = max(max_score, score);
    }

    // Step 2: Compute exp(score - max) and sum
    float exp_sum = 0.0f;
    // Also accumulate weighted V
    float result = 0.0f;

    for (uint m = 0; m < params.N; m++) {
        float score = 0.0f;
        uint q_base = head_base + n * params.D;
        uint k_base = head_base + m * params.D;
        for (uint dd = 0; dd < params.D; dd++) {
            score += Q[q_base + dd] * K[k_base + dd];
        }
        score *= params.scale;
        float w = exp(score - max_score);
        exp_sum += w;
        result += w * V[head_base + m * params.D + d];
    }

    out[tid.x] = result / exp_sum;
}
