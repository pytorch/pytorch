// Rotary Position Embedding (RoPE)
// Input: [B, H, N, D] flattened (D must be even)
// Applies rotation to pairs (x[2i], x[2i+1]) using position-dependent angles.

struct Params {
    uint B;
    uint H;
    uint N;
    uint D;       // head dim (must be even)
    float theta;  // base frequency (default 10000.0)
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
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
    // h and b not needed for the rotation

    uint half_d = params.D / 2;
    uint pair_idx = d / 2;
    float freq = 1.0f / pow(params.theta, float(pair_idx) / float(half_d));
    float angle = float(n) * freq;

    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x = input[tid.x];

    if (d % 2 == 0) {
        // Even index: x * cos - x_next * sin
        float x_next = input[tid.x + 1];
        out[tid.x] = x * cos_a - x_next * sin_a;
    } else {
        // Odd index: x_prev * sin + x * cos
        float x_prev = input[tid.x - 1];
        out[tid.x] = x_prev * sin_a + x * cos_a;
    }
}
