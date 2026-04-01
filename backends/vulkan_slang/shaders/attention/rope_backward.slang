// RoPE backward: apply inverse rotation to gradients
// grad_input[2i]   = grad[2i]*cos + grad[2i+1]*sin
// grad_input[2i+1] = -grad[2i]*sin + grad[2i+1]*cos
// Input: grad_output [B, H, N, D]
// Output: grad_input [B, H, N, D]

struct Params {
    uint B;
    uint H;
    uint N;
    uint D;
    float theta;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] RWStructuredBuffer<float> grad_input;
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

    // Only process even indices (pairs)
    if (d % 2 != 0) return;

    uint pair_idx = d / 2;
    float freq = float(n) / pow(params.theta, 2.0f * float(pair_idx) / float(params.D));
    float cos_val = cos(freq);
    float sin_val = sin(freq);

    uint base = tid.x;  // even index
    float g_even = grad_output[base];
    float g_odd = grad_output[base + 1];

    // Inverse rotation (transpose of rotation matrix)
    grad_input[base]     = g_even * cos_val + g_odd * sin_val;
    grad_input[base + 1] = -g_even * sin_val + g_odd * cos_val;
}
