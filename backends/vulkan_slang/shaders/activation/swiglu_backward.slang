// Fused SwiGLU backward: y = silu(gate) * up
// d_gate = grad * up * silu'(gate)
// d_up   = grad * silu(gate)
// where silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> gate;
[[vk::binding(2)]] StructuredBuffer<float> up;
[[vk::binding(3)]] RWStructuredBuffer<float> grad_gate;
[[vk::binding(4)]] RWStructuredBuffer<float> grad_up;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float go = grad_output[tid.x];
    float g = gate[tid.x];
    float u = up[tid.x];

    float sig_g = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sig_g;

    // silu'(x) = sig(x) * (1 + x * (1 - sig(x)))
    float silu_prime_g = sig_g * (1.0f + g * (1.0f - sig_g));

    grad_gate[tid.x] = go * u * silu_prime_g;
    grad_up[tid.x] = go * silu_g;
}
