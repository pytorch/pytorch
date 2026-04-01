// Fused SwiGLU: out[i] = silu(gate[i]) * up[i]
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Saves one kernel launch + one intermediate buffer vs separate silu + mul.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> gate;
[[vk::binding(1)]] StructuredBuffer<float> up;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float g = gate[tid.x];
    float u = up[tid.x];
    float silu_g = g / (1.0f + exp(-g));
    out[tid.x] = silu_g * u;
}
