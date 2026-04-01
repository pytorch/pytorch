// Fused tanh backward: grad_input = grad_output * (1 - output * output)
// Single-pass instead of 3 separate element-wise ops.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> output;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float o = output[tid.x];
    grad_input[tid.x] = grad_output[tid.x] * (1.0f - o * o);
}
