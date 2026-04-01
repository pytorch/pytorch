// MSE Loss backward: grad_input[i] = grad_output[i] * 2 * (input[i] - target[i]) / divisor
// divisor = numel for 'mean' reduction, 1 for 'sum'

struct Params {
    uint numel;
    float divisor;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> input;
[[vk::binding(2)]] StructuredBuffer<float> target;
[[vk::binding(3)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float diff = input[tid.x] - target[tid.x];
    grad_input[tid.x] = grad_output[tid.x] * 2.0f * diff / params.divisor;
}
