// Fused threshold (ReLU) backward: grad_input = grad_output * (self > threshold ? 1 : 0)
// Single pass replacing: gt + to_float + mul (3 dispatches → 1)

struct Params {
    uint numel;
    float threshold;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> self;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    grad_input[tid.x] = (self[tid.x] > params.threshold) ? grad_output[tid.x] : 0.0f;
}
