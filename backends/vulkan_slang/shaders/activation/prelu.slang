// PReLU activation: max(0, x) + weight * min(0, x)
// Weight can be scalar or per-channel.
// This shader handles the simple case where weight is broadcast to match numel.

struct Params {
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> weight;  // broadcast to match input
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    float w = weight[tid.x];
    out[tid.x] = (x >= 0.0f) ? x : w * x;
}
