// addcdiv: self + value * tensor1 / tensor2
// Used by Adam optimizer

struct Params {
    float value;
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> self_buf;
[[vk::binding(1)]] StructuredBuffer<float> tensor1;
[[vk::binding(2)]] StructuredBuffer<float> tensor2;
[[vk::binding(3)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = self_buf[tid.x] + params.value * tensor1[tid.x] / tensor2[tid.x];
}
