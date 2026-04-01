// Masked fill: out[i] = mask[i] != 0 ? fill_value : input[i]

struct Params {
    float fill_value;
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> mask;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = (mask[tid.x] != 0.0f) ? params.fill_value : input[tid.x];
}
