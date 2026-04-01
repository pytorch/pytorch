// lerp: self + weight * (end - self)

struct Params {
    float weight;
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> self_buf;
[[vk::binding(1)]] StructuredBuffer<float> end_buf;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float s = self_buf[tid.x];
    out[tid.x] = s + params.weight * (end_buf[tid.x] - s);
}
