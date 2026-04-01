// Scalar less-or-equal: out[i] = (input[i] <= scalar) ? 1.0 : 0.0

struct Params { uint numel; float scalar; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = (input[tid.x] <= params.scalar) ? 1.0f : 0.0f;
}
