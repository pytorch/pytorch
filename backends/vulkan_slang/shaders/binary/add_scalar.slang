// Scalar addition: out[i] = input[i] + scalar * alpha
// Uses push constant scalar instead of allocating a full tensor.

struct Params {
    uint numel;
    float scalar;
    float alpha;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = input[tid.x] + params.scalar * params.alpha;
}
