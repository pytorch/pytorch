// MSE Loss forward: out[i] = (input[i] - target[i])^2
// Reduction (mean/sum) is done on the C++ side after this kernel.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> target;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float diff = input[tid.x] - target[tid.x];
    out[tid.x] = diff * diff;
}
