// Create valid-target mask on GPU.
// Input: target [N] (int32 packed as float via asint/asfloat)
// Output: mask [N] (1.0 if target != ignore_index, 0.0 otherwise)
// The caller sums this mask to get the count.

struct Params {
    uint N;
    int ignore_index;
};

[[vk::binding(0)]] StructuredBuffer<float> target;    // [N] int32 packed as float
[[vk::binding(1)]] RWStructuredBuffer<float> mask;     // [N] output mask
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.N) return;
    int t = asint(target[tid.x]);
    mask[tid.x] = (t != params.ignore_index) ? 1.0f : 0.0f;
}
