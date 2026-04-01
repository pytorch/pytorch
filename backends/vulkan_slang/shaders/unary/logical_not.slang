// Logical NOT: out = (input == 0.0) ? 1.0 : 0.0
// Works on float representation of bool values.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = (input[tid.x] == 0.0f) ? 1.0f : 0.0f;
}
