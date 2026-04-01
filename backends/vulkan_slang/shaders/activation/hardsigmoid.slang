// Hardsigmoid: clamp(x/6 + 0.5, 0, 1)
// = 0 if x <= -3
// = 1 if x >= 3
// = x/6 + 0.5 otherwise

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    out[tid.x] = clamp(x / 6.0f + 0.5f, 0.0f, 1.0f);
}
