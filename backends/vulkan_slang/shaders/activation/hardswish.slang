// Hardswish: x * relu6(x+3) / 6
// = 0 if x <= -3
// = x if x >= 3
// = x * (x+3) / 6 otherwise

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
    float result;
    if (x <= -3.0f) result = 0.0f;
    else if (x >= 3.0f) result = x;
    else result = x * (x + 3.0f) / 6.0f;
    out[tid.x] = result;
}
