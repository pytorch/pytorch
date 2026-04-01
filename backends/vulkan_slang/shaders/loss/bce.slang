// BCE Loss forward: out[i] = -(target[i] * log(input[i]) + (1 - target[i]) * log(1 - input[i]))
// Clamps log arguments to avoid -inf

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
    float x = clamp(input[tid.x], 1e-7f, 1.0f - 1e-7f);
    float t = target[tid.x];
    out[tid.x] = -(t * log(x) + (1.0f - t) * log(1.0f - x));
}
