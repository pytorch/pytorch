// Hardsigmoid backward:
// 0 if x <= -3 or x >= 3
// grad / 6 otherwise

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> self;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = self[tid.x];
    float g = (x > -3.0f && x < 3.0f) ? (1.0f / 6.0f) : 0.0f;
    grad_input[tid.x] = grad_output[tid.x] * g;
}
