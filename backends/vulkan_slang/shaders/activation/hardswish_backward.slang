// Hardswish backward:
// 0 if x <= -3
// grad if x >= 3
// grad * (2x + 3) / 6 otherwise

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
    float g;
    if (x <= -3.0f) g = 0.0f;
    else if (x >= 3.0f) g = 1.0f;
    else g = (2.0f * x + 3.0f) / 6.0f;
    grad_input[tid.x] = grad_output[tid.x] * g;
}
