// Hardtanh backward: grad * (min_val <= self <= max_val ? 1 : 0)

struct Params { uint numel; float min_val; float max_val; };

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
    float mask = (x >= params.min_val && x <= params.max_val) ? 1.0f : 0.0f;
    grad_input[tid.x] = grad_output[tid.x] * mask;
}
