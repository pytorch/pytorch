// BCE Loss backward: grad_input[i] = grad_output[i] * (-target[i]/input[i] + (1-target[i])/(1-input[i]))

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> input;
[[vk::binding(2)]] StructuredBuffer<float> target;
[[vk::binding(3)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = clamp(input[tid.x], 1e-7f, 1.0f - 1e-7f);
    float t = target[tid.x];
    float g = (-t / x + (1.0f - t) / (1.0f - x));
    grad_input[tid.x] = grad_output[tid.x] * g;
}
