// BCE with Logits backward: grad_input = grad_output * (sigmoid(input) - target)

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
    float x = input[tid.x];
    float t = target[tid.x];
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    grad_input[tid.x] = grad_output[tid.x] * (sigmoid_x - t);
}
