// SELU backward: grad_input = grad_output * (x > 0 ? scale : scale * alpha * exp(x))
// scale = 1.0507009873554804934193349852946, alpha = 1.6732632423543772848170429916717

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> input;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    float x = input[tid.x];
    float go = grad_output[tid.x];

    float scale = 1.0507009873554804934193349852946f;
    float alpha = 1.6732632423543772848170429916717f;

    float grad = (x > 0.0f) ? scale : scale * alpha * exp(x);
    grad_input[tid.x] = go * grad;
}
