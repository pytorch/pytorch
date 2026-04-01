// Leaky ReLU backward: grad_input = grad_output * (x > 0 ? 1.0 : negative_slope)

struct Params {
    uint numel;
    float negative_slope;
};

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

    float grad = (x > 0.0f) ? 1.0f : params.negative_slope;
    grad_input[tid.x] = go * grad;
}
