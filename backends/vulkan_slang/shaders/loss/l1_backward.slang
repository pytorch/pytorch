// L1 Loss backward: grad_input[i] = grad_output[i] * sign(self[i] - target[i])

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> self;
[[vk::binding(2)]] StructuredBuffer<float> target;
[[vk::binding(3)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float diff = self[tid.x] - target[tid.x];
    float s = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    grad_input[tid.x] = grad_output[tid.x] * s;
}
