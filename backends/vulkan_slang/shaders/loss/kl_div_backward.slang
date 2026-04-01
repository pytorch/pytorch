// KL Divergence backward:
// log_target=false: grad_input[i] = -grad_output[i] * target[i]
// log_target=true:  grad_input[i] = -grad_output[i] * exp(target[i])

struct Params { uint numel; uint log_target; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> target;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float t = target[tid.x];
    float g = grad_output[tid.x];
    if (params.log_target != 0) {
        grad_input[tid.x] = -g * exp(t);
    } else {
        grad_input[tid.x] = -g * t;
    }
}
