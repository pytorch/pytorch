// KL Divergence: out[i] = target[i] * (log(target[i]) - self[i])
// When log_target=false: self is log-probabilities, target is probabilities
// When log_target=true:  self is log-probabilities, target is log-probabilities
// This shader handles log_target=false (default PyTorch behavior)

struct Params { uint numel; uint log_target; };

[[vk::binding(0)]] StructuredBuffer<float> self;
[[vk::binding(1)]] StructuredBuffer<float> target;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float s = self[tid.x];
    float t = target[tid.x];
    if (params.log_target != 0) {
        // log_target mode: target is log-probabilities
        out[tid.x] = exp(t) * (t - s);
    } else {
        // default: target is probabilities
        out[tid.x] = (t > 0.0f) ? t * (log(t) - s) : 0.0f;
    }
}
