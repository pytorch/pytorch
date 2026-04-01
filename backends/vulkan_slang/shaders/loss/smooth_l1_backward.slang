// Smooth L1 backward:
// if |diff| < beta: grad * diff / beta
// else:             grad * sign(diff)

struct Params { uint numel; float beta; };

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
    float diff = input[tid.x] - target[tid.x];
    float abs_diff = abs(diff);
    float beta = params.beta;
    float g;
    if (beta == 0.0f) {
        g = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    } else if (abs_diff < beta) {
        g = diff / beta;
    } else {
        g = (diff > 0.0f) ? 1.0f : -1.0f;
    }
    grad_input[tid.x] = grad_output[tid.x] * g;
}
