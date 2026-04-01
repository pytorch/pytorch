// Smooth L1 / Huber Loss forward:
// if |x - t| < beta: 0.5 * (x-t)^2 / beta
// else:              |x-t| - 0.5 * beta

struct Params { uint numel; float beta; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> target;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float diff = input[tid.x] - target[tid.x];
    float abs_diff = abs(diff);
    float beta = params.beta;
    if (beta == 0.0f) {
        out[tid.x] = abs_diff;
    } else if (abs_diff < beta) {
        out[tid.x] = 0.5f * diff * diff / beta;
    } else {
        out[tid.x] = abs_diff - 0.5f * beta;
    }
}
