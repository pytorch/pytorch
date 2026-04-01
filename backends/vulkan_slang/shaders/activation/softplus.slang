// Softplus: (1/beta) * log(1 + exp(beta * x))
// For numerical stability: if beta*x > threshold, return x

struct Params { uint numel; float beta; float threshold; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    float bx = params.beta * x;
    if (bx > params.threshold) {
        out[tid.x] = x;
    } else {
        out[tid.x] = log(1.0f + exp(bx)) / params.beta;
    }
}
