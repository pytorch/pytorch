// Fused SGD without momentum:
//   param = param - lr * (grad + weight_decay * param)

struct Params {
    uint numel;
    float lr;
    float weight_decay;
};

[[vk::binding(0)]] RWStructuredBuffer<float> param;     // [numel] — updated in-place
[[vk::binding(1)]] StructuredBuffer<float> grad;         // [numel]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    float g = grad[tid.x];
    float p = param[tid.x];

    if (params.weight_decay != 0.0f) {
        g = g + params.weight_decay * p;
    }

    param[tid.x] = p - params.lr * g;
}
