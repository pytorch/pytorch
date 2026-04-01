// Fused SGD with momentum update:
//   buf = momentum * buf + grad + weight_decay * param
//   param = param - lr * buf
// All in a single dispatch.

struct Params {
    uint numel;
    float lr;
    float momentum;
    float dampening;
    float weight_decay;
    uint nesterov;     // 0 or 1
};

[[vk::binding(0)]] RWStructuredBuffer<float> param;     // [numel] — updated in-place
[[vk::binding(1)]] StructuredBuffer<float> grad;         // [numel]
[[vk::binding(2)]] RWStructuredBuffer<float> momentum_buf; // [numel] — updated in-place
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    float g = grad[tid.x];
    float p = param[tid.x];

    // Weight decay (L2 regularization)
    if (params.weight_decay != 0.0f) {
        g = g + params.weight_decay * p;
    }

    // Momentum update
    float buf = params.momentum * momentum_buf[tid.x] + (1.0f - params.dampening) * g;
    momentum_buf[tid.x] = buf;

    // Nesterov: use g + momentum * buf instead of buf
    float update = buf;
    if (params.nesterov != 0u) {
        update = g + params.momentum * buf;
    }

    // Parameter update
    param[tid.x] = p - params.lr * update;
}
