// Fused AdamW update (decoupled weight decay):
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
//
// Buffers m and v are stored in float32 (master state).
// param is float32 (called with widened copy, result cast back by host).

struct Params {
    uint numel;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;   // 1 - beta1^step
    float bias_correction2;   // 1 - beta2^step
};

[[vk::binding(0)]] RWStructuredBuffer<float> param;  // [numel] float32, in-place
[[vk::binding(1)]] StructuredBuffer<float> grad;     // [numel] float32
[[vk::binding(2)]] RWStructuredBuffer<float> m;      // [numel] float32, in-place
[[vk::binding(3)]] RWStructuredBuffer<float> v;      // [numel] float32, in-place
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    float g = grad[tid.x];
    float p = param[tid.x];

    // Update moments
    float m_val = params.beta1 * m[tid.x] + (1.0f - params.beta1) * g;
    float v_val = params.beta2 * v[tid.x] + (1.0f - params.beta2) * g * g;
    m[tid.x] = m_val;
    v[tid.x] = v_val;

    // Bias-corrected estimates
    float m_hat = m_val / params.bias_correction1;
    float v_hat = v_val / params.bias_correction2;

    // AdamW decoupled weight decay + param update
    float new_p = p * (1.0f - params.lr * params.weight_decay)
                  - params.lr * m_hat / (sqrt(v_hat) + params.eps);
    param[tid.x] = new_p;
}
