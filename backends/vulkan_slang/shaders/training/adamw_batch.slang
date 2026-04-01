// Batched AdamW (decoupled weight decay): update up to 3 params in one dispatch.
// 2D dispatch: X = element index (256 threads/wg), Y = param index (0..n_params-1)
// All config in push constants (4 + 3*32 = 100 bytes, fits in Vulkan min 128-byte limit).
// Reduces optimizer step from N dispatches to ceil(N/3) dispatches.

struct ParamConfig {
    uint  numel;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
};

struct PushConstants {
    uint n_params;
    ParamConfig cfg[3];
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0)]]  RWStructuredBuffer<float> param0;
[[vk::binding(1)]]  StructuredBuffer<float>   grad0;
[[vk::binding(2)]]  RWStructuredBuffer<float> m0;
[[vk::binding(3)]]  RWStructuredBuffer<float> v0;

[[vk::binding(4)]]  RWStructuredBuffer<float> param1;
[[vk::binding(5)]]  StructuredBuffer<float>   grad1;
[[vk::binding(6)]]  RWStructuredBuffer<float> m1;
[[vk::binding(7)]]  RWStructuredBuffer<float> v1;

[[vk::binding(8)]]  RWStructuredBuffer<float> param2;
[[vk::binding(9)]]  StructuredBuffer<float>   grad2;
[[vk::binding(10)]] RWStructuredBuffer<float> m2;
[[vk::binding(11)]] RWStructuredBuffer<float> v2;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint param_idx = gid.y;
    if (param_idx >= pc.n_params) return;

    ParamConfig cfg = pc.cfg[param_idx];
    uint elem = gid.x * 256 + lid.x;
    if (elem >= cfg.numel) return;

    float p, g, m_val, v_val;
    if (param_idx == 0) {
        p = param0[elem]; g = grad0[elem]; m_val = m0[elem]; v_val = v0[elem];
    } else if (param_idx == 1) {
        p = param1[elem]; g = grad1[elem]; m_val = m1[elem]; v_val = v1[elem];
    } else {
        p = param2[elem]; g = grad2[elem]; m_val = m2[elem]; v_val = v2[elem];
    }

    p = p * (1.0f - cfg.lr * cfg.weight_decay);
    m_val = cfg.beta1 * m_val + (1.0f - cfg.beta1) * g;
    v_val = cfg.beta2 * v_val + (1.0f - cfg.beta2) * g * g;
    float m_hat = m_val / cfg.bias_correction1;
    float v_hat = v_val / cfg.bias_correction2;
    p = p - cfg.lr * m_hat / (sqrt(v_hat) + cfg.eps);

    if (param_idx == 0) {
        param0[elem] = p; m0[elem] = m_val; v0[elem] = v_val;
    } else if (param_idx == 1) {
        param1[elem] = p; m1[elem] = m_val; v1[elem] = v_val;
    } else {
        param2[elem] = p; m2[elem] = m_val; v2[elem] = v_val;
    }
}
