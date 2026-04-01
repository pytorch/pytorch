// Batched AdamW (decoupled weight decay): update up to 7 params in one dispatch.
// 2D dispatch: X = element index (256 threads/wg), Y = param index (0..n_params-1)
// Push constants: 4 + 7*32 = 228 bytes (within NVIDIA's 256-byte guarantee).
// Reduces optimizer step from N dispatches to ceil(N/7) dispatches.
// Use when device maxPushConstantsSize >= 256 (NVIDIA GPUs).

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
    uint n_params;           // how many params in this batch (1..7)
    ParamConfig cfg[7];      // per-param config (8 floats each = 32 bytes)
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

[[vk::binding(12)]] RWStructuredBuffer<float> param3;
[[vk::binding(13)]] StructuredBuffer<float>   grad3;
[[vk::binding(14)]] RWStructuredBuffer<float> m3;
[[vk::binding(15)]] RWStructuredBuffer<float> v3;

[[vk::binding(16)]] RWStructuredBuffer<float> param4;
[[vk::binding(17)]] StructuredBuffer<float>   grad4;
[[vk::binding(18)]] RWStructuredBuffer<float> m4;
[[vk::binding(19)]] RWStructuredBuffer<float> v4;

[[vk::binding(20)]] RWStructuredBuffer<float> param5;
[[vk::binding(21)]] StructuredBuffer<float>   grad5;
[[vk::binding(22)]] RWStructuredBuffer<float> m5;
[[vk::binding(23)]] RWStructuredBuffer<float> v5;

[[vk::binding(24)]] RWStructuredBuffer<float> param6;
[[vk::binding(25)]] StructuredBuffer<float>   grad6;
[[vk::binding(26)]] RWStructuredBuffer<float> m6;
[[vk::binding(27)]] RWStructuredBuffer<float> v6;

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
    if      (param_idx == 0) { p = param0[elem]; g = grad0[elem]; m_val = m0[elem]; v_val = v0[elem]; }
    else if (param_idx == 1) { p = param1[elem]; g = grad1[elem]; m_val = m1[elem]; v_val = v1[elem]; }
    else if (param_idx == 2) { p = param2[elem]; g = grad2[elem]; m_val = m2[elem]; v_val = v2[elem]; }
    else if (param_idx == 3) { p = param3[elem]; g = grad3[elem]; m_val = m3[elem]; v_val = v3[elem]; }
    else if (param_idx == 4) { p = param4[elem]; g = grad4[elem]; m_val = m4[elem]; v_val = v4[elem]; }
    else if (param_idx == 5) { p = param5[elem]; g = grad5[elem]; m_val = m5[elem]; v_val = v5[elem]; }
    else                     { p = param6[elem]; g = grad6[elem]; m_val = m6[elem]; v_val = v6[elem]; }

    p = p * (1.0f - cfg.lr * cfg.weight_decay);
    m_val = cfg.beta1 * m_val + (1.0f - cfg.beta1) * g;
    v_val = cfg.beta2 * v_val + (1.0f - cfg.beta2) * g * g;
    float m_hat = m_val / cfg.bias_correction1;
    float v_hat = v_val / cfg.bias_correction2;
    p = p - cfg.lr * m_hat / (sqrt(v_hat) + cfg.eps);

    if      (param_idx == 0) { param0[elem] = p; m0[elem] = m_val; v0[elem] = v_val; }
    else if (param_idx == 1) { param1[elem] = p; m1[elem] = m_val; v1[elem] = v_val; }
    else if (param_idx == 2) { param2[elem] = p; m2[elem] = m_val; v2[elem] = v_val; }
    else if (param_idx == 3) { param3[elem] = p; m3[elem] = m_val; v3[elem] = v_val; }
    else if (param_idx == 4) { param4[elem] = p; m4[elem] = m_val; v4[elem] = v_val; }
    else if (param_idx == 5) { param5[elem] = p; m5[elem] = m_val; v5[elem] = v_val; }
    else                     { param6[elem] = p; m6[elem] = m_val; v6[elem] = v_val; }
}
