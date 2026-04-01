// Batched SGD without momentum: update up to 7 params in one dispatch.
// 2D dispatch: X = element index (256 threads/wg), Y = param index (0..n_params-1)
// All config in push constants (88 bytes max for 7 params).
// Reduces optimizer step from N dispatches to ceil(N/7) dispatches.

struct ParamConfig {
    uint  numel;
    float lr;
    float weight_decay;
};

struct PushConstants {
    uint n_params;          // how many params in this batch (1..7)
    ParamConfig cfg[7];     // per-param config (3 floats each)
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0)]]  RWStructuredBuffer<float> param0;
[[vk::binding(1)]]  StructuredBuffer<float>   grad0;
[[vk::binding(2)]]  RWStructuredBuffer<float> param1;
[[vk::binding(3)]]  StructuredBuffer<float>   grad1;
[[vk::binding(4)]]  RWStructuredBuffer<float> param2;
[[vk::binding(5)]]  StructuredBuffer<float>   grad2;
[[vk::binding(6)]]  RWStructuredBuffer<float> param3;
[[vk::binding(7)]]  StructuredBuffer<float>   grad3;
[[vk::binding(8)]]  RWStructuredBuffer<float> param4;
[[vk::binding(9)]]  StructuredBuffer<float>   grad4;
[[vk::binding(10)]] RWStructuredBuffer<float> param5;
[[vk::binding(11)]] StructuredBuffer<float>   grad5;
[[vk::binding(12)]] RWStructuredBuffer<float> param6;
[[vk::binding(13)]] StructuredBuffer<float>   grad6;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint param_idx = gid.y;
    if (param_idx >= pc.n_params) return;

    ParamConfig cfg = pc.cfg[param_idx];
    uint elem = gid.x * 256 + lid.x;
    if (elem >= cfg.numel) return;

    float p, g;
    if (param_idx == 0) { p = param0[elem]; g = grad0[elem]; }
    else if (param_idx == 1) { p = param1[elem]; g = grad1[elem]; }
    else if (param_idx == 2) { p = param2[elem]; g = grad2[elem]; }
    else if (param_idx == 3) { p = param3[elem]; g = grad3[elem]; }
    else if (param_idx == 4) { p = param4[elem]; g = grad4[elem]; }
    else if (param_idx == 5) { p = param5[elem]; g = grad5[elem]; }
    else                     { p = param6[elem]; g = grad6[elem]; }

    if (cfg.weight_decay != 0.0f) g = g + cfg.weight_decay * p;
    float p_new = p - cfg.lr * g;

    if (param_idx == 0) param0[elem] = p_new;
    else if (param_idx == 1) param1[elem] = p_new;
    else if (param_idx == 2) param2[elem] = p_new;
    else if (param_idx == 3) param3[elem] = p_new;
    else if (param_idx == 4) param4[elem] = p_new;
    else if (param_idx == 5) param5[elem] = p_new;
    else                     param6[elem] = p_new;
}
