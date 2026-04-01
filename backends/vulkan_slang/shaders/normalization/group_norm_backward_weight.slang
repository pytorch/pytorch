// Group normalization backward for grad_weight and grad_bias.
// One thread per channel, each accumulates over batch and spatial dims.
//
// Input is [N, C, HxW], mean/rstd are [N*G].
// grad_weight[c] = sum_n sum_hw (grad_output[n,c,hw] * x_hat[n,c,hw])
// grad_bias[c] = sum_n sum_hw (grad_output[n,c,hw])
// where x_hat = (x - mean[n*G + group_of_c]) * rstd[n*G + group_of_c]

struct Params {
    uint N;       // batch size
    uint C;       // number of channels
    uint HxW;     // spatial size per channel
    uint G;       // number of groups
    uint cpg;     // channels per group = C / G
    uint compute_weight;
    uint compute_bias;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [N, C, HxW]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [N, C, HxW]
[[vk::binding(2)]] StructuredBuffer<float> mean;          // [N * G]
[[vk::binding(3)]] StructuredBuffer<float> rstd;          // [N * G]
[[vk::binding(4)]] RWStructuredBuffer<float> grad_weight; // [C]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_bias;   // [C]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint c = gid.x * 256 + lid.x;
    if (c >= params.C) return;

    uint group_idx = c / params.cpg;
    float sum_gw = 0.0f;
    float sum_gb = 0.0f;

    for (uint n = 0; n < params.N; n++) {
        uint mg_idx = n * params.G + group_idx;
        float mean_val = mean[mg_idx];
        float rstd_val = rstd[mg_idx];

        uint base = (n * params.C + c) * params.HxW;
        for (uint hw = 0; hw < params.HxW; hw++) {
            float go = grad_output[base + hw];
            if (params.compute_weight != 0) {
                float x_hat = (input[base + hw] - mean_val) * rstd_val;
                sum_gw += go * x_hat;
            }
            if (params.compute_bias != 0) {
                sum_gb += go;
            }
        }
    }

    if (params.compute_weight != 0) {
        grad_weight[c] = sum_gw;
    }
    if (params.compute_bias != 0) {
        grad_bias[c] = sum_gb;
    }
}
