// Batch normalization backward for grad_weight and grad_bias.
// One thread per channel, accumulates over batch + spatial.
//
// grad_weight[c] = sum_n sum_hw (grad_output[n,c,hw] * x_hat[n,c,hw])
// grad_bias[c] = sum_n sum_hw (grad_output[n,c,hw])

struct Params {
    uint C;
    uint N;
    uint HxW;
    uint compute_weight;
    uint compute_bias;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [N, C, HxW]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [N, C, HxW]
[[vk::binding(2)]] StructuredBuffer<float> mean;          // [C]
[[vk::binding(3)]] StructuredBuffer<float> invstd;        // [C]
[[vk::binding(4)]] RWStructuredBuffer<float> grad_weight; // [C]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_bias;   // [C]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint c = gid.x * 256 + lid.x;
    if (c >= params.C) return;

    float mean_val = mean[c];
    float invstd_val = invstd[c];
    float sum_gw = 0.0f;
    float sum_gb = 0.0f;

    for (uint n = 0; n < params.N; n++) {
        uint base = n * params.C * params.HxW + c * params.HxW;
        for (uint hw = 0; hw < params.HxW; hw++) {
            float go = grad_output[base + hw];
            if (params.compute_weight != 0) {
                float x_hat = (input[base + hw] - mean_val) * invstd_val;
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
