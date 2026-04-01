// Layer normalization backward for grad_weight and grad_bias.
// One thread per norm_size element, each accumulates over all rows (batch dim).
//
// grad_weight[d] = sum_n(grad_output[n, d] * x_hat[n, d])
// grad_bias[d] = sum_n(grad_output[n, d])
// where x_hat[n, d] = (input[n, d] - mean[n]) * rstd[n]

struct Params {
    uint num_rows;
    uint norm_size;
    uint compute_weight;  // 1 if grad_weight needed
    uint compute_bias;    // 1 if grad_bias needed
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [num_rows, norm_size]
[[vk::binding(2)]] StructuredBuffer<float> mean;          // [num_rows]
[[vk::binding(3)]] StructuredBuffer<float> rstd;          // [num_rows]
[[vk::binding(4)]] RWStructuredBuffer<float> grad_weight; // [norm_size]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_bias;   // [norm_size]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint d = gid.x * 256 + lid.x;
    if (d >= params.norm_size) return;

    float sum_gw = 0.0f;
    float sum_gb = 0.0f;

    for (uint n = 0; n < params.num_rows; n++) {
        uint idx = n * params.norm_size + d;
        float go = grad_output[idx];
        if (params.compute_weight != 0) {
            float x_hat = (input[idx] - mean[n]) * rstd[n];
            sum_gw += go * x_hat;
        }
        if (params.compute_bias != 0) {
            sum_gb += go;
        }
    }

    if (params.compute_weight != 0) {
        grad_weight[d] = sum_gw;
    }
    if (params.compute_bias != 0) {
        grad_bias[d] = sum_gb;
    }
}
