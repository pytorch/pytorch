// RMSNorm grad_weight accumulation
// grad_weight[d] = sum_n(grad_output[n,d] * input[n,d] * rstd[n])
// One thread per element in weight (norm_size).

struct Params {
    uint num_rows;
    uint norm_size;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;    // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> input;           // [num_rows, norm_size]
[[vk::binding(2)]] StructuredBuffer<float> rstd;            // [num_rows]
[[vk::binding(3)]] RWStructuredBuffer<float> grad_weight;   // [norm_size]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.norm_size) return;
    uint d = tid.x;

    float acc = 0.0f;
    for (uint n = 0; n < params.num_rows; n++) {
        acc += grad_output[n * params.norm_size + d] * input[n * params.norm_size + d] * rstd[n];
    }
    grad_weight[d] = acc;
}
