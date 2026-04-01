// RMSNorm backward
// Forward: y = x * rstd * w,  where rstd = 1/sqrt(mean(x^2) + eps)
// Backward:
//   Let c = (1/D) * sum_d(grad_out[d] * w[d] * x[d]) * rstd^2
//   grad_input[d] = rstd * (grad_out[d] * w[d] - x[d] * c)
//   grad_weight[d] += sum_n(grad_out[n,d] * x[n,d] * rstd[n])
// One workgroup per row (like forward).

struct Params {
    uint num_rows;
    uint norm_size;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [num_rows, norm_size]
[[vk::binding(2)]] StructuredBuffer<float> weight;        // [norm_size]
[[vk::binding(3)]] StructuredBuffer<float> rstd;          // [num_rows]
[[vk::binding(4)]] RWStructuredBuffer<float> grad_input;  // [num_rows, norm_size]
[[vk::push_constant]] Params params;

groupshared float smem[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.num_rows) return;

    uint tid = lid.x;
    uint base = row * params.norm_size;
    float rstd_val = rstd[row];

    // Step 1: Compute c = (1/D) * sum(grad_out * weight * input) * rstd^2
    float partial = 0.0f;
    for (uint d = tid; d < params.norm_size; d += 256) {
        partial += grad_output[base + d] * weight[d] * input[base + d];
    }
    smem[tid] = partial;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    float c = smem[0] * rstd_val * rstd_val / float(params.norm_size);

    // Step 2: grad_input[d] = rstd * (grad_out[d] * w[d] - x[d] * c)
    for (uint d = tid; d < params.norm_size; d += 256) {
        float go = grad_output[base + d];
        float w = weight[d];
        float x = input[base + d];
        grad_input[base + d] = rstd_val * (go * w - x * c);
    }
}
