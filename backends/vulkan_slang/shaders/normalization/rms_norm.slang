// RMSNorm: out[n,d] = weight[d] * (input[n,d] / sqrt(mean(input[n,:]^2) + eps))
// One workgroup per row — each thread cooperates on the reduction.
// Uses groupshared memory for the variance computation.

struct Params {
    uint num_rows;    // N (batch * seq_len)
    uint norm_size;   // D (hidden_size / normalized_shape)
    float eps;        // variance epsilon (1e-6)
};

[[vk::binding(0)]] StructuredBuffer<float> input;    // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> weight;   // [norm_size]
[[vk::binding(2)]] RWStructuredBuffer<float> out;     // [num_rows, norm_size]
[[vk::binding(3)]] RWStructuredBuffer<float> rstd;    // [num_rows] — rsqrt(var + eps), saved for backward
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

    // Step 1: Compute sum of squares (partial per thread)
    float partial_sq = 0.0f;
    for (uint d = tid; d < params.norm_size; d += 256) {
        float v = input[base + d];
        partial_sq += v * v;
    }
    smem[tid] = partial_sq;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    // Step 2: Compute rstd = rsqrt(mean_sq + eps)
    float mean_sq = smem[0] / float(params.norm_size);
    float rstd_val = rsqrt(mean_sq + params.eps);

    // Save rstd for backward
    if (tid == 0) rstd[row] = rstd_val;

    // Step 3: Normalize and scale by weight
    for (uint d = tid; d < params.norm_size; d += 256) {
        out[base + d] = input[base + d] * rstd_val * weight[d];
    }
}
