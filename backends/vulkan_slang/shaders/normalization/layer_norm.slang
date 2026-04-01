// Layer normalization over the last `norm_size` elements of each row.
// Each workgroup handles one row. Supports any norm_size (not just <= 256)
// via strided thread access (each thread processes norm_size/256 elements).

groupshared float sdata[256];

struct Params {
    uint norm_size;   // number of elements to normalize over
    uint num_rows;    // outer dimensions flattened
    float eps;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> weight;  // gamma, length = norm_size
[[vk::binding(2)]] StructuredBuffer<float> bias;    // beta, length = norm_size
[[vk::binding(3)]] RWStructuredBuffer<float> out;
[[vk::binding(4)]] RWStructuredBuffer<float> out_mean;  // [num_rows] — per-row mean for backward
[[vk::binding(5)]] RWStructuredBuffer<float> out_rstd;  // [num_rows] — per-row rstd for backward
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint tid = lid.x;
    uint N = params.norm_size;

    if (row >= params.num_rows) return;

    uint base = row * N;

    // Step 1: Compute mean via strided partial sum
    float partial_sum = 0.0f;
    for (uint d = tid; d < N; d += 256) {
        partial_sum += input[base + d];
    }
    sdata[tid] = partial_sum;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction for sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = sdata[0] / float(N);
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Compute variance via strided partial sum of (x - mean)^2
    float partial_var = 0.0f;
    for (uint d = tid; d < N; d += 256) {
        float diff = input[base + d] - mean;
        partial_var += diff * diff;
    }
    sdata[tid] = partial_var;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction for variance
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float var = sdata[0] / float(N);
    float inv_std = rsqrt(var + params.eps);

    // Step 3: Normalize and apply affine transform (strided write)
    for (uint d = tid; d < N; d += 256) {
        float normed = (input[base + d] - mean) * inv_std;
        out[base + d] = normed * weight[d] + bias[d];
    }

    // Write per-row statistics for backward pass (only thread 0 per workgroup)
    if (tid == 0) {
        out_mean[row] = mean;
        out_rstd[row] = inv_std;
    }
}
