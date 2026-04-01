// Fused layer normalization backward for grad_input.
// One workgroup per row, each thread handles multiple elements via striding.
//
// Given forward: y = (x - mean) * rstd * weight + bias
// Backward for grad_input:
//   x_hat = (x - mean) * rstd
//   go_w = grad_output * weight
//   c1 = (1/D) * sum(go_w)
//   c2 = (1/D) * sum(go_w * x_hat)
//   grad_input = rstd * (go_w - c1 - x_hat * c2)

struct Params {
    uint num_rows;
    uint norm_size;
    uint has_weight;  // 1 if weight is provided, 0 otherwise
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [num_rows, norm_size]
[[vk::binding(2)]] StructuredBuffer<float> mean;          // [num_rows]
[[vk::binding(3)]] StructuredBuffer<float> rstd;          // [num_rows]
[[vk::binding(4)]] StructuredBuffer<float> weight;        // [norm_size] (may be unused)
[[vk::binding(5)]] RWStructuredBuffer<float> grad_input;  // [num_rows, norm_size]
[[vk::push_constant]] Params params;

groupshared float smem[256];
groupshared float smem2[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.num_rows) return;

    uint tid = lid.x;
    uint D = params.norm_size;
    uint base = row * D;
    float mean_val = mean[row];
    float rstd_val = rstd[row];

    // Step 1: Compute c1 = sum(go_w) and c2 = sum(go_w * x_hat) simultaneously
    float partial_c1 = 0.0f;
    float partial_c2 = 0.0f;
    for (uint d = tid; d < D; d += 256) {
        float go = grad_output[base + d];
        float w = (params.has_weight != 0) ? weight[d] : 1.0f;
        float go_w = go * w;
        float x_hat = (input[base + d] - mean_val) * rstd_val;
        partial_c1 += go_w;
        partial_c2 += go_w * x_hat;
    }

    smem[tid] = partial_c1;
    smem2[tid] = partial_c2;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction for both sums
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem2[tid] += smem2[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float c1 = smem[0] / float(D);
    float c2 = smem2[0] / float(D);

    // Step 2: grad_input = rstd * (go_w - c1 - x_hat * c2)
    for (uint d = tid; d < D; d += 256) {
        float go = grad_output[base + d];
        float w = (params.has_weight != 0) ? weight[d] : 1.0f;
        float go_w = go * w;
        float x_hat = (input[base + d] - mean_val) * rstd_val;
        grad_input[base + d] = rstd_val * (go_w - c1 - x_hat * c2);
    }
}
