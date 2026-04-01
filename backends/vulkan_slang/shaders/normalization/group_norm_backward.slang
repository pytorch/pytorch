// Fused group normalization backward for grad_input.
// One workgroup per row (N*G rows), each thread handles multiple elements.
//
// Input reshaped as [N*G, group_size] where group_size = cpg * HxW.
// Weight is [C] = [G * cpg], expanded per-spatial-element within each group.
//
// For each row (batch_group_idx):
//   group_idx = batch_group_idx % G
//   For element d in [0, group_size):
//     channel_in_group = d / HxW
//     w[d] = weight[group_idx * cpg + channel_in_group]
//
// Same formula as layer_norm backward:
//   x_hat = (x - mean) * rstd
//   go_w = grad_output * w
//   c1 = (1/D) * sum(go_w)
//   c2 = (1/D) * sum(go_w * x_hat)
//   grad_input = rstd * (go_w - c1 - x_hat * c2)

struct Params {
    uint num_rows;    // N * G
    uint group_size;  // cpg * HxW
    uint num_groups;  // G
    uint cpg;         // channels per group
    uint HxW;         // spatial size
    uint has_weight;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [num_rows, group_size]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [num_rows, group_size]
[[vk::binding(2)]] StructuredBuffer<float> mean;          // [num_rows]
[[vk::binding(3)]] StructuredBuffer<float> rstd;          // [num_rows]
[[vk::binding(4)]] StructuredBuffer<float> weight;        // [C] = [G * cpg]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_input;  // [num_rows, group_size]
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
    uint D = params.group_size;
    uint base = row * D;
    float mean_val = mean[row];
    float rstd_val = rstd[row];

    // Group index for weight lookup
    uint group_idx = row % params.num_groups;
    uint weight_base = group_idx * params.cpg;

    // Step 1: Compute c1 and c2 simultaneously
    float partial_c1 = 0.0f;
    float partial_c2 = 0.0f;
    for (uint d = tid; d < D; d += 256) {
        float go = grad_output[base + d];
        float w = 1.0f;
        if (params.has_weight != 0) {
            uint channel_in_group = d / params.HxW;
            w = weight[weight_base + channel_in_group];
        }
        float go_w = go * w;
        float x_hat = (input[base + d] - mean_val) * rstd_val;
        partial_c1 += go_w;
        partial_c2 += go_w * x_hat;
    }

    smem[tid] = partial_c1;
    smem2[tid] = partial_c2;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem2[tid] += smem2[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float c1 = smem[0] / float(D);
    float c2 = smem2[0] / float(D);

    // Step 2: grad_input
    for (uint d = tid; d < D; d += 256) {
        float go = grad_output[base + d];
        float w = 1.0f;
        if (params.has_weight != 0) {
            uint channel_in_group = d / params.HxW;
            w = weight[weight_base + channel_in_group];
        }
        float go_w = go * w;
        float x_hat = (input[base + d] - mean_val) * rstd_val;
        grad_input[base + d] = rstd_val * (go_w - c1 - x_hat * c2);
    }
}
