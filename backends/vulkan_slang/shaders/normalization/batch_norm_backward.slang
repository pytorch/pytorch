// Fused batch normalization backward for grad_input (training mode).
// One workgroup per channel, each thread handles multiple (n, hw) elements.
//
// Input is [N, C, HxW]. mean/invstd are [C].
// For each channel c:
//   x_hat[n,c,hw] = (input[n,c,hw] - mean[c]) * invstd[c]
//   go_w[n,c,hw] = grad_output[n,c,hw] * weight[c]
//   c1 = (1/M) * sum_n_hw(go_w)
//   c2 = (1/M) * sum_n_hw(go_w * x_hat)
//   grad_input[n,c,hw] = invstd[c] * (go_w - c1 - x_hat * c2)

struct Params {
    uint C;          // number of channels
    uint M;          // N * HxW (total elements per channel)
    uint N;          // batch size
    uint HxW;        // spatial size
    uint has_weight;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [N, C, HxW]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [N, C, HxW]
[[vk::binding(2)]] StructuredBuffer<float> mean;          // [C]
[[vk::binding(3)]] StructuredBuffer<float> invstd;        // [C]
[[vk::binding(4)]] StructuredBuffer<float> weight;        // [C]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_input;  // [N, C, HxW]
[[vk::push_constant]] Params params;

groupshared float smem[256];
groupshared float smem2[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint c = gid.x;
    if (c >= params.C) return;

    uint tid = lid.x;
    uint M = params.M;
    float mean_val = mean[c];
    float invstd_val = invstd[c];
    float w = (params.has_weight != 0) ? weight[c] : 1.0f;

    // Step 1: Compute c1 = sum(go_w) and c2 = sum(go_w * x_hat)
    // Elements for channel c are at indices: n * C * HxW + c * HxW + hw
    float partial_c1 = 0.0f;
    float partial_c2 = 0.0f;
    for (uint i = tid; i < M; i += 256) {
        uint n = i / params.HxW;
        uint hw = i % params.HxW;
        uint idx = n * params.C * params.HxW + c * params.HxW + hw;
        float go = grad_output[idx];
        float go_w = go * w;
        float x_hat = (input[idx] - mean_val) * invstd_val;
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

    float c1 = smem[0] / float(M);
    float c2 = smem2[0] / float(M);

    // Step 2: Write grad_input
    for (uint i = tid; i < M; i += 256) {
        uint n = i / params.HxW;
        uint hw = i % params.HxW;
        uint idx = n * params.C * params.HxW + c * params.HxW + hw;
        float go = grad_output[idx];
        float go_w = go * w;
        float x_hat = (input[idx] - mean_val) * invstd_val;
        grad_input[idx] = invstd_val * (go_w - c1 - x_hat * c2);
    }
}
