// Fused backward for add_rms_norm: h_new = residual + shortcut; normed = rms_norm(h_new, w)
// Both outputs are used: grad flows through normed AND h_new.
//
// Given grad_normed and grad_h_new (grad through the h_new output):
//   Let c = (1/D) * sum_d(grad_normed[d] * w[d] * h_new[d]) * rstd^2
//   grad_h_new_from_norm[d] = rstd * (grad_normed[d] * w[d] - h_new[d] * c)
//   grad_h_total[d] = grad_h_new_from_norm[d] + grad_h_new[d]
//   grad_residual = grad_shortcut = grad_h_total (add is symmetric)
//
// This fuses rms_norm_backward + add_grad into a single workgroup pass.
// Saves 1 dispatch vs separate rms_norm_backward (pass1) + add.

struct Params {
    uint num_rows;
    uint norm_size;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_normed;   // [num_rows, norm_size] grad through normed output
[[vk::binding(1)]] StructuredBuffer<float> grad_h_new;    // [num_rows, norm_size] grad through h_new output
[[vk::binding(2)]] StructuredBuffer<float> h_new;         // [num_rows, norm_size] saved from forward
[[vk::binding(3)]] StructuredBuffer<float> weight;        // [norm_size]
[[vk::binding(4)]] StructuredBuffer<float> rstd;          // [num_rows]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_h;      // [num_rows, norm_size] = grad_residual = grad_shortcut
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

    // Step 1: Compute c = (1/D) * sum(grad_normed * weight * h_new) * rstd^2
    float partial = 0.0f;
    for (uint d = tid; d < params.norm_size; d += 256) {
        partial += grad_normed[base + d] * weight[d] * h_new[base + d];
    }
    smem[tid] = partial;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    float c = smem[0] * rstd_val * rstd_val / float(params.norm_size);

    // Step 2: grad_h = rms_norm_grad + grad_h_new (fused add)
    for (uint d = tid; d < params.norm_size; d += 256) {
        float go = grad_normed[base + d];
        float w = weight[d];
        float x = h_new[base + d];
        float gnorm = rstd_val * (go * w - x * c);
        grad_h[base + d] = gnorm + grad_h_new[base + d];
    }
}
