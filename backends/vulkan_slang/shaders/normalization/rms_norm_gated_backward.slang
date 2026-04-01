// RMSNormGated backward: grad_input and grad_gate given grad_output.
// forward: out = (input * rstd * weight) * silu(gate)
// dL/d(input[d])  = grad_output[d] * weight[d] * silu(gate[d]) * (rstd - input[d]*rstd^3 * dot(input, input*weight*silu_g*rstd^2) / norm_size)
//                 — simplified as in standard RMSNorm backward + chain rule through gating
// dL/d(gate[d])   = grad_output[d] * (input[d]*rstd*weight[d]) * dsilu(gate[d])
//   where dsilu(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
//
// This uses 2 passes: pass 1 computes grad_input, pass 2 computes grad_weight.
// For simplicity, we combine grad_input and grad_gate in a single pass since they share rstd.

struct Params {
    uint num_rows;
    uint norm_size;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output; // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> input;       // [num_rows, norm_size]
[[vk::binding(2)]] StructuredBuffer<float> gate;        // [num_rows, norm_size]
[[vk::binding(3)]] StructuredBuffer<float> weight;      // [norm_size]
[[vk::binding(4)]] StructuredBuffer<float> rstd;        // [num_rows]
[[vk::binding(5)]] RWStructuredBuffer<float> grad_input;// [num_rows, norm_size]
[[vk::binding(6)]] RWStructuredBuffer<float> grad_gate; // [num_rows, norm_size]
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

    // Compute dot: sum_d(grad_output[d] * silu(gate[d]) * weight[d] * input[d])
    // This raw dot (no rstd factor) appears in the RMSNorm backward correction term.
    // Correct formula: grad_in[d] = go[d]*w[d]*silu_g[d]*rstd - x[d]*rstd^3/N * dot
    float partial_dot = 0.0f;
    for (uint d = tid; d < params.norm_size; d += 256) {
        float go = grad_output[base + d];
        float g = gate[base + d];
        float sig_g = 1.0f / (1.0f + exp(-g));
        float silu_g = g * sig_g;
        partial_dot += go * silu_g * weight[d] * input[base + d];
    }
    smem[tid] = partial_dot;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float dot_val = smem[0];
    float correction = rstd_val * rstd_val * rstd_val / float(params.norm_size) * dot_val;

    // Compute grad_input and grad_gate
    for (uint d = tid; d < params.norm_size; d += 256) {
        float go = grad_output[base + d];
        float g = gate[base + d];
        float sig_g = 1.0f / (1.0f + exp(-g));
        float silu_g = g * sig_g;
        float dsilu_g = sig_g * (1.0f + g * (1.0f - sig_g));  // d/dg[silu(g)]
        float w = weight[d];
        float x = input[base + d];

        // grad_input: via RMSNorm backward chain rule + gating
        // dL/dx_d = go_d * w_d * silu_g_d * rstd - x_d * (rstd^3/N) * dot
        float grad_in = go * w * silu_g * rstd_val - x * correction;
        grad_input[base + d] = grad_in;

        // grad_gate: dL/dgate_d = grad_output[d] * (x[d]*rstd*w[d]) * dsilu(gate[d])
        float x_normed = x * rstd_val * w;
        grad_gate[base + d] = go * x_normed * dsilu_g;
    }
}
