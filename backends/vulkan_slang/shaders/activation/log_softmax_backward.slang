// Fused log_softmax backward — single workgroup per row.
// grad_input[i] = grad_output[i] - exp(output[i]) * sum(grad_output[row])
// Uses groupshared memory for row-wise sum reduction.

struct Params {
    uint row_size;
    uint num_rows;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [num_rows, row_size]
[[vk::binding(1)]] StructuredBuffer<float> output;       // [num_rows, row_size] (log_softmax output)
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;  // [num_rows, row_size]

[[vk::push_constant]] Params params;

groupshared float shared_data[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint row = gid.x;
    uint col = tid.x;
    uint row_size = params.row_size;
    uint base = row * row_size;

    // Step 1: Compute sum(grad_output[row]) using shared memory reduction
    float go_val = 0.0f;
    if (col < row_size)
        go_val = grad_output[base + col];

    shared_data[col] = go_val;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction for sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (col < s && col + s < row_size)
            shared_data[col] += shared_data[col + s];
        GroupMemoryBarrierWithGroupSync();
    }

    float sum_go = shared_data[0];

    // Step 2: grad_input = grad_output - exp(output) * sum(grad_output)
    if (col < row_size) {
        float exp_out = exp(output[base + col]);
        grad_input[base + col] = go_val - exp_out * sum_go;
    }
}
