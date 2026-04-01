// Fused log_softmax backward for large rows (> 256 elements).
// One workgroup per row, each thread handles multiple elements.
// grad_input[i] = grad_output[i] - exp(output[i]) * sum(grad_output[row])

struct Params {
    uint row_size;
    uint num_rows;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> output;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;

[[vk::push_constant]] Params params;

groupshared float shared_data[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.num_rows) return;

    uint col = tid.x;
    uint row_size = params.row_size;
    uint base = row * row_size;

    // Step 1: Each thread accumulates sum over its strided elements
    float local_sum = 0.0f;
    for (uint i = col; i < row_size; i += 256) {
        local_sum += grad_output[base + i];
    }

    shared_data[col] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction for sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (col < s)
            shared_data[col] += shared_data[col + s];
        GroupMemoryBarrierWithGroupSync();
    }

    float sum_go = shared_data[0];

    // Step 2: Apply grad_input = grad_output - exp(output) * sum
    for (uint i = col; i < row_size; i += 256) {
        float go_val = grad_output[base + i];
        float exp_out = exp(output[base + i]);
        grad_input[base + i] = go_val - exp_out * sum_go;
    }
}
