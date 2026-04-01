// Fused softmax backward for large rows (> 256 elements).
// One workgroup per row, each thread handles multiple elements.
// grad_input[row, j] = output[row, j] * (grad_output[row, j] - dot(grad_output[row], output[row]))

struct Params {
    uint row_size;
    uint num_rows;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> output;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

groupshared float sdata[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.num_rows) return;

    uint col = tid.x;
    uint row_size = params.row_size;
    uint base = row * row_size;

    // Step 1: Each thread accumulates dot(grad_output, output) over its strided elements
    float local_dot = 0.0f;
    for (uint i = col; i < row_size; i += 256) {
        local_dot += grad_output[base + i] * output[base + i];
    }

    sdata[col] = local_dot;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction for dot product
    for (uint s = 128; s > 0; s >>= 1) {
        if (col < s)
            sdata[col] += sdata[col + s];
        GroupMemoryBarrierWithGroupSync();
    }

    float dot_val = sdata[0];

    // Step 2: Apply grad_input = output * (grad_output - dot)
    for (uint i = col; i < row_size; i += 256) {
        float go_val = grad_output[base + i];
        float out_val = output[base + i];
        grad_input[base + i] = out_val * (go_val - dot_val);
    }
}
