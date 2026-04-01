// Fused softmax backward along last dimension.
// grad_input[row, j] = output[row, j] * (grad_output[row, j] - dot(grad_output[row], output[row]))
//
// Each workgroup handles one row. Shared memory for the dot product reduction.
// Row size must be <= 256.

groupshared float sdata[256];

struct Params {
    uint row_size;
    uint num_rows;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> output;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint local_id = lid.x;
    uint S = params.row_size;

    if (row >= params.num_rows) return;

    uint base = row * S;

    // Load grad_output * output for dot product
    float go_val = (local_id < S) ? grad_output[base + local_id] : 0.0f;
    float out_val = (local_id < S) ? output[base + local_id] : 0.0f;

    sdata[local_id] = go_val * out_val;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction for dot product
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s && local_id + s < 256)
            sdata[local_id] += sdata[local_id + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float dot_val = sdata[0];

    // Write result
    if (local_id < S) {
        grad_input[base + local_id] = out_val * (go_val - dot_val);
    }
}
