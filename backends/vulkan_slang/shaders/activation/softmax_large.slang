// Softmax along the last dimension for large rows (> 256 elements).
// One workgroup per row, each thread handles multiple elements via striding.
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

groupshared float sdata[256];

struct Params {
    uint inner_size;
    uint num_rows;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.num_rows) return;

    uint col = tid.x;
    uint inner = params.inner_size;
    uint base = row * inner;

    // Step 1: Find max via strided access
    float local_max = -3.402823466e+38f;
    for (uint i = col; i < inner; i += 256) {
        local_max = max(local_max, input[base + i]);
    }

    sdata[col] = local_max;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (col < s)
            sdata[col] = max(sdata[col], sdata[col + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float row_max = sdata[0];
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Compute sum(exp(x - max)) via strided access
    float local_sum = 0.0f;
    for (uint i = col; i < inner; i += 256) {
        local_sum += exp(input[base + i] - row_max);
    }

    sdata[col] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (col < s)
            sdata[col] += sdata[col + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float row_sum = sdata[0];

    // Step 3: Write normalized values
    for (uint i = col; i < inner; i += 256) {
        out[base + i] = exp(input[base + i] - row_max) / row_sum;
    }
}
