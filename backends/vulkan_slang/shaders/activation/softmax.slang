// Softmax along the last dimension (innermost)
// Each workgroup handles one row of length inner_size.
// Assumes inner_size <= 256 (one workgroup per row).
// For larger inner_size, host must tile.

groupshared float sdata[256];

struct Params {
    uint inner_size;  // size of dimension being softmaxed
    uint num_rows;    // number of rows (outer dimensions flattened)
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 tid : SV_DispatchThreadID,
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint local_id = lid.x;
    uint inner = params.inner_size;

    if (row >= params.num_rows) return;

    uint base = row * inner;

    // Step 1: Load element or -inf
    float val = (local_id < inner) ? input[base + local_id] : -3.402823466e+38f;
    sdata[local_id] = val;
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Find max (tree reduction)
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride && local_id + stride < 256) {
            sdata[local_id] = max(sdata[local_id], sdata[local_id + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float row_max = sdata[0];
    GroupMemoryBarrierWithGroupSync();

    // Step 3: Compute exp(x - max)
    float exp_val = (local_id < inner) ? exp(val - row_max) : 0.0f;
    sdata[local_id] = exp_val;
    GroupMemoryBarrierWithGroupSync();

    // Step 4: Sum (tree reduction)
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride && local_id + stride < 256) {
            sdata[local_id] += sdata[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float row_sum = sdata[0];
    GroupMemoryBarrierWithGroupSync();

    // Step 5: Normalize
    if (local_id < inner) {
        out[base + local_id] = exp_val / row_sum;
    }
}
