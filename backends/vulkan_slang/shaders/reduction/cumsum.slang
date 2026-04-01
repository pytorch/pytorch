// Cumulative sum along the last dimension.
// Input/Output: [num_rows, row_size] flattened.
// Each workgroup handles one row. Assumes row_size <= 256.

groupshared float sdata[256];

struct Params {
    uint row_size;
    uint num_rows;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint local_id = lid.x;

    if (row >= params.num_rows) return;

    uint base = row * params.row_size;

    // Load
    sdata[local_id] = (local_id < params.row_size) ? input[base + local_id] : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    // Blelloch scan (up-sweep)
    for (uint d = 1; d < 256; d *= 2) {
        uint ai = local_id;
        float val = 0.0f;
        if (ai >= d && ai < params.row_size) {
            val = sdata[ai - d];
        }
        GroupMemoryBarrierWithGroupSync();
        if (ai >= d && ai < params.row_size) {
            sdata[ai] += val;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write output
    if (local_id < params.row_size) {
        out[base + local_id] = sdata[local_id];
    }
}
