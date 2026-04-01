// Per-dimension min reduction.
// Input: [num_rows, row_size]
// Output: [num_rows] — one min per row.

groupshared float sdata[256];

struct Params {
    uint num_rows;
    uint row_size;
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

    float acc = 3.402823466e+38f;  // FLT_MAX
    for (uint i = local_id; i < params.row_size; i += 256) {
        acc = min(acc, input[base + i]);
    }

    sdata[local_id] = acc;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s)
            sdata[local_id] = min(sdata[local_id], sdata[local_id + s]);
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        out[row] = sdata[0];
    }
}
