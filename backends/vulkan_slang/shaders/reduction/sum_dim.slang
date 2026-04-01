// Per-dimension sum reduction.
// Input: [num_rows, row_size] (target dim moved to last position)
// Output: [num_rows] — one sum per row.
//
// Each workgroup handles one row. Uses shared memory tree reduction.
// Row size must be <= 1024 (each thread handles ceil(row_size/256) elements).

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

    // Each thread accumulates multiple elements (for row_size > 256)
    float acc = 0.0f;
    for (uint i = local_id; i < params.row_size; i += 256) {
        acc += input[base + i];
    }

    sdata[local_id] = acc;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s)
            sdata[local_id] += sdata[local_id + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        out[row] = sdata[0];
    }
}
