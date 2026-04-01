// Per-row argmax: one thread per row, find index of max element.
// Input is reshaped to (num_rows, row_size) with reduction along last dim.

struct Params {
    uint num_rows;
    uint row_size;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> out;  // output indices (one per row)
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.num_rows) return;

    uint base = tid.x * params.row_size;
    float max_val = input[base];
    uint max_idx = 0;

    for (uint i = 1; i < params.row_size; i++) {
        float val = input[base + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    out[tid.x] = max_idx;
}
