// Cumulative product along flattened data.
// Dispatched per-row: each workgroup handles one row of data.
// For simplicity, uses sequential scan within each thread for a row.

struct Params {
    uint num_rows;
    uint row_size;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.num_rows) return;

    uint base = tid.x * params.row_size;
    float accum = 1.0f;
    for (uint i = 0; i < params.row_size; i++) {
        accum *= input[base + i];
        out[base + i] = accum;
    }
}
