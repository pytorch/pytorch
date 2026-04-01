// Per-row all: one thread per row.
// Returns 1.0 if all elements in the row are nonzero, else 0.0.

struct Params {
    uint num_rows;
    uint row_size;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.num_rows) return;

    uint base = tid.x * params.row_size;
    float result = 1.0f;
    for (uint i = 0; i < params.row_size; i++) {
        if (input[base + i] == 0.0f) {
            result = 0.0f;
            break;
        }
    }
    out[tid.x] = result;
}
