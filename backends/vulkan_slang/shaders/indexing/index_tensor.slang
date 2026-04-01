// index.Tensor: advanced indexing with one or two index tensors.
//
// Single index: out[i, ...] = input[indices0[i], ...]
//   (same as index_select dim=0)
//
// Two indices (same shape): out[i] = input[indices0[i], indices1[i]]
//   (element selection from 2D tensor)

struct Params {
    uint numel;         // total output elements
    uint num_indices;   // length of index tensor(s)
    uint slice_size;    // elements per selected slice (for single index)
    uint num_cols;      // input.size(1) — only used for two-index mode
    uint mode;          // 0 = single index (row select), 1 = two indices (element select)
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<uint> indices0;
[[vk::binding(2)]] StructuredBuffer<uint> indices1;  // unused in single-index mode
[[vk::binding(3)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    if (params.mode == 0) {
        // Single index: out[i, j] = input[indices0[i], j]
        uint row = tid.x / params.slice_size;
        uint col = tid.x % params.slice_size;
        uint src_row = indices0[row];
        out[tid.x] = input[src_row * params.slice_size + col];
    } else {
        // Two indices: out[i] = input[indices0[i], indices1[i]]
        uint idx0 = indices0[tid.x];
        uint idx1 = indices1[tid.x];
        out[tid.x] = input[idx0 * params.num_cols + idx1];
    }
}
