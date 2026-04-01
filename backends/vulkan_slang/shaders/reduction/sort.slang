// Per-row insertion sort: one thread per row.
// Sorts along last dim (data reshaped to [num_rows, row_size]).
// Returns sorted values and indices.

struct Params {
    uint num_rows;
    uint row_size;
    uint descending;  // 1 = descending, 0 = ascending
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out_values;
[[vk::binding(2)]] RWStructuredBuffer<uint> out_indices;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.num_rows) return;

    uint base_in = tid.x * params.row_size;
    uint base_out = tid.x * params.row_size;

    // Copy to output and initialize indices
    for (uint i = 0; i < params.row_size; i++) {
        out_values[base_out + i] = input[base_in + i];
        out_indices[base_out + i] = i;
    }

    // Insertion sort
    for (uint i = 1; i < params.row_size; i++) {
        float key = out_values[base_out + i];
        uint key_idx = out_indices[base_out + i];
        int j = int(i) - 1;
        while (j >= 0) {
            bool should_swap;
            if (params.descending != 0) {
                should_swap = out_values[base_out + uint(j)] < key;
            } else {
                should_swap = out_values[base_out + uint(j)] > key;
            }
            if (!should_swap) break;
            out_values[base_out + uint(j + 1)] = out_values[base_out + uint(j)];
            out_indices[base_out + uint(j + 1)] = out_indices[base_out + uint(j)];
            j--;
        }
        out_values[base_out + uint(j + 1)] = key;
        out_indices[base_out + uint(j + 1)] = key_idx;
    }
}
