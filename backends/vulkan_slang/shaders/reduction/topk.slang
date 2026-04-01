// Per-row top-k: one thread per row.
// Finds k largest (or smallest) values and their indices.
// Uses partial insertion sort — maintains sorted buffer of k elements.

struct Params {
    uint num_rows;
    uint row_size;
    uint k;
    uint largest;  // 1 = largest, 0 = smallest
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
    uint base_out = tid.x * params.k;

    // Initialize with first k elements
    for (uint i = 0; i < params.k; i++) {
        out_values[base_out + i] = input[base_in + i];
        out_indices[base_out + i] = i;
    }

    // Sort initial k elements (insertion sort)
    for (uint i = 1; i < params.k; i++) {
        float key = out_values[base_out + i];
        uint key_idx = out_indices[base_out + i];
        int j = int(i) - 1;
        while (j >= 0) {
            bool should_swap;
            if (params.largest != 0) {
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

    // Scan remaining elements
    for (uint i = params.k; i < params.row_size; i++) {
        float val = input[base_in + i];
        // Check if this value should replace the worst in our buffer
        float worst = out_values[base_out + params.k - 1];
        bool should_insert;
        if (params.largest != 0) {
            should_insert = val > worst;
        } else {
            should_insert = val < worst;
        }
        if (!should_insert) continue;

        // Insert into sorted position
        int j = int(params.k) - 2;
        while (j >= 0) {
            bool should_shift;
            if (params.largest != 0) {
                should_shift = out_values[base_out + uint(j)] < val;
            } else {
                should_shift = out_values[base_out + uint(j)] > val;
            }
            if (!should_shift) break;
            out_values[base_out + uint(j + 1)] = out_values[base_out + uint(j)];
            out_indices[base_out + uint(j + 1)] = out_indices[base_out + uint(j)];
            j--;
        }
        out_values[base_out + uint(j + 1)] = val;
        out_indices[base_out + uint(j + 1)] = i;
    }
}
