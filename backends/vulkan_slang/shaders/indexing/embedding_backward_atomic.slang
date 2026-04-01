// Embedding backward with atomic-safe accumulation for duplicate indices.
// Each thread processes one (index, dim) pair.
// Uses float atomics via asuint/asfloat + InterlockedCompareExchange loop
// to handle duplicate indices correctly without CPU readback.

struct Params {
    uint num_indices;
    uint embedding_dim;
    int  padding_idx;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;    // [num_indices, embedding_dim]
[[vk::binding(1)]] StructuredBuffer<float> indices;        // [num_indices] (uint packed as float)
[[vk::binding(2)]] RWByteAddressBuffer grad_weight;        // [num_embeddings, embedding_dim] as raw bytes

[[vk::push_constant]] Params params;

// Atomic float add using CAS loop on RWByteAddressBuffer
void atomicFloatAdd(uint byte_offset, float value)
{
    uint expected = grad_weight.Load(byte_offset);
    uint desired;
    // CAS loop for float atomic add
    [loop]
    for (;;)
    {
        desired = asuint(asfloat(expected) + value);
        uint original;
        grad_weight.InterlockedCompareExchange(byte_offset, expected, desired, original);
        if (original == expected)
            break;
        expected = original;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.num_indices * params.embedding_dim;
    if (tid.x >= total) return;

    uint d = tid.x % params.embedding_dim;
    uint i = tid.x / params.embedding_dim;

    int row = int(asuint(indices[i]));
    if (row == params.padding_idx) return;

    uint dst_idx = uint(row) * params.embedding_dim + d;
    float val = grad_output[tid.x];

    // Use atomic CAS-based float add for correctness with duplicate indices
    atomicFloatAdd(dst_idx * 4, val);
}
