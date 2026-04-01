// Embedding lookup with bf16 weight: out[i, :] = weight[indices[i], :]
// Weight is bf16 (stored as uint16), output is bf16.
// Each thread handles one output element.
// bf16 is packed: the buffer stores raw uint16 bit patterns.

struct Params {
    uint num_indices;    // number of indices
    uint embedding_dim;  // width of each embedding vector
};

[[vk::binding(0)]] StructuredBuffer<uint> weight_u32; // bf16 packed as uint32 (2 values per uint32)
[[vk::binding(1)]] StructuredBuffer<uint> indices;    // [num_indices] (stored as uint32)
[[vk::binding(2)]] RWStructuredBuffer<uint> out_u32;  // bf16 output packed as uint32
[[vk::push_constant]] Params params;

// Extract bf16 from packed uint32 (2 bf16 per uint32)
// bf16 index i: word index = i/2, bit offset = (i%2)*16
uint16_t read_bf16(StructuredBuffer<uint> buf, uint i) {
    uint word = buf[i / 2];
    if (i % 2 == 0) {
        return (uint16_t)(word & 0xFFFFu);
    } else {
        return (uint16_t)((word >> 16) & 0xFFFFu);
    }
}

void write_bf16(RWStructuredBuffer<uint> buf, uint i, uint16_t val) {
    uint word_idx = i / 2;
    uint word = buf[word_idx];
    if (i % 2 == 0) {
        buf[word_idx] = (word & 0xFFFF0000u) | uint(val);
    } else {
        buf[word_idx] = (word & 0x0000FFFFu) | (uint(val) << 16);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.num_indices * params.embedding_dim;
    if (tid.x >= total) return;

    uint row = tid.x / params.embedding_dim;
    uint col = tid.x % params.embedding_dim;

    uint idx = indices[row];
    uint src_i = idx * params.embedding_dim + col;
    write_bf16(out_u32, tid.x, read_bf16(weight_u32, src_i));
}
