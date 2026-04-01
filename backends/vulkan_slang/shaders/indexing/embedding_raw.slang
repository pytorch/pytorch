// Embedding lookup: out[i, :] = weight[indices[i], :]
// Works on raw uint32 (2 float16/bfloat16 values per uint32, or 1 float32).
// Used when weight dtype is f16/bf16 to avoid full-weight f32 upcast.
// embedding_dim_u32 = embedding_dim / 2 for f16/bf16 (2 values packed per uint32)
//                  = embedding_dim     for f32 (1 value per uint32)

struct Params {
    uint num_indices;       // number of indices to look up
    uint embedding_dim_u32; // number of uint32 per row (embedding_dim / pack_factor)
};

[[vk::binding(0)]] StructuredBuffer<uint> weight;  // packed: bf16 packs 2 per uint32, f32 is 1 per uint32
[[vk::binding(1)]] StructuredBuffer<uint> indices; // [num_indices] as uint32 (i32 values cast to uint32)
[[vk::binding(2)]] RWStructuredBuffer<uint> out;   // [num_indices, embedding_dim_u32]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.num_indices * params.embedding_dim_u32;
    if (tid.x >= total) return;

    uint row = tid.x / params.embedding_dim_u32;
    uint col = tid.x % params.embedding_dim_u32;

    uint idx = indices[row];
    out[tid.x] = weight[idx * params.embedding_dim_u32 + col];
}
