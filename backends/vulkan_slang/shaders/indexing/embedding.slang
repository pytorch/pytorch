// Embedding lookup: out[i, :] = weight[indices[i], :]
// Each thread handles one output element.

struct Params {
    uint num_indices;   // number of indices
    uint embedding_dim; // width of each embedding vector
};

[[vk::binding(0)]] StructuredBuffer<float> weight;   // [num_embeddings, embedding_dim]
[[vk::binding(1)]] StructuredBuffer<uint> indices;    // [num_indices] (stored as uint)
[[vk::binding(2)]] RWStructuredBuffer<float> out;     // [num_indices, embedding_dim]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.num_indices * params.embedding_dim;
    if (tid.x >= total) return;

    uint row = tid.x / params.embedding_dim;
    uint col = tid.x % params.embedding_dim;

    uint idx = indices[row];
    out[tid.x] = weight[idx * params.embedding_dim + col];
}
