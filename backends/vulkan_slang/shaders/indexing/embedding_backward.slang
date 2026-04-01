// Embedding backward pass
// Accumulates grad_output into grad_weight at the rows specified by indices.
// Each thread handles one element of grad_output.

struct Params {
    uint num_indices;
    uint embedding_dim;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;    // [num_indices, embedding_dim]
[[vk::binding(1)]] StructuredBuffer<float> indices;        // [num_indices] (uint packed as float)
[[vk::binding(2)]] RWStructuredBuffer<float> grad_weight;  // [num_embeddings, embedding_dim]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.num_indices * params.embedding_dim;
    if (tid.x >= total) return;

    uint d = tid.x % params.embedding_dim;
    uint i = tid.x / params.embedding_dim;

    uint row = asuint(indices[i]);
    uint dst_idx = row * params.embedding_dim + d;

    // Accumulate gradient (grad_weight should be zeroed before dispatch)
    grad_weight[dst_idx] = grad_weight[dst_idx] + grad_output[tid.x];
}
