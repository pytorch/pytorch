// Strided per-dimension sum reduction.
// Reduces along an arbitrary axis of an N-D tensor without requiring physical permutation.
//
// For tensor with shape [..., outer, dim_size, inner, ...] summed over dim_size:
//   num_rows  = outer * inner
//   dim_size  = number of elements to sum
//   outer_stride = stride of the outer block in the flat buffer
//   dim_stride   = stride along the reduction dimension
//   inner_stride = stride of the innermost (fastest-varying) index = 1 for contiguous
//
// For contiguous [d0,...,d(k-1), dk, d(k+1),...,dn] summing over dk:
//   outer = d0*...*d(k-1)
//   inner = d(k+1)*...*dn
//   outer_stride = dk * inner
//   dim_stride   = inner
//   inner_stride = 1
//
// Each workgroup computes one output element (one row of the reduction).
// Thread row = gid.x, decomposed as: outer_idx = row / inner, inner_idx = row % inner.
// input index = outer_idx * outer_stride + j * dim_stride + inner_idx
// (for j = 0..dim_size-1)

groupshared float sdata[256];

struct Params {
    uint outer;       // number of outer blocks
    uint dim_size;    // number of elements to sum along
    uint inner;       // number of inner elements
    uint outer_stride; // input stride for the outer dimension
    uint dim_stride;   // input stride along the reduction dimension
    uint inner_stride; // input stride for inner elements (1 for contiguous)
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint row = gid.x;
    uint num_rows = params.outer * params.inner;
    if (row >= num_rows) return;

    uint local_id = lid.x;
    uint outer_idx = row / params.inner;
    uint inner_idx = row % params.inner;
    uint base = outer_idx * params.outer_stride + inner_idx * params.inner_stride;

    // Each thread accumulates multiple elements (for dim_size > 256)
    float acc = 0.0f;
    for (uint j = local_id; j < params.dim_size; j += 256) {
        acc += input[base + j * params.dim_stride];
    }

    sdata[local_id] = acc;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s)
            sdata[local_id] += sdata[local_id + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        out[row] = sdata[0];
    }
}
