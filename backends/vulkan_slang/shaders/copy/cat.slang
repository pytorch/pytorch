// Cat (concatenate) two tensors along a dimension.
// out[i] = (coord_in_dim < dim_size_a) ? a[...] : b[...]

struct Params {
    uint numel;        // total output elements
    uint outer_size;   // product of dims before cat dim
    uint inner_size;   // product of dims after cat dim
    uint dim_size_a;   // size of cat dim in tensor a
    uint dim_size_b;   // size of cat dim in tensor b
};

[[vk::binding(0)]] StructuredBuffer<float> a;
[[vk::binding(1)]] StructuredBuffer<float> b;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint total_dim = params.dim_size_a + params.dim_size_b;
    uint outer_inner = total_dim * params.inner_size;
    uint outer = tid.x / outer_inner;
    uint rem = tid.x % outer_inner;
    uint dim_idx = rem / params.inner_size;
    uint inner = rem % params.inner_size;

    if (dim_idx < params.dim_size_a) {
        uint src = outer * (params.dim_size_a * params.inner_size) +
                   dim_idx * params.inner_size + inner;
        out[tid.x] = a[src];
    } else {
        uint b_dim_idx = dim_idx - params.dim_size_a;
        uint src = outer * (params.dim_size_b * params.inner_size) +
                   b_dim_idx * params.inner_size + inner;
        out[tid.x] = b[src];
    }
}
