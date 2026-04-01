// Strided copy: copies elements from a strided source buffer
// to a contiguous destination buffer.
// Supports up to 5 dimensions.
//
// For each linear index i in [0, numel):
//   Decompose i into multi-dimensional indices (d0, d1, d2, d3, d4)
//   Read src at offset = d0*src_stride[0] + d1*src_stride[1] + ...
//   Write dst at offset = i

struct Params {
    uint numel;
    uint ndim;
    // sizes[0..4] and src_strides[0..4], packed
    uint sizes0, sizes1, sizes2, sizes3, sizes4;
    uint strides0, strides1, strides2, strides3, strides4;
};

[[vk::binding(0)]] StructuredBuffer<float> src;
[[vk::binding(1)]] RWStructuredBuffer<float> dst;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= params.numel) return;

    // Decompose linear index into multi-dim indices and compute strided offset
    uint remaining = idx;
    uint src_offset = 0;

    uint sizes[5] = { params.sizes0, params.sizes1, params.sizes2, params.sizes3, params.sizes4 };
    uint strides[5] = { params.strides0, params.strides1, params.strides2, params.strides3, params.strides4 };

    // Compute divisors for each dimension (product of sizes below it)
    // For contiguous dst: idx = d0 * (s1*s2*s3*s4) + d1 * (s2*s3*s4) + ...
    for (uint d = 0; d < params.ndim; d++) {
        // Compute the size of all dimensions after d
        uint divisor = 1;
        for (uint dd = d + 1; dd < params.ndim; dd++) {
            divisor *= sizes[dd];
        }
        uint dim_idx = remaining / divisor;
        remaining = remaining % divisor;
        src_offset += dim_idx * strides[d];
    }

    dst[idx] = src[src_offset];
}
