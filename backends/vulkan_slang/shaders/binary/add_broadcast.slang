// Broadcast element-wise addition: out[i] = a[i] + b[i % numel_b]
// Handles the common case where b has fewer elements and broadcasts along leading dims.
// a: contiguous [numel_a] elements
// b: contiguous [numel_b] elements where numel_b divides numel_a evenly
// out: contiguous [numel_a] elements

struct Params {
    uint numel_a;
    uint numel_b;
};

[[vk::binding(0)]] StructuredBuffer<float> a;
[[vk::binding(1)]] StructuredBuffer<float> b;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel_a) return;
    out[tid.x] = a[tid.x] + b[tid.x % params.numel_b];
}
