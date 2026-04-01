// Element-wise remainder: out[i] = a[i] - floor(a[i]/b[i]) * b[i]
// Python-style modulo (result has same sign as divisor)

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> a;
[[vk::binding(1)]] StructuredBuffer<float> b;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float av = a[tid.x];
    float bv = b[tid.x];
    out[tid.x] = av - floor(av / bv) * bv;
}
