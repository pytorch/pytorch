// Element-wise fmod: out[i] = fmod(a[i], b[i])

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> a;
[[vk::binding(1)]] StructuredBuffer<float> b;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    out[tid.x] = fmod(a[tid.x], b[tid.x]);
}
