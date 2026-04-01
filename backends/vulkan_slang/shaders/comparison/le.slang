// Element-wise less-or-equal: out[i] = (a[i] <= b[i]) ? 1.0 : 0.0

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> a;
[[vk::binding(1)]] StructuredBuffer<float> b;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    out[tid.x] = (a[tid.x] <= b[tid.x]) ? 1.0f : 0.0f;
}
