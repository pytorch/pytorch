// L1 Loss: out[i] = |self[i] - target[i]|

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> self;
[[vk::binding(1)]] StructuredBuffer<float> target;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    out[tid.x] = abs(self[tid.x] - target[tid.x]);
}
