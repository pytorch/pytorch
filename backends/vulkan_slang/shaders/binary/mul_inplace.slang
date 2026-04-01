// In-place element-wise multiplication: self[i] *= other[i]

struct Params {
    uint numel;
};

[[vk::binding(0)]] RWStructuredBuffer<float> self_buf;
[[vk::binding(1)]] StructuredBuffer<float> other;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    self_buf[tid.x] = self_buf[tid.x] * other[tid.x];
}
