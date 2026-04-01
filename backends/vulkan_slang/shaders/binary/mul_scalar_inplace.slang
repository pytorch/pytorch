// In-place scalar multiplication: self[i] *= scalar

struct Params {
    uint numel;
    float scalar;
};

[[vk::binding(0)]] RWStructuredBuffer<float> self_buf;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    self_buf[tid.x] = self_buf[tid.x] * params.scalar;
}
