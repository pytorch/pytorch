// Copy bool buffer (uint8 packed as uint32): out[i] = input[i]
// Each thread copies 4 bool bytes packed in one uint32.

struct Params {
    uint num_packed;  // ceil(numel / 4)
};

[[vk::binding(0)]] StructuredBuffer<uint> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.num_packed) return;
    out[tid.x] = input[tid.x];
}
