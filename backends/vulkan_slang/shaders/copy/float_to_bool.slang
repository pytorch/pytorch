// Convert float 0.0/1.0 buffer to packed bool (uint8) buffer.
// Each thread packs 4 float values into 1 uint32 (4 bytes = 4 bools).
// The uint32 is written in little-endian byte order.

struct Params {
    uint numel;       // total number of bool elements
    uint num_packed;  // ceil(numel / 4) — number of uint32 writes
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.num_packed) return;

    uint base = tid.x * 4;
    uint packed = 0;

    // Pack up to 4 bool bytes into one uint32
    for (uint i = 0; i < 4; i++) {
        uint idx = base + i;
        if (idx < params.numel) {
            uint byte_val = (input[idx] != 0.0f) ? 1u : 0u;
            packed |= (byte_val << (i * 8));
        }
    }

    out[tid.x] = packed;
}
