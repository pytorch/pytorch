// Cast float32 -> bfloat16 via simple bit shift (truncation).
// BFloat16 is the upper 16 bits of float32.
// Input: float32 in StructuredBuffer<float>.
// Output: packed bf16 pairs in RWStructuredBuffer<uint> (2 x bf16 per uint32).
// Each thread processes a PAIR of elements to avoid write races.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> output;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    // Support 2D dispatch for tensors > 65535*256*2 elements.
    uint pair_idx = tid.y * (65535u * 256u) + tid.x;
    uint idx0 = pair_idx * 2;
    if (idx0 >= params.numel) return;

    // f32 -> bf16: just take upper 16 bits (truncation, no rounding)
    uint lo = asuint(input[idx0]) >> 16;
    uint hi = 0u;
    if (idx0 + 1 < params.numel) {
        hi = asuint(input[idx0 + 1]) >> 16;
    }
    output[pair_idx] = lo | (hi << 16);
}
