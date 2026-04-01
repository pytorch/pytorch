// Cast float32 -> float16 via uint32 bit manipulation.
// Input: float32 in StructuredBuffer<float>.
// Output: packed f16 pairs in RWStructuredBuffer<uint> (2 x f16 per uint32).
// Each thread processes a PAIR of elements to avoid write races on the uint32 word.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> output;
[[vk::push_constant]] Params params;

uint f32_to_f16_bits(float f) {
    uint bits = asuint(f);
    uint sign = (bits >> 16) & 0x8000u;
    int exp = int((bits >> 23) & 0xFFu) - 127 + 15;
    uint mant = bits & 0x7FFFFFu;

    // Handle special cases
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Inf or NaN
        if (mant != 0u) return sign | 0x7E00u; // NaN
        return sign | 0x7C00u; // Inf
    }
    if (exp <= 0) return sign; // Underflow to zero
    if (exp >= 31) return sign | 0x7C00u; // Overflow to inf
    return sign | (uint(exp) << 10) | (mant >> 13);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    // Each thread processes a pair of f32 values -> one uint32 packed word
    // Support 2D dispatch for tensors > 65535*256*2 elements.
    uint pair_idx = tid.y * (65535u * 256u) + tid.x;
    uint idx0 = pair_idx * 2;
    if (idx0 >= params.numel) return;

    uint lo = f32_to_f16_bits(input[idx0]);
    uint hi = 0u;
    if (idx0 + 1 < params.numel) {
        hi = f32_to_f16_bits(input[idx0 + 1]);
    }
    output[pair_idx] = lo | (hi << 16);
}
