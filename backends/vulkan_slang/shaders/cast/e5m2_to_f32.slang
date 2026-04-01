// Cast float8_e5m2 -> float32 via uint32 bit manipulation.
// Input: packed fp8 quads in StructuredBuffer<uint> (4 x e5m2 per uint32).
// Output: float32 in RWStructuredBuffer<float>.
//
// E5M2 layout: [S][EEEEE][MM]  bias=15
// Has the same exponent structure as fp16 (5 bits, bias 15).
// Has infinity (exp=31, mant=0) and NaN (exp=31, mant!=0).
// Max representable: ±57344.0

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<uint> input;
[[vk::binding(1)]] RWStructuredBuffer<float> output;
[[vk::push_constant]] Params params;

float e5m2_bits_to_f32(uint h) {
    uint sign = (h & 0x80u) << 24;  // sign bit -> bit 31
    uint exp = (h >> 2) & 0x1Fu;    // 5-bit exponent
    uint mant = h & 0x3u;           // 2-bit mantissa

    if (exp == 0u) {
        if (mant == 0u) return asfloat(sign); // +/- zero
        // Subnormal: normalize
        while ((mant & 0x4u) == 0u) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x3u;
    } else if (exp == 31u) {
        if (mant == 0u) return asfloat(sign | 0x7F800000u); // infinity
        return asfloat(sign | 0x7FC00000u); // NaN
    }
    // Exponent bias conversion: f32_exp = e5m2_exp + (127 - 15) = e5m2_exp + 112
    // Mantissa: 2 bits -> 23 bits (shift left by 21)
    return asfloat(sign | ((exp + 112u) << 23) | (mant << 21));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    // Support 2D dispatch for tensors > 65535*256 elements.
    uint idx = tid.y * (65535u * 256u) + tid.x;
    if (idx >= params.numel) return;

    uint word_idx = idx >> 2;           // 4 fp8 values per uint32
    uint byte_pos = idx & 3u;           // which byte within the word
    uint word = input[word_idx];
    uint bits = (word >> (byte_pos * 8u)) & 0xFFu;
    output[idx] = e5m2_bits_to_f32(bits);
}
