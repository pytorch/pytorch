// Cast float8_e4m3fn -> float32 via uint32 bit manipulation.
// Input: packed fp8 quads in StructuredBuffer<uint> (4 x e4m3fn per uint32).
// Output: float32 in RWStructuredBuffer<float>.
//
// E4M3FN layout: [S][EEEE][MMM]  bias=7
// No infinity — max exponent+mantissa (0x7F=0.1111.111) is NaN.
// Max representable: ±448.0  (0.1111.110)
// Min normal: 2^(-6) = 0.015625

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<uint> input;
[[vk::binding(1)]] RWStructuredBuffer<float> output;
[[vk::push_constant]] Params params;

float e4m3fn_bits_to_f32(uint h) {
    uint sign = (h & 0x80u) << 24;  // sign bit -> bit 31
    uint exp = (h >> 3) & 0xFu;     // 4-bit exponent
    uint mant = h & 0x7u;           // 3-bit mantissa

    if (exp == 0u) {
        if (mant == 0u) return asfloat(sign); // +/- zero
        // Subnormal: normalize by shifting mantissa until implicit 1 bit
        while ((mant & 0x8u) == 0u) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x7u;
    } else if (exp == 15u && mant == 7u) {
        // NaN (the only NaN pattern in E4M3FN: 0.1111.111 or 1.1111.111)
        return asfloat(sign | 0x7FC00000u);
    }
    // Exponent bias conversion: f32_exp = e4m3_exp + (127 - 7) = e4m3_exp + 120
    // Mantissa: 3 bits -> 23 bits (shift left by 20)
    return asfloat(sign | ((exp + 120u) << 23) | (mant << 20));
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
    output[idx] = e4m3fn_bits_to_f32(bits);
}
