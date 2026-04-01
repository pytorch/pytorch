// Cast float32 -> float8_e4m3fn via uint32 bit manipulation.
// Input: float32 in StructuredBuffer<float>.
// Output: packed fp8 quads in RWStructuredBuffer<uint> (4 x e4m3fn per uint32).
// Each thread processes a QUAD of f32 values to avoid write races on the uint32 word.
//
// E4M3FN layout: [S][EEEE][MMM]  bias=7
// Max: ±448.0, saturate to NaN (0x7F) above.
// Min normal: 2^(-6), subnormal range: [2^(-9), 2^(-6))

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> output;
[[vk::push_constant]] Params params;

uint f32_to_e4m3fn_bits(float f) {
    uint bits = asuint(f);
    uint sign = (bits >> 24) & 0x80u;  // sign -> bit 7 of result
    uint abs_bits = bits & 0x7FFFFFFFu;

    // NaN -> NaN (0x7F)
    if (abs_bits > 0x7F800000u) return sign | 0x7Fu;
    // Overflow (>= 448.0 = 0x43E00000): saturate to max (0x7E = 448.0)
    if (abs_bits >= 0x43E00000u) return sign | 0x7Eu;
    // Zero
    if (abs_bits == 0u) return sign;

    // Extract f32 exponent and mantissa
    int f32_exp = int((abs_bits >> 23) & 0xFFu) - 127;
    uint f32_mant = abs_bits & 0x7FFFFFu;

    // Subnormal e4m3fn range: f32_exp < -6
    if (f32_exp < -6) {
        // Shift mantissa right to create subnormal
        // The implicit 1 bit needs to be included
        int shift = -6 - f32_exp;  // how many positions to shift right
        uint full_mant = (1u << 23) | f32_mant;  // include implicit 1
        if (shift >= 24) return sign;  // underflow to zero
        uint shifted = full_mant >> (20 + shift);  // 23 - 3 = 20 base shift
        // Round to nearest even
        uint round_bit = (full_mant >> (19 + shift)) & 1u;
        uint sticky = (full_mant & ((1u << (19 + shift)) - 1u)) != 0u ? 1u : 0u;
        if (round_bit != 0u && (sticky != 0u || (shifted & 1u) != 0u)) shifted++;
        if (shifted > 7u) {
            // Rounded up into normal range
            return sign | 0x08u;  // smallest normal: exp=1, mant=0
        }
        return sign | (shifted & 0x7u);
    }

    // Normal range: rebias exponent
    uint e4m3_exp = uint(f32_exp + 7);
    // Get 3-bit mantissa from 23-bit f32 mantissa with rounding
    uint mant3 = f32_mant >> 20;  // top 3 bits
    uint round_bit = (f32_mant >> 19) & 1u;
    uint sticky = (f32_mant & 0x7FFFFu) != 0u ? 1u : 0u;
    // Round to nearest even
    if (round_bit != 0u && (sticky != 0u || (mant3 & 1u) != 0u)) {
        mant3++;
        if (mant3 >= 8u) {
            mant3 = 0u;
            e4m3_exp++;
            // Check overflow after rounding
            if (e4m3_exp >= 15u) return sign | 0x7Eu;  // saturate to max
        }
    }
    // Check NaN pattern (exp=15, mant=7)
    if (e4m3_exp == 15u && mant3 == 7u) return sign | 0x7Eu;  // saturate to max, not NaN

    return sign | (e4m3_exp << 3) | mant3;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    // Each thread processes a quad of f32 values -> one uint32 packed word
    // Support 2D dispatch for tensors > 65535*256*4 elements.
    uint quad_idx = tid.y * (65535u * 256u) + tid.x;
    uint idx0 = quad_idx * 4;
    if (idx0 >= params.numel) return;

    uint byte0 = f32_to_e4m3fn_bits(input[idx0]);
    uint byte1 = 0u, byte2 = 0u, byte3 = 0u;
    if (idx0 + 1 < params.numel) byte1 = f32_to_e4m3fn_bits(input[idx0 + 1]);
    if (idx0 + 2 < params.numel) byte2 = f32_to_e4m3fn_bits(input[idx0 + 2]);
    if (idx0 + 3 < params.numel) byte3 = f32_to_e4m3fn_bits(input[idx0 + 3]);

    output[quad_idx] = byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24);
}
