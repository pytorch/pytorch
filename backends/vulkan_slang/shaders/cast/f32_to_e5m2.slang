// Cast float32 -> float8_e5m2 via uint32 bit manipulation.
// Input: float32 in StructuredBuffer<float>.
// Output: packed fp8 quads in RWStructuredBuffer<uint> (4 x e5m2 per uint32).
// Each thread processes a QUAD of f32 values to avoid write races on the uint32 word.
//
// E5M2 layout: [S][EEEEE][MM]  bias=15
// Max: ±57344.0, overflow to inf (0x7C).
// Same exponent structure as fp16.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> output;
[[vk::push_constant]] Params params;

uint f32_to_e5m2_bits(float f) {
    uint bits = asuint(f);
    uint sign = (bits >> 24) & 0x80u;  // sign -> bit 7 of result
    uint abs_bits = bits & 0x7FFFFFFFu;

    // NaN -> NaN (0x7F)
    if (abs_bits > 0x7F800000u) return sign | 0x7Fu;
    // Infinity -> Inf (0x7C)
    if (abs_bits == 0x7F800000u) return sign | 0x7Cu;
    // Overflow (>= 57344 = 0x47600000 after rounding): clamp to inf
    if (abs_bits >= 0x47600000u) return sign | 0x7Cu;
    // Zero
    if (abs_bits == 0u) return sign;

    // Extract f32 exponent and mantissa
    int f32_exp = int((abs_bits >> 23) & 0xFFu) - 127;
    uint f32_mant = abs_bits & 0x7FFFFFu;

    // Subnormal e5m2 range: f32_exp < -14
    if (f32_exp < -14) {
        int shift = -14 - f32_exp;
        uint full_mant = (1u << 23) | f32_mant;
        if (shift >= 24) return sign;  // underflow to zero
        uint shifted = full_mant >> (21 + shift);  // 23 - 2 = 21 base shift
        // Round to nearest even
        uint round_bit = (full_mant >> (20 + shift)) & 1u;
        uint sticky = (full_mant & ((1u << (20 + shift)) - 1u)) != 0u ? 1u : 0u;
        if (round_bit != 0u && (sticky != 0u || (shifted & 1u) != 0u)) shifted++;
        if (shifted > 3u) {
            return sign | 0x04u;  // smallest normal: exp=1, mant=0
        }
        return sign | (shifted & 0x3u);
    }

    // Normal range: rebias exponent
    uint e5m2_exp = uint(f32_exp + 15);
    // Get 2-bit mantissa from 23-bit f32 mantissa with rounding
    uint mant2 = f32_mant >> 21;  // top 2 bits
    uint round_bit = (f32_mant >> 20) & 1u;
    uint sticky = (f32_mant & 0xFFFFFu) != 0u ? 1u : 0u;
    // Round to nearest even
    if (round_bit != 0u && (sticky != 0u || (mant2 & 1u) != 0u)) {
        mant2++;
        if (mant2 >= 4u) {
            mant2 = 0u;
            e5m2_exp++;
            if (e5m2_exp >= 31u) return sign | 0x7Cu;  // overflow to inf
        }
    }

    return sign | (e5m2_exp << 2) | mant2;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    // Each thread processes a quad of f32 values -> one uint32 packed word
    // Support 2D dispatch for tensors > 65535*256*4 elements.
    uint quad_idx = tid.y * (65535u * 256u) + tid.x;
    uint idx0 = quad_idx * 4;
    if (idx0 >= params.numel) return;

    uint byte0 = f32_to_e5m2_bits(input[idx0]);
    uint byte1 = 0u, byte2 = 0u, byte3 = 0u;
    if (idx0 + 1 < params.numel) byte1 = f32_to_e5m2_bits(input[idx0 + 1]);
    if (idx0 + 2 < params.numel) byte2 = f32_to_e5m2_bits(input[idx0 + 2]);
    if (idx0 + 3 < params.numel) byte3 = f32_to_e5m2_bits(input[idx0 + 3]);

    output[quad_idx] = byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24);
}
