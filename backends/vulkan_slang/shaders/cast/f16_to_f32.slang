// Cast float16 -> float32 via uint32 bit manipulation.
// Input: packed f16 pairs in StructuredBuffer<uint> (2 x f16 per uint32).
// Output: float32 in RWStructuredBuffer<float>.
// Works on all Vulkan devices (no VK_KHR_shader_float16_int8 needed).

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<uint> input;
[[vk::binding(1)]] RWStructuredBuffer<float> output;
[[vk::push_constant]] Params params;

float f16_bits_to_f32(uint h) {
    uint sign = (h & 0x8000u) << 16;
    uint exp = (h >> 10) & 0x1Fu;
    uint mant = h & 0x3FFu;

    if (exp == 0u) {
        if (mant == 0u) return asfloat(sign); // +/- zero
        // Denormalized: normalize
        while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x3FFu;
    } else if (exp == 31u) {
        return asfloat(sign | 0x7F800000u | (mant << 13)); // inf/nan
    }
    return asfloat(sign | ((exp + 112u) << 23) | (mant << 13));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    // Support 2D dispatch for tensors > 65535*256 elements:
    // C++ side sets wg_x=65535, wg_y=ceil(total_wg/65535) for large tensors.
    uint idx = tid.y * (65535u * 256u) + tid.x;
    if (idx >= params.numel) return;

    uint word_idx = idx >> 1;
    uint word = input[word_idx];
    uint bits = (idx & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16);
    output[idx] = f16_bits_to_f32(bits);
}
