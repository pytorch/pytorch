// Cast bfloat16 -> float32 via simple bit shift.
// BFloat16 is the upper 16 bits of float32, so conversion is trivial.
// Input: packed bf16 pairs in StructuredBuffer<uint> (2 x bf16 per uint32).
// Output: float32 in RWStructuredBuffer<float>.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<uint> input;
[[vk::binding(1)]] RWStructuredBuffer<float> output;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    // Support 2D dispatch for tensors > 65535*256 elements.
    uint idx = tid.y * (65535u * 256u) + tid.x;
    if (idx >= params.numel) return;

    uint word_idx = idx >> 1;
    uint word = input[word_idx];
    uint bits = (idx & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16);
    // bf16 -> f32: just shift left 16 bits (bf16 = top 16 bits of f32)
    output[idx] = asfloat(bits << 16);
}
