// Convert int64 indices to int32 on GPU (avoids CPU roundtrip).
// Int64 stored as pairs of uint32 (little-endian: [low, high]).
// We extract just the low 32 bits since indices always fit in int32.

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<uint> input;   // int64 data as pairs of uint32
[[vk::binding(1)]] RWStructuredBuffer<uint> out;    // int32 output
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.numel) return;
    // Each int64 is 2 uint32 words. Low word is at 2*idx.
    out[idx] = input[idx * 2u];
}
