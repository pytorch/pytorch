// Product reduction via shared memory tree reduction.
// Each workgroup reduces 256 elements. Host does multi-pass for large tensors.

groupshared float sdata[256];

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 tid : SV_DispatchThreadID,
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint local_id = lid.x;
    uint global_id = tid.x;

    // Load element or identity (1.0) if out of range
    sdata[local_id] = (global_id < params.numel) ? input[global_id] : 1.0f;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction with multiplication
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            sdata[local_id] *= sdata[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        out[gid.x] = sdata[0];
    }
}
