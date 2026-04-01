// All reduction: returns 1.0 if all elements are nonzero, else 0.0
// Each workgroup produces one partial result.

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

    // Out-of-bounds threads are "all true" (identity for AND)
    sdata[local_id] = (global_id < params.numel) ? ((input[global_id] != 0.0f) ? 1.0f : 0.0f) : 1.0f;
    GroupMemoryBarrierWithGroupSync();

    // Reduce with AND (min of 0/1 values)
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            sdata[local_id] = min(sdata[local_id], sdata[local_id + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        out[gid.x] = sdata[0];
    }
}
