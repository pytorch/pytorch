// Parallel sum reduction using shared memory
// Each workgroup reduces 256 elements, output has one partial sum per workgroup.
// Host must launch multiple passes or do final reduction on CPU for large tensors.

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

    // Load element or 0 if out of range
    sdata[local_id] = (global_id < params.numel) ? input[global_id] : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            sdata[local_id] += sdata[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write workgroup result
    if (local_id == 0) {
        out[gid.x] = sdata[0];
    }
}
