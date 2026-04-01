// Argmax: find index of maximum value
// Each workgroup produces one (max_val, max_idx) pair.

groupshared float sval[256];
groupshared uint sidx[256];

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<uint> out;  // output indices
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

    if (global_id < params.numel) {
        sval[local_id] = input[global_id];
        sidx[local_id] = global_id;
    } else {
        sval[local_id] = -3.402823466e+38f;
        sidx[local_id] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            if (sval[local_id + stride] > sval[local_id]) {
                sval[local_id] = sval[local_id + stride];
                sidx[local_id] = sidx[local_id + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        out[gid.x] = sidx[0];
    }
}
