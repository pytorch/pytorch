// Log-softmax along the last dimension
// log_softmax(x) = x - max - log(sum(exp(x - max)))

groupshared float sdata[256];

struct Params {
    uint inner_size;
    uint num_rows;
};

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
    uint row = gid.x;
    uint local_id = lid.x;
    uint inner = params.inner_size;

    if (row >= params.num_rows) return;

    uint base = row * inner;

    // Load
    float val = (local_id < inner) ? input[base + local_id] : -3.402823466e+38f;
    sdata[local_id] = val;
    GroupMemoryBarrierWithGroupSync();

    // Max reduction
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride && local_id + stride < 256) {
            sdata[local_id] = max(sdata[local_id], sdata[local_id + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float row_max = sdata[0];
    GroupMemoryBarrierWithGroupSync();

    // Exp(x - max) and sum
    float exp_val = (local_id < inner) ? exp(val - row_max) : 0.0f;
    sdata[local_id] = exp_val;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride && local_id + stride < 256) {
            sdata[local_id] += sdata[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float log_sum = log(sdata[0]);
    GroupMemoryBarrierWithGroupSync();

    // log_softmax = (x - max) - log(sum)
    if (local_id < inner) {
        out[base + local_id] = (val - row_max) - log_sum;
    }
}
