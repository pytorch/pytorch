// Fused NLL Loss Mean: computes mean(-input[target]) in a single pass.
// For N <= 256: single workgroup reduction outputs scalar mean loss directly + count.
// For N > 256: outputs partial sums and counts (one per workgroup), requires second pass.
//
// Input: [N, C] (log-probabilities, float32)
// Target: [N] (int32 packed as float via asfloat/asint)
// Output loss: [num_workgroups] partial loss sums (or out_scalar[0] when single WG)
// Output count: [num_workgroups] partial valid-sample counts
// Output scalar: [1] final mean loss (only written when single workgroup, i.e. N <= 256)
//
// When num_workgroups == 1 (N <= 256), out_scalar[0] = final mean loss (avoids div dispatch).

groupshared float s_loss[256];
groupshared float s_count[256];

struct Params {
    uint N;
    uint C;
    int ignore_index;
    uint single_workgroup;  // 1 if N <= 256 (single WG), 0 otherwise
};

[[vk::binding(0)]] StructuredBuffer<float> input;       // [N, C]
[[vk::binding(1)]] StructuredBuffer<float> target;      // [N] int32 packed as float
[[vk::binding(2)]] RWStructuredBuffer<float> out_loss;  // [num_workgroups] partial sums
[[vk::binding(3)]] RWStructuredBuffer<float> out_count; // [num_workgroups] partial counts
[[vk::binding(4)]] RWStructuredBuffer<float> out_scalar; // [1] final mean (single WG only)
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

    float my_loss = 0.0f;
    float my_count = 0.0f;

    if (global_id < params.N) {
        int t = asint(target[global_id]);
        if (t != params.ignore_index) {
            my_loss = -input[global_id * params.C + uint(t)];
            my_count = 1.0f;
        }
    }

    s_loss[local_id] = my_loss;
    s_count[local_id] = my_count;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            s_loss[local_id] += s_loss[local_id + stride];
            s_count[local_id] += s_count[local_id + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        out_loss[gid.x] = s_loss[0];
        out_count[gid.x] = s_count[0];

        // For single-workgroup case: compute final mean directly (avoids div dispatch)
        // When count == 0 (all targets ignored), return NaN to match PyTorch behavior.
        if (params.single_workgroup != 0) {
            out_scalar[0] = s_loss[0] / s_count[0];  // 0/0 = NaN, which matches CPU behavior
        }
    }
}
