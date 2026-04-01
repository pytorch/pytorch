// Compute D_i[b,h,q] = dot(grad_out[b,h,q,:], out[b,h,q,:])
// Needed as input to flash_attention_bwd_kv.slang.
// One workgroup per (b, h, q) position.

struct Params {
    uint B;
    uint H;
    uint N;
    uint D;
    uint q_seq_major;  // out_fwd layout: 0=[B,H,N,D] head-major, 1=[B,N,H,D] seq-major
    uint go_seq_major; // grad_out layout: 0=[B,H,N,D] head-major, 1=[B,N,H,D] seq-major
};

[[vk::binding(0)]] StructuredBuffer<float> grad_out; // [B, H, N, D]
[[vk::binding(1)]] StructuredBuffer<float> out_fwd;  // [B, H, N, D]
[[vk::binding(2)]] RWStructuredBuffer<float> D_i;    // [B, H, N]
[[vk::push_constant]] Params params;

groupshared float smem[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint flat  = gid.x;
    uint q_pos = flat % params.N;
    uint h     = (flat / params.N) % params.H;
    uint b     = flat / (params.N * params.H);
    if (b >= params.B) return;

    uint tid  = lid.x;
    // grad_out: head-major when go_seq_major=0, seq-major [B,N,H,D] when go_seq_major=1
    uint go_base;
    if (params.go_seq_major != 0u) {
        go_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        go_base = b * params.H * params.N * params.D + h * params.N * params.D + q_pos * params.D;
    }
    // out_fwd may be head-major or seq-major depending on q_seq_major
    uint out_base;
    if (params.q_seq_major != 0u) {
        out_base = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        out_base = go_base; // same as grad_out base for head-major
    }

    float dot_part = 0.0f;
    for (uint d = tid; d < params.D; d += 256) {
        dot_part += grad_out[go_base + d] * out_fwd[out_base + d];
    }
    smem[tid] = dot_part;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0) {
        D_i[b * params.H * params.N + h * params.N + q_pos] = smem[0];
    }
}
