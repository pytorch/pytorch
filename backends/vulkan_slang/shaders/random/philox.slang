// Philox4x32-10 CBRNG for generating uniform random floats in [0, 1)
// Each thread generates one float using its global ID as the counter.

struct Params {
    uint numel;
    uint seed_lo;    // lower 32 bits of seed
    uint seed_hi;    // upper 32 bits of seed
    uint offset;     // counter offset
};

[[vk::binding(0)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

// 32-bit multiply-high: returns upper 32 bits of a * b without Int64
uint mulhi32(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16;
    uint p0 = a_lo * b_lo;
    uint p1 = a_lo * b_hi;
    uint p2 = a_hi * b_lo;
    uint p3 = a_hi * b_hi;
    uint carry = ((p0 >> 16) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu)) >> 16;
    return p3 + (p1 >> 16) + (p2 >> 16) + carry;
}

// Philox round function
uint2 philox_round(uint2 ctr, uint2 key) {
    uint lo = ctr.x;
    uint hi = ctr.y;
    // Philox S-box constants
    uint product_lo = lo * 0xD2511F53u;
    uint product_hi = mulhi32(lo, 0xD2511F53u);
    ctr.x = hi ^ key.x ^ product_hi;
    ctr.y = product_lo;
    return ctr;
}

// Philox key schedule: bump key
uint2 philox_bumpkey(uint2 key) {
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
    return key;
}

// Full Philox4x32-10 (simplified to 2x32 for single float output)
float philox_uniform(uint counter, uint seed_lo, uint seed_hi) {
    uint2 ctr = uint2(counter, 0);
    uint2 key = uint2(seed_lo, seed_hi);

    // 10 rounds
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key); key = philox_bumpkey(key);
    ctr = philox_round(ctr, key);

    // Convert to float in [0, 1)
    return float(ctr.x) / 4294967296.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = philox_uniform(tid.x + params.offset, params.seed_lo, params.seed_hi);
}
