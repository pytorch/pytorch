// Dropout: randomly zero out elements with probability p.
// Uses Philox RNG. mask = (uniform > p), output = input * mask / (1 - p)

struct Params {
    uint numel;
    uint seed_lo;
    uint seed_hi;
    uint offset;
    float p;          // dropout probability
    float scale;      // 1.0 / (1.0 - p)
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

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

uint2 philox_round(uint2 ctr, uint2 key) {
    uint lo = ctr.x;
    uint hi = ctr.y;
    uint product_hi = mulhi32(lo, 0xD2511F53u);
    uint product_lo = lo * 0xD2511F53u;
    ctr.x = hi ^ key.x ^ product_hi;
    ctr.y = product_lo;
    return ctr;
}

uint2 philox_bumpkey(uint2 key) {
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
    return key;
}

float philox_uniform(uint counter, uint seed_lo, uint seed_hi) {
    uint2 ctr = uint2(counter, 2);
    uint2 key = uint2(seed_lo, seed_hi);
    for (int i = 0; i < 10; i++) {
        ctr = philox_round(ctr, key);
        key = philox_bumpkey(key);
    }
    return float(ctr.x) / 4294967296.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float u = philox_uniform(tid.x + params.offset, params.seed_lo, params.seed_hi);
    float mask = (u >= params.p) ? 1.0f : 0.0f;
    out[tid.x] = input[tid.x] * mask * params.scale;
}
