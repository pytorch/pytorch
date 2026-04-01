// Generate normal (Gaussian) random floats using Box-Muller transform.
// Requires two uniform samples to produce two normal samples.
// Uses Philox CBRNG internally.

struct Params {
    uint numel;
    uint seed_lo;
    uint seed_hi;
    uint offset;
};

[[vk::binding(0)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

static const float PI = 3.14159265358979323846f;

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

float2 philox_uniform2(uint counter, uint seed_lo, uint seed_hi) {
    uint2 ctr = uint2(counter, 1);  // different stream
    uint2 key = uint2(seed_lo, seed_hi);
    for (int i = 0; i < 10; i++) {
        ctr = philox_round(ctr, key);
        key = philox_bumpkey(key);
    }
    float u1 = (float(ctr.x) + 1.0f) / 4294967297.0f;  // (0, 1)
    float u2 = float(ctr.y) / 4294967296.0f;
    return float2(u1, u2);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    float2 u = philox_uniform2(tid.x + params.offset, params.seed_lo, params.seed_hi);
    // Box-Muller transform
    float r = sqrt(-2.0f * log(u.x));
    float theta = 2.0f * PI * u.y;
    out[tid.x] = r * cos(theta);
}
