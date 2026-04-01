// AMP gradient unscale + inf/nan check.
// For each element: check if inf/nan, multiply by inv_scale.
// If any element is inf/nan, write 1.0 to found_inf[0].
// Uses atomicMax-like approach: any thread finding inf/nan sets found_inf to 1.0.

struct Params {
    uint numel;
    float inv_scale;
};

[[vk::binding(0)]] RWStructuredBuffer<float> grad;      // in-place unscale
[[vk::binding(1)]] RWStructuredBuffer<float> found_inf;  // single-element: 0.0 or 1.0
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    float v = grad[tid.x];

    // Check inf/nan: NaN != NaN, and abs(inf) == inf
    bool is_nan = (v != v);
    bool is_inf = (abs(v) == 1.0f / 0.0f);

    if (is_nan || is_inf) {
        found_inf[0] = 1.0f;  // Non-atomic, but idempotent (all writers write 1.0)
    }

    // Unscale
    grad[tid.x] = v * params.inv_scale;
}
