// grid_sampler_2d: bilinear grid sampling for spatial transformer networks.
//
// Input:  [N, C, H_in, W_in]
// Grid:   [N, H_out, W_out, 2]  — (x, y) coordinates in [-1, 1]
// Output: [N, C, H_out, W_out]
//
// Supports:
//   interpolation_mode: 0 = bilinear (only mode implemented)
//   padding_mode: 0 = zeros, 1 = border, 2 = reflection
//   align_corners: bool

struct Params {
    uint numel;          // N * C * H_out * W_out
    uint N;
    uint C;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint padding_mode;   // 0 = zeros, 1 = border, 2 = reflection
    uint align_corners;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> grid;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

// Denormalize grid coordinate from [-1,1] to pixel space
float denormalize(float coord, uint size, bool align_corners)
{
    if (align_corners) {
        // [-1, 1] -> [0, size-1]
        return ((coord + 1.0f) * 0.5f) * float(size - 1);
    } else {
        // [-1, 1] -> [-0.5, size-0.5]
        return ((coord + 1.0f) * float(size) - 1.0f) * 0.5f;
    }
}

// Apply padding mode to a coordinate
float apply_padding(float coord, uint size, uint padding_mode)
{
    if (padding_mode == 1) {
        // Border: clamp
        return clamp(coord, 0.0f, float(size - 1));
    } else if (padding_mode == 2) {
        // Reflection
        float sz = float(size);
        if (sz <= 1) return 0.0f;
        // Reflect into [0, size-1]
        float x = coord;
        if (x < 0.0f) x = -x;
        float period = 2.0f * (sz - 1.0f);
        x = x - floor(x / period) * period;
        if (x >= sz) x = period - x;
        return clamp(x, 0.0f, sz - 1.0f);
    }
    // Zeros mode: return as-is (we'll check bounds when reading)
    return coord;
}

// Safe read from input with bounds checking (returns 0 for out-of-bounds in zeros mode)
float safe_read(uint base, int h, int w, uint padding_mode)
{
    if (padding_mode == 0) {
        // Zeros mode
        if (h < 0 || h >= (int)params.in_h || w < 0 || w >= (int)params.in_w)
            return 0.0f;
    }
    uint uh = (uint)clamp(h, 0, (int)params.in_h - 1);
    uint uw = (uint)clamp(w, 0, (int)params.in_w - 1);
    return input[base + uh * params.in_w + uw];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    // Decompose: tid.x = n * C * out_h * out_w + c * out_h * out_w + oh * out_w + ow
    uint out_spatial = params.out_h * params.out_w;
    uint nc = tid.x / out_spatial;
    uint spatial_idx = tid.x % out_spatial;
    uint oh = spatial_idx / params.out_w;
    uint ow = spatial_idx % params.out_w;
    uint c = nc % params.C;
    uint n = nc / params.C;

    // Read grid coordinates: grid[n, oh, ow, 0..1]
    uint grid_base = n * params.out_h * params.out_w * 2;
    uint grid_idx = grid_base + oh * params.out_w * 2 + ow * 2;
    float gx = grid[grid_idx];      // x coordinate
    float gy = grid[grid_idx + 1];  // y coordinate

    bool ac = (params.align_corners != 0);

    // Denormalize to pixel space
    float ix = denormalize(gx, params.in_w, ac);
    float iy = denormalize(gy, params.in_h, ac);

    // Apply padding mode
    if (params.padding_mode != 0) {
        ix = apply_padding(ix, params.in_w, params.padding_mode);
        iy = apply_padding(iy, params.in_h, params.padding_mode);
    }

    // Bilinear interpolation
    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float fx = ix - float(ix0);
    float fy = iy - float(iy0);

    uint in_base = (n * params.C + c) * params.in_h * params.in_w;

    float v00 = safe_read(in_base, iy0, ix0, params.padding_mode);
    float v01 = safe_read(in_base, iy0, ix1, params.padding_mode);
    float v10 = safe_read(in_base, iy1, ix0, params.padding_mode);
    float v11 = safe_read(in_base, iy1, ix1, params.padding_mode);

    float value = v00 * (1.0f - fy) * (1.0f - fx)
                + v01 * (1.0f - fy) * fx
                + v10 * fy * (1.0f - fx)
                + v11 * fy * fx;

    out[tid.x] = value;
}
