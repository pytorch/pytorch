#ifndef MPSCNNShaders_h
#define MPSCNNShaders_h

static const char* METAL_SHADERS = R"METAL_SHADERS(
#include <metal_stdlib>
using namespace metal;

constant ushort ushort_arg_0[[function_constant(0)]];
constant ushort ushort_arg_1[[function_constant(1)]];
constant ushort ushort_arg_2[[function_constant(2)]];
constant ushort ushort_arg_3[[function_constant(3)]];
constant ushort ushort_arg_4[[function_constant(4)]];
constant ushort ushort_arg_5[[function_constant(5)]];
constant ushort ushort_arg_6[[function_constant(6)]];
constant ushort ushort_arg_7[[function_constant(7)]];
constant ushort ushort_arg_8[[function_constant(8)]];
constant ushort ushort_arg_9[[function_constant(9)]];
constant float float_arg_0 [[function_constant(10)]];
constant float float_arg_1 [[function_constant(11)]];


inline constexpr ushort divRoundUp(ushort x, ushort y) { return (x + (y - 1)) / y; }

kernel void elementwise_add_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    out.write(in0.read(gid) + in1.read(gid), gid);
}

kernel void elementwise_add(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in0.read(gid_, gid.z) + in1.read(gid_, gid.z), gid_, gid.z);
}

kernel void elementwise_sub_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid2{0,0};
    out.write(in0.read(gid) - in1.read(gid2), gid);
}

kernel void elementwise_sub(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid1 = gid.xy;
    ushort2 gid2{0,0};
    out.write(in0.read(gid1, gid.z) - in1.read(gid2, gid.z), gid1, gid.z);
}
kernel void elementwise_mul_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid2{0,0};
    out.write(in0.read(gid) * in1.read(gid2), gid);
}

kernel void elementwise_mul(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid1 = gid.xy;
    ushort2 gid2{0,0};
    out.write(in0.read(gid1, gid.z) * in1.read(gid2, gid.z), gid1, gid.z);
}

kernel void copy_nchw_to_metal(constant float* in[[buffer(0)]],
                               texture2d_array<half, access::write> out[[texture(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    // TODO: are the `else` branches needed?
    // TODO: trick the optimizer for case where C == 4?
#define CHW_TO_CHWP4(idx, n, c_, h, w)                                     \
if ((c_) < C) {                                                          \
trns[idx] = in[n * H * W * C + int(c_) * H * W + int(h) * W + int(w)]; \
} else {                                                                 \
trns[idx] = 0.0h;                                                      \
}
    half4 trns;
    CHW_TO_CHWP4(0, n, c * 4 + 0, gid.y, gid.x);
    CHW_TO_CHWP4(1, n, c * 4 + 1, gid.y, gid.x);
    CHW_TO_CHWP4(2, n, c * 4 + 2, gid.y, gid.x);
    CHW_TO_CHWP4(3, n, c * 4 + 3, gid.y, gid.x);
#undef CHW_TO_CHWP4
    out.write(trns, gid.xy, gid.z);
}

kernel void copy_nchw_to_metal_nonarray(constant float* in[[buffer(0)]],
                                        texture2d<half, access::write> out[[texture(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    half4 trns;
    // TODO: are the `else` branches needed?
    // TODO: trick the optimizer for case where C % 4 == 0?
#define CHW_TO_CHWP4(idx, c, h, w)                        \
if ((c) < C) {                                          \
trns[idx] = in[int(c) * H * W + int(h) * W + int(w)]; \
} else {                                                \
trns[idx] = 0.0h;                                     \
}
    CHW_TO_CHWP4(0, 0, gid.y, gid.x);
    CHW_TO_CHWP4(1, 1, gid.y, gid.x);
    CHW_TO_CHWP4(2, 2, gid.y, gid.x);
    CHW_TO_CHWP4(3, 3, gid.y, gid.x);
#undef CHW_TO_CHWP4
    out.write(trns, gid.xy);
}

kernel void copy_metal_to_nchw(texture2d_array<half, access::read> in[[texture(0)]],
                               device float* out[[buffer(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    half4 cs = in.read(gid.xy, gid.z);
#define CHWP4_TO_CHW(idx, n, c_, h, w)                                    \
if ((c_) < C) {                                                         \
out[n * H * W * C + int(c_) * H * W + int(h) * W + int(w)] = cs[idx]; \
}
    CHWP4_TO_CHW(0, n, c * 4 + 0, gid.y, gid.x);
    CHWP4_TO_CHW(1, n, c * 4 + 1, gid.y, gid.x);
    CHWP4_TO_CHW(2, n, c * 4 + 2, gid.y, gid.x);
    CHWP4_TO_CHW(3, n, c * 4 + 3, gid.y, gid.x);
#undef CHWP4_TO_CHW
}

kernel void copy_metal_to_nchw_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                        device float* out[[buffer(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    half4 cs = in.read(gid.xy);
#define CHWP4_TO_CHW(idx, c, h, w)                       \
if ((c) < C) {                                         \
out[int(c) * H * W + int(h) * W + int(w)] = cs[idx]; \
}
    CHWP4_TO_CHW(0, 0, gid.y, gid.x);
    CHWP4_TO_CHW(1, 1, gid.y, gid.x);
    CHWP4_TO_CHW(2, 2, gid.y, gid.x);
    CHWP4_TO_CHW(3, 3, gid.y, gid.x);
#undef CHWP4_TO_CHW
}

kernel void copy(texture2d_array<half, access::read> in[[texture(0)]],
                 texture2d_array<half, access::write> out[[texture(1)]],
                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in.read(gid_, gid.z), gid_, gid.z);
}

kernel void copy_nonarray(texture2d<half, access::read> in[[texture(0)]],
                          texture2d<half, access::write> out[[texture(1)]],
                          ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    out.write(in.read(gid), gid);
}

kernel void clamp_half4(texture2d_array<half, access::read> in[[texture(0)]],
                 texture2d_array<half, access::write> out[[texture(1)]],
                 constant half* clamp_buf[[buffer(0)]],
                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const half4 min_(clamp_buf[0], clamp_buf[0], clamp_buf[0], clamp_buf[0]);
    const half4 max_(clamp_buf[1], clamp_buf[1], clamp_buf[1], clamp_buf[1]);
    ushort2 gid_ = gid.xy;
    half4 value = in.read(gid_, gid.z);
    half4 clamped = clamp(value, min_, max_);
    out.write(clamped, gid_, gid.z);
}

kernel void clamp_half4_nonarray(texture2d<half, access::read> in[[texture(0)]],
                          texture2d<half, access::write> out[[texture(1)]],
                          constant half* clamp_buf[[buffer(0)]],
                          ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const half4 min_(clamp_buf[0], clamp_buf[0], clamp_buf[0], clamp_buf[0]);
    const half4 max_(clamp_buf[1], clamp_buf[1], clamp_buf[1], clamp_buf[1]);
    half4 value = in.read(gid);
    half4 clamped = clamp(value, min_, max_);
    out.write(clamped, gid);
}

kernel void resize_nearest(texture2d_array<half, access::sample> in[[texture(0)]],
                           texture2d_array<half, access::write> out[[texture(1)]],
                           ushort3 gid[[thread_position_in_grid]]) {
    const ushort oH = ushort_arg_0;
    const ushort oW = ushort_arg_1;
    if (gid.x >= oW || gid.y >= oH) {
        return;
    }
    const float height_scale = float(ushort_arg_2) / 10000;
    const float width_scale = float(ushort_arg_3) / 10000;
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    const int in_y = (int)(gid.y / height_scale);
    const int in_x = (int)(gid.x / width_scale);
    out.write(in.sample(s, float2(in_x, in_y), gid.z), gid.xy, gid.z);
}

kernel void resize_nearest_nonarray(texture2d<half, access::sample> in[[texture(0)]],
                                    texture2d<half, access::write> out[[texture(1)]],
                                    ushort2 gid[[thread_position_in_grid]]) {
    const ushort oH = ushort_arg_0;
    const ushort oW = ushort_arg_1;
    if (gid.x >= oW || gid.y >= oH) {
        return;
    }
    const float height_scale = float(ushort_arg_2) / 10000;
    const float width_scale = float(ushort_arg_3) / 10000;
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    const int in_y = (int)(gid.y / height_scale);
    const int in_x = (int)(gid.x / width_scale);
    out.write(in.sample(s, float2(in_x, in_y)), gid.xy);
}

)METAL_SHADERS";

#endif /* MPSCNNShaders_h */
