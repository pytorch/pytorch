#ifndef MPSCNNShaders_h
#define MPSCNNShaders_h

static const char* PT_METAL_SHADERS = R"PT_METAL_SHADERS(
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
constant ushort ushort_arg_10[[function_constant(10)]];
constant ushort ushort_arg_11[[function_constant(11)]];
constant float float_arg_0 [[function_constant(12)]];
constant float float_arg_1 [[function_constant(13)]];

inline constexpr ushort divRoundUp(ushort x, ushort y) { return (x + (y - 1)) / y; }

enum broadcastOp {
    Add,
    Sub,
    Mul,
    Div,
};

void elementwise_broadcast_nonarray(texture2d<half, access::read> in0,
                                   texture2d<half, access::read> in1,
                                   texture2d<half, access::write> out,
                                   ushort2 gid,
                                   broadcastOp op) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 in0_stride = ushort2(in0.get_width() > 1, in0.get_height() > 1);
    ushort2 in1_stride = ushort2(in1.get_width() > 1, in1.get_height() > 1);

    ushort2 gid0 = gid.xy * in0_stride;
    ushort2 gid1 = gid.xy * in1_stride;

    if(op == Add) {
        out.write(in0.read(gid0) + in1.read(gid1), gid);
    } else if(op == Sub) {
        out.write(in0.read(gid0) - in1.read(gid1), gid);
    } else if(op == Mul) {
        out.write(in0.read(gid0) * in1.read(gid1), gid);
    } else if(op == Div) {
        out.write(in0.read(gid0) / in1.read(gid1), gid);
    }
}

void elementwise_broadcast(texture2d_array<half, access::read> in0,
                           texture2d_array<half, access::read> in1,
                           texture2d_array<half, access::write> out,
                           ushort3 gid,
                           broadcastOp op) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }

    ushort2 in0_stride = ushort2(in0.get_width() > 1, in0.get_height() > 1);
    ushort2 in1_stride = ushort2(in1.get_width() > 1, in1.get_height() > 1);

    ushort2 gid0 = gid.xy * in0_stride;
    ushort2 gid1 = gid.xy * in1_stride;

    if(op == Add) {
        out.write(in0.read(gid0, gid.z) + in1.read(gid1, gid.z), gid.xy, gid.z);
    } else if(op == Sub) {
        out.write(in0.read(gid0, gid.z) - in1.read(gid1, gid.z), gid.xy, gid.z);
    } else if(op == Mul) {
        out.write(in0.read(gid0, gid.z) * in1.read(gid1, gid.z), gid.xy, gid.z);
    } else if(op == Div) {
        out.write(in0.read(gid0, gid.z) / in1.read(gid1, gid.z), gid.xy, gid.z);
    }
}

kernel void elementwise_add_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    elementwise_broadcast_nonarray(in0, in1, out, gid, Add);
}

kernel void elementwise_add(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    elementwise_broadcast(in0, in1, out, gid, Add);
}

kernel void elementwise_sub_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    elementwise_broadcast_nonarray(in0, in1, out, gid, Sub);
}

kernel void elementwise_sub(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    elementwise_broadcast(in0, in1, out, gid, Sub);
}

kernel void elementwise_mul_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    elementwise_broadcast_nonarray(in0, in1, out, gid, Mul);
}

kernel void elementwise_mul(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    elementwise_broadcast(in0, in1, out, gid, Mul);
}

kernel void elementwise_div_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    elementwise_broadcast_nonarray(in0, in1, out, gid, Div);
}

kernel void elementwise_div(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    elementwise_broadcast(in0, in1, out, gid, Div);
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

kernel void copy_offset(texture2d_array<half, access::read> in[[texture(0)]],
                        texture2d_array<half, access::write> out[[texture(1)]],
                        constant ushort* offset_buf[[buffer(0)]],
                        ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in.read(gid_, gid.z), gid_, gid.z + offset_buf[0]);
}

kernel void copy_offset_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                 texture2d_array<half, access::write> out[[texture(1)]],
                                 constant ushort* offset_buf[[buffer(0)]],
                                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in.read(gid_), gid_, gid.z + offset_buf[0]);
}

kernel void append_features_off0(texture2d_array<half, access::read> in[[texture(0)]],
                                 texture2d_array<half, access::read_write> out[[texture(1)]],
                                 constant ushort* offset_buf[[buffer(0)]],
                                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height() || gid.z >= offset_buf[4]) {
        return;
    }
    ushort2 gid_ = gid.xy;

    ushort batch = gid.z / offset_buf[0];
    ushort feature = gid.z % offset_buf[0];
    ushort outz = batch * offset_buf[1] + offset_buf[2] + feature;
    ushort inz = batch * offset_buf[3] + feature;

    half4 intex1 = in.read(gid_, inz);
    half4 outtex = intex1;

    out.write(outtex, gid_, outz);
}

kernel void append_features_off0_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                          texture2d_array<half, access::read_write> out[[texture(1)]],
                                          constant ushort* offset_buf[[buffer(0)]],
                                          ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    out.write(in.read(gid_), gid_, offset_buf[2]);
}

kernel void append_features_off1(texture2d_array<half, access::read> in[[texture(0)]],
                                 texture2d_array<half, access::read_write> out[[texture(1)]],
                                 constant ushort* offset_buf[[buffer(0)]],
                                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height() || gid.z >= offset_buf[4]) {
        return;
    }
    ushort2 gid_ = gid.xy;

    ushort batch = gid.z / offset_buf[0];
    ushort feature = gid.z % offset_buf[0];
    ushort outz = batch * offset_buf[1] + offset_buf[2] + feature;
    ushort inz = batch * offset_buf[3] + feature;

    half4 outtex = out.read(gid_, outz);
    half4 intex1 = in.read(gid_, inz);
    if (feature == 0) {
      outtex.y = intex1.x;
      outtex.z = intex1.y;
      outtex.w = intex1.z;
      out.write(outtex, gid_, outz);
      return;
    }
    half4 intex0 = in.read(gid_, inz-1);
    outtex.x = intex0.w;
    outtex.y = intex1.x;
    outtex.z = intex1.y;
    outtex.w = intex1.z;

    out.write(outtex, gid_, outz);
}

kernel void append_features_off1_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                          texture2d_array<half, access::read_write> out[[texture(1)]],
                                          constant ushort* offset_buf[[buffer(0)]],
                                          ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;

    ushort feature = gid.z;
    ushort outz = offset_buf[2] + feature;

    half4 outtex = out.read(gid_, outz);
    half4 intex = in.read(gid_);
    if (feature == 0) {
      outtex.y = intex.x;
      outtex.z = intex.y;
      outtex.w = intex.z;
    }
    else {
      outtex.x = intex.w;
    }

    out.write(outtex, gid_, outz);
}

kernel void append_features_off2(texture2d_array<half, access::read> in[[texture(0)]],
                                 texture2d_array<half, access::read_write> out[[texture(1)]],
                                 constant ushort* offset_buf[[buffer(0)]],
                                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height() || gid.z >= offset_buf[4]) {
        return;
    }
    ushort2 gid_ = gid.xy;

    ushort batch = gid.z / offset_buf[0];
    ushort feature = gid.z % offset_buf[0];
    ushort outz = batch * offset_buf[1] + offset_buf[2] + feature;
    ushort inz = batch * offset_buf[3] + feature;

    half4 outtex = out.read(gid_, outz);
    half4 intex1 = in.read(gid_, inz);
    if (feature == 0) {
      outtex.z = intex1.x;
      outtex.w = intex1.y;
      out.write(outtex, gid_, outz);
      return;
    }
    half4 intex0 = in.read(gid_, inz-1);
    outtex.x = intex0.z;
    outtex.y = intex0.w;
    outtex.z = intex1.x;
    outtex.w = intex1.y;

    out.write(outtex, gid_, outz);
}

kernel void append_features_off2_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                          texture2d_array<half, access::read_write> out[[texture(1)]],
                                          constant ushort* offset_buf[[buffer(0)]],
                                          ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;

    ushort feature = gid.z;
    ushort outz = offset_buf[2] + feature;

    half4 outtex = out.read(gid_, outz);
    half4 intex = in.read(gid_);
    if (feature == 0) {
      outtex.z = intex.x;
      outtex.w = intex.y;
    }
    else {
      outtex.x = intex.z;
      outtex.y = intex.w;
    }

    out.write(outtex, gid_, outz);
}

kernel void append_features_off3(texture2d_array<half, access::read> in[[texture(0)]],
                                 texture2d_array<half, access::read_write> out[[texture(1)]],
                                 constant ushort* offset_buf[[buffer(0)]],
                                 ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height() || gid.z >= offset_buf[4]) {
        return;
    }
    ushort2 gid_ = gid.xy;

    ushort batch = gid.z / offset_buf[0];
    ushort feature = gid.z % offset_buf[0];
    ushort outz = batch * offset_buf[1] + offset_buf[2] + feature;
    ushort inz = batch * offset_buf[3] + feature;

    half4 outtex = out.read(gid_, outz);
    half4 intex1 = in.read(gid_, inz);
    if (feature == 0) {
      outtex.w = intex1.x;
      out.write(outtex, gid_, outz);
      return;
    }
    half4 intex0 = in.read(gid_, inz-1);
    outtex.x = intex0.y;
    outtex.y = intex0.z;
    outtex.z = intex0.w;
    outtex.w = intex1.x;

    out.write(outtex, gid_, outz);
}

kernel void append_features_off3_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                          texture2d_array<half, access::read_write> out[[texture(1)]],
                                          constant ushort* offset_buf[[buffer(0)]],
                                          ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;

    ushort feature = gid.z;
    ushort outz = offset_buf[2] + feature;

    half4 outtex = out.read(gid_, outz);
    half4 intex = in.read(gid_);
    if (feature == 0) {
      outtex.w = intex.x;
    }
    else {
      outtex.x = intex.y;
      outtex.y = intex.z;
      outtex.z = intex.w;
    }

    out.write(outtex, gid_, outz);
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

kernel void hardswish(texture2d_array<half, access::read> in[[texture(0)]],
                      texture2d_array<half, access::write> out[[texture(1)]],
                      ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = gid.xy;
    half4 value = in.read(gid_, gid.z);
    half4 mask1 = half4(value < 3.0);
    half4 mask2 = half4(value > -3.0);
    half4 outval = mask2*(mask1*(value*(value + 3.0)/6.0) + (1 - mask1)*value);
    out.write(outval, gid_, gid.z);
}

kernel void hardswish_nonarray(texture2d<half, access::read> in[[texture(0)]],
                               texture2d<half, access::write> out[[texture(1)]],
                               ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    half4 value = in.read(gid);
    half4 mask1 = half4(value < 3);
    half4 mask2 = half4(value > -3.0);
    half4 outval = mask2*(mask1*(value*(value + 3.0)/6.0) + (1 - mask1)*value);
    out.write(outval, gid);
}

constant bool out_is_arr = (ushort_arg_3 > 1 || ushort_arg_2 > 4);
constant bool out_is_tex = !out_is_arr;
constant bool in_is_arr = (ushort_arg_7 > 1 || ushort_arg_6 > 4);
constant bool in_is_tex = !in_is_arr;
kernel void reflection_pad2d(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(in_is_arr)]],
                             texture2d<half, access::read> in_tex[[texture(0),function_constant(in_is_tex)]],
                             texture2d_array<half, access::write> out_arr[[texture(1), function_constant(out_is_arr)]],
                             texture2d<half, access::write> out_tex[[texture(1), function_constant(out_is_tex)]],
                             ushort3 gid[[thread_position_in_grid]]) {
  const ushort H2 = ushort_arg_0;
  const ushort W2 = ushort_arg_1;
  if (gid.x >= W2 || gid.y >= H2) {
      return;
  }

  const ushort pad_left = ushort_arg_8;
  const ushort pad_right = ushort_arg_9;
  const ushort pad_top = ushort_arg_10;
  const ushort pad_bottom = ushort_arg_11;

  const ushort2 out_size = ushort2(W2, H2);
  const ushort xoff_pre  = 2*max(pad_left - gid.x, 0);
  const ushort xoff_post = 2*max(gid.x - (out_size.x - 1 - pad_right), 0);
  const ushort yoff_pre  = 2*max(pad_top - gid.y, 0);
  const ushort yoff_post = 2*max(gid.y - (out_size.y - 1 - pad_bottom), 0);
  ushort2 inpos = ushort2(
      gid.x + xoff_pre - xoff_post - pad_left,
      gid.y + yoff_pre - yoff_post - pad_top);

  half4 intex;
  if (in_is_arr) {
    intex = in_arr.read(inpos, gid.z);
  } else {
    intex = in_tex.read(inpos);
  }

  if (out_is_arr) {
      out_arr.write(intex, gid.xy, gid.z);
  } else {
      out_tex.write(intex, gid.xy);
  }
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

constant bool reshape_out_is_arr = (ushort_arg_3 > 1 || ushort_arg_2 > 4);
constant bool reshape_out_is_tex = !reshape_out_is_arr;
constant bool reshape_in_is_arr = (ushort_arg_7 > 1 || ushort_arg_6 > 4);
constant bool reshape_in_is_tex = !reshape_in_is_arr;
kernel void reshape(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(reshape_in_is_arr)]],
                    texture2d<half, access::read> in_tex[[texture(0),function_constant(reshape_in_is_tex)]],
                    texture2d_array<half, access::write> out_arr[[texture(1), function_constant(reshape_out_is_arr)]],
                    texture2d<half, access::write> out_tex[[texture(1),
                        function_constant(reshape_out_is_tex)]],
                    ushort3 gid[[thread_position_in_grid]]) {
    const ushort H2 = ushort_arg_0;
    const ushort W2 = ushort_arg_1;
    const ushort C2 = ushort_arg_2;
    if (gid.x >= W2 || gid.y >= H2) {
        return;
    }
    const ushort H1 = ushort_arg_4;
    const ushort W1 = ushort_arg_5;
    const ushort C1 = ushort_arg_6;
    const ushort N1 = ushort_arg_7;

    const size_t numel1 = H1 * W1 * C1 * N1;
    const ushort slices2 = divRoundUp(C2, 4);
    const ushort slices1 = divRoundUp(C1, 4);
    const ushort n2 = gid.z / slices2; //image index
    const ushort s2 = gid.z - n2 * slices2; // slice offest
    half4 value;
    for (int idx = 0; idx < 4; ++idx){
        // we compute the "linear index" of the output element,
        // and convert it to the equivalent "linear index" of the input element.
        ushort offset = 4 * s2 + idx;
        size_t linear_idx = n2 * C2 * H2 * W2 + offset * H2 * W2 + gid.y * W2 + gid.x;
        if(linear_idx >= numel1){
            value[idx] = 0;
            continue;
        }
        auto x1 = linear_idx % W1;
        auto y1 = ((int)(linear_idx/W1)) % H1;
        auto s1 = ((int)(linear_idx/W1/H1) % C1);
        auto n1 = ((int)(linear_idx/W1/H1/C1) % N1);
        auto z1 = (int)s1 / 4 + n1 * slices1;
        auto pos = s1 % 4;
        if(reshape_in_is_arr) {
            value[idx] = in_arr.read(ushort2(x1, y1), z1)[pos];
        } else {
            value[idx] = in_tex.read(ushort2(x1, y1))[pos];
        }

    }
    if(reshape_out_is_arr) {
        out_arr.write(value, gid.xy, gid.z);
    } else {
        out_tex.write(value, gid.xy);
    }
}

constant bool transpose_in_is_arr = (ushort_arg_3 > 1 || ushort_arg_4 > 4);
constant bool transpose_in_is_tex = !transpose_in_is_arr;
constant bool transpose_out_is_arr = (ushort_arg_5 > 1 || ushort_arg_6 > 4);
constant bool transpose_out_is_tex = !transpose_out_is_arr;
kernel void transpose(texture2d_array<half, access::read>in_arr[[texture(0),function_constant(transpose_in_is_arr)]],
                      texture2d<half, access::read> in_tex[[texture(0), function_constant(transpose_in_is_tex)]],
                      texture2d_array<half, access::write>out_arr[[texture(1),function_constant(transpose_out_is_arr)]],
                      texture2d<half, access::write> out_tex[[texture(1), function_constant(transpose_out_is_tex)]],
                      constant ushort* inSizeBuffer [[buffer(0)]],
                      constant ushort* outSizeBuffer [[buffer(1)]],
                      ushort3 gid[[thread_position_in_grid]]) {

    const ushort dim0 = ushort_arg_0;
    const ushort dim1 = ushort_arg_1;
    const ushort dim = ushort_arg_2;
    const ushort N1 = ushort_arg_3;
    const ushort C1 = ushort_arg_4;
    const ushort N2 = ushort_arg_5;
    const ushort C2 = ushort_arg_6;
    ushort W1,W2,H1,H2;
    if(transpose_in_is_arr) {
        W1 = in_arr.get_width();
        H1 = in_arr.get_height();
    } else {
        W1 = in_tex.get_width();
        H1 = in_tex.get_height();
    }
    if(transpose_out_is_arr) {
        W2 = out_arr.get_width();
        H2 = out_arr.get_height();
    } else {
        W2 = out_tex.get_width();
        H2 = out_tex.get_height();
    }
    if (gid.x >= W2 || gid.y >= H2) {
        return;
    }
    const size_t numel = H2 * W2 * C2 * N2;
    const ushort slices2 = divRoundUp(C2, 4);
    const ushort slices1 = divRoundUp(C1, 4);
    const ushort n2 = gid.z / slices2;
    const ushort s2 = gid.z - n2 * slices2;
    half4 value;
    ushort4 threadIndexBufferLower{1, 1, 1, 1};
    ushort4 threadIndexBufferUpper{1, 1, 1 ,1};
    for (int idx = 0; idx < 4; ++idx){
        ushort offset = 4 * s2 + idx;
        size_t linear_idx2 = n2 * C2 * H2 * W2 + offset * H2 * W2 + gid.y * W2 + gid.x;
        if(linear_idx2 >= numel) {
            value[idx] = 0;
            continue;
        }

        ushort d2 = 0;
        for(int j = dim-1; j>=0; --j){
            d2  = outSizeBuffer[j];
            if(j > 3) {
                threadIndexBufferUpper[j-3] = linear_idx2 % d2;
            } else {
                threadIndexBufferLower[j] = linear_idx2 % d2;
            }
            linear_idx2 /= d2;
        }

        // swap dims
        ushort tmp;
        if(dim0 > 3) {
            tmp = threadIndexBufferUpper[dim0-3];
        } else {
            tmp = threadIndexBufferLower[dim0];
        }
        if(dim0 > 3 && dim1 > 3) {
            threadIndexBufferUpper[dim0-3] = threadIndexBufferUpper[dim1-3];
        } else if (dim0 > 3 && dim1 < 3) {
            threadIndexBufferUpper[dim0-3] = threadIndexBufferLower[dim1];
        } else if (dim0 < 3 && dim1 > 3) {
            threadIndexBufferLower[dim0] = threadIndexBufferUpper[dim1-3];
        } else {
            threadIndexBufferLower[dim0] = threadIndexBufferLower[dim1];
        }
        if(dim1 > 3) {
            threadIndexBufferUpper[dim1-3] = tmp;
        } else {
            threadIndexBufferLower[dim1] = tmp;
        }

        size_t linear_idx1 = 0;
        ushort m = 1;
        ushort d1 = 0;
        for(int k = dim-1; k>=0; --k) {
            if(k > 3) {
                d1 = threadIndexBufferUpper[k-3];
            } else {
                d1 = threadIndexBufferLower[k];
            }
            linear_idx1 += d1 * m;
            m *= inSizeBuffer[k];
        }

        auto x1 = linear_idx1 % W1;
        auto y1 = ((int)(linear_idx1/W1)) % H1;
        auto c1 = ((int)(linear_idx1/W1/H1) % C1);
        auto n1 = ((int)(linear_idx1/W1/H1/C1) % N1);
        auto z1 = (int)c1 / 4 + n1 * slices1;
        auto pos = c1 % 4;
        if(transpose_in_is_arr) {
            value[idx] = in_arr.read(ushort2(x1, y1), z1)[pos];
        } else {
            value[idx] = in_tex.read(ushort2(x1, y1))[pos];
        }
    }
    if(transpose_out_is_arr) {
        out_arr.write(value, gid.xy, gid.z);
    } else {
        out_tex.write(value, gid.xy);
    }
}

constant bool split_channels_in_is_arr = (ushort_arg_0 > 4);
constant bool split_channels_in_is_tex = !split_channels_in_is_arr;
constant bool split_channels_out1_is_arr = (ushort_arg_1 > 4);
constant bool split_channels_out1_is_tex = !split_channels_out1_is_arr;
constant bool split_channels_out2_is_arr = (ushort_arg_2 > 4);
constant bool split_channels_out2_is_tex = !(split_channels_out2_is_arr);
// A naive implementation to split the input texture into two on channel dimension
kernel void split_channels(texture2d_array<half, access::read> in_arr[[texture(0), function_constant(split_channels_in_is_arr)]],
                           texture2d<half, access::read> in_tex[[texture(0), function_constant(split_channels_in_is_tex)]],
                           texture2d_array<half, access::write> out1_arr[[texture(1),function_constant(split_channels_out1_is_arr)]],
                           texture2d<half, access::write> out1_tex[[texture(1),function_constant(split_channels_out1_is_tex)]],
                           texture2d_array<half, access::write> out2_arr[[texture(2), function_constant(split_channels_out2_is_arr)]],
                           texture2d<half, access::write> out2_tex[[texture(2),function_constant(split_channels_out2_is_tex)]],
                           ushort3 gid[[thread_position_in_grid]]) {
    ushort W,H;
    if(split_channels_in_is_arr) {
        W = in_arr.get_width();
        H = in_arr.get_height();
    } else {
        W = in_tex.get_width();
        H = in_tex.get_height();
    }
    if(gid.x >= W || gid.y >= H){
        return;
    }
    const ushort C1 = ushort_arg_1;
    const ushort s1 = divRoundUp(C1, 4);
    const ushort c_offset = C1 % 4;
    half4 tmp1(0.0, 0.0, 0.0, 0.0);
    half4 tmp2(0.0, 0.0, 0.0, 0.0);
    half4 in41 = split_channels_in_is_arr ? in_arr.read(gid.xy, gid.z) : in_tex.read(gid.xy);
    half4 in42 = split_channels_in_is_arr ? in_arr.read(gid.xy, gid.z+1) : half4(0,0,0,0);
    if(gid.z < s1 - 1) {
        if(split_channels_out1_is_arr) {
            out1_arr.write(in41, gid.xy, gid.z);
        }
    }
    else if(gid.z == s1 - 1) {
        if(c_offset == 0){
            if(split_channels_out1_is_arr) {
                out1_arr.write(in41, gid.xy, gid.z);
            } else {
                out1_tex.write(in41, gid.xy);
            }
            return;
        } else if(c_offset == 1) {
            tmp1.x = in41.x;
            tmp2.xyz = in41.yzw;
            tmp2.w = in42.x;
        } else if (c_offset == 2) {
            tmp1.xy = in41.xy;
            tmp2.xy = in41.zw;
            tmp2.zw = in42.xy;
        } else {
            tmp1.xyz = in41.xyz;
            tmp2.x = in41.w;
            tmp2.yzw = in42.xyz;
        }
        if(split_channels_out1_is_arr) {
            out1_arr.write(tmp1, gid.xy, gid.z);
        } else {
            out1_tex.write(tmp1, gid.xy);
        }
        if(split_channels_out2_is_arr) {
            out2_arr.write(tmp2, gid.xy, 0);
        } else {
            out2_tex.write(tmp2, gid.xy);
        }
    }
    else {
        if (c_offset == 0) {
            if(split_channels_out2_is_arr) {
                out2_arr.write(in41, gid.xy, gid.z - s1);
            } else {
                out2_tex.write(in41, gid.xy);
            }
            return;
        }
        else if (c_offset == 1 ){
            tmp2.xyz = in41.yzw;
            tmp2.w = in42.x;
        } else if (c_offset == 2){
            tmp2.xy = in41.zw;
            tmp2.zw = in42.xy;
        } else {
            tmp2.x = in41.w;
            tmp2.yzw = in42.xyz;
        }
        if(split_channels_out2_is_arr) {
            out2_arr.write(tmp2, gid.xy, gid.z - s1 + 1);
        } else {
            out2_tex.write(tmp2, gid.xy);
        }
    }
}
)PT_METAL_SHADERS";

#endif /* MPSCNNShaders_h */
