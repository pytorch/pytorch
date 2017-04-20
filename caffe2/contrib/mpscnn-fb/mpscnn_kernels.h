// Copyright 2004-present Facebook. All Rights Reserved.
// @generated

static const char* MPSCNN_KERNELS = R"V0G0N(

//  Copyright 2004-present Facebook. All Rights Reserved.

#include <metal_stdlib>

using namespace metal;

constant ushort ushort_arg_0[[function_constant(0)]];
constant ushort ushort_arg_1[[function_constant(1)]];
constant ushort ushort_arg_2[[function_constant(2)]];
constant ushort ushort_arg_3[[function_constant(3)]];
constant ushort ushort_arg_4[[function_constant(4)]];
constant ushort ushort_arg_5[[function_constant(5)]];
constant ushort ushort_arg_6[[function_constant(6)]];

inline constexpr ushort divRoundUp(ushort x, ushort y) { return (x + (y - 1)) / y; }

kernel void affine(constant half4* scale[[buffer(0)]],
                   constant half4* shift[[buffer(1)]],
                   texture2d_array<half, access::read> in[[texture(0)]],
                   texture2d_array<half, access::write> out[[texture(1)]],
                   ushort3 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const half4 scale_c = scale[gid.z % divRoundUp(C, 4)];
    const half4 shift_c = shift[gid.z % divRoundUp(C, 4)];
    ushort2 gid_(gid.x, gid.y);
    const half4 x = in.read(gid_, gid.z);
    const half4 y = scale_c * x + shift_c;
    out.write(y, gid_, gid.z);
}

kernel void affine_nonarray(constant half4* scale[[buffer(0)]],
                            constant half4* shift[[buffer(1)]],
                            texture2d<half, access::read> in[[texture(0)]],
                            texture2d<half, access::write> out[[texture(1)]],
                            ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const half4 scale_c = scale[0];
    const half4 shift_c = shift[0];
    half4 x = in.read(gid);
    const half4 y = scale_c * x + shift_c;
    out.write(y, gid);
}

kernel void prelu_nonshared(constant half4* weights[[buffer(0)]],
                            texture2d_array<half, access::read> in[[texture(0)]],
                            texture2d_array<half, access::write> out[[texture(1)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort S = ushort_arg_1;
    const bool channel_shared = S == 1;
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    half4 w = channel_shared ? half4(weights[0][0], weights[0][0], weights[0][0], weights[0][0])
    : weights[gid.z % divRoundUp(C, 4)];
    ushort2 gid_(gid.x, gid.y);
    half4 x = in.read(gid_, gid.z);
    half4 y = select(x * w, x, x > 0.0h);
    out.write(y, gid_, gid.z);
}

kernel void prelu_nonshared_nonarray(constant half4* weights[[buffer(0)]],
                                     texture2d<half, access::read> in[[texture(0)]],
                                     texture2d<half, access::write> out[[texture(1)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    // const ushort C = ushort_arg_0;
    const ushort S = ushort_arg_1;
    const bool channel_shared = S == 1;
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    half4 w = channel_shared ? half4(weights[0][0], weights[0][0], weights[0][0], weights[0][0])
    : weights[0];
    half4 x = in.read(gid);
    half4 y = select(x * w, x, x > 0.0h);
    out.write(y, gid);
}

// One block per texture.
// 256 threads per block.
using AccT = float4;
kernel void instance_norm(constant half* weights[[buffer(0)]],
                          constant half* bias[[buffer(1)]],
                          texture2d_array<half, access::read> in[[texture(0)]],
                          texture2d_array<half, access::write> out[[texture(1)]],
                          ushort3 gid[[thread_position_in_grid]],
                          ushort tid[[thread_index_in_threadgroup]],
                          ushort3 tcount[[threads_per_threadgroup]]) {
    if (gid.z >= out.get_array_size()) {
        return;
    }
    
    constexpr ushort THREADGROUP_SIZE = 256;
    
    threadgroup AccT per_thread_state[THREADGROUP_SIZE];
    // Each block handles a single texture.
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            per_thread_state[tid] += static_cast<AccT>(in.read(ushort2(x, y), gid.z));
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT mean = per_thread_state[0];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT delta = static_cast<AccT>(in.read(ushort2(x, y), gid.z)) - mean;
            per_thread_state[tid] += delta * delta;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = 1.0 / sqrt(max(sum, AccT(1e-5, 1e-5, 1e-5, 1e-5)) + 1.0e-5);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT inv_var = per_thread_state[0];
    
    const AccT c_weights(
                         weights[4 * gid.z], weights[4 * gid.z + 1], weights[4 * gid.z + 2], weights[4 * gid.z + 3]);
    const AccT c_bias(bias[4 * gid.z], bias[4 * gid.z + 1], bias[4 * gid.z + 2], bias[4 * gid.z + 3]);
    
    const AccT scale = inv_var * c_weights;
    const AccT shift = c_bias - mean * scale;
    
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT scaled = static_cast<AccT>(in.read(ushort2(x, y), gid.z)) * scale + shift;
            out.write(static_cast<half4>(scaled), ushort2(x, y), gid.z);
        }
    }
}

// One block per texture.
// 256 threads per block.
kernel void instance_norm_nonarray(constant half* weights[[buffer(0)]],
                                   constant half* bias[[buffer(1)]],
                                   texture2d<half, access::read> in[[texture(0)]],
                                   texture2d<half, access::write> out[[texture(1)]],
                                   ushort3 gid[[thread_position_in_grid]],
                                   ushort tid[[thread_index_in_threadgroup]],
                                   ushort3 tcount[[threads_per_threadgroup]]) {
    constexpr ushort THREADGROUP_SIZE = 256;
    
    threadgroup AccT per_thread_state[THREADGROUP_SIZE];
    // Each block handles a single texture.
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            per_thread_state[tid] += static_cast<AccT>(in.read(ushort2(x, y)));
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT mean = per_thread_state[0];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT delta = static_cast<AccT>(in.read(ushort2(x, y))) - mean;
            per_thread_state[tid] += delta * delta;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = 1.0 / sqrt(max(sum, AccT(1e-5, 1e-5, 1e-5, 1e-5)) + 1.0e-5);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT inv_var = per_thread_state[0];
    
    const AccT c_weights(
                         weights[4 * gid.z], weights[4 * gid.z + 1], weights[4 * gid.z + 2], weights[4 * gid.z + 3]);
    const AccT c_bias(bias[4 * gid.z], bias[4 * gid.z + 1], bias[4 * gid.z + 2], bias[4 * gid.z + 3]);
    
    const AccT scale = inv_var * c_weights;
    const AccT shift = c_bias - mean * scale;
    
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT scaled = static_cast<AccT>(in.read(ushort2(x, y))) * scale + shift;
            out.write(static_cast<half4>(scaled), ushort2(x, y));
        }
    }
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
    
    out.write(trns, ushort2(gid.x, gid.y), gid.z);
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
    
    out.write(trns, ushort2(gid.x, gid.y));
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
    
    half4 cs = in.read(ushort2(gid.x, gid.y), gid.z);
    
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
    
    half4 cs = in.read(ushort2(gid.x, gid.y));
    
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

kernel void convtranspose_upscale(texture2d_array<half, access::read> in[[texture(0)]],
                                  texture2d_array<half, access::write> out[[texture(1)]],
                                  ushort3 gid[[thread_position_in_grid]]) {
    // All resolved at compile time.
    // Assume symmetric kernel/stride/pad for now.
    const ushort kernel_ = ushort_arg_0;
    const ushort stride = ushort_arg_1;
    const ushort pad = ushort_arg_2;
    
    half4 zero(0.0h, 0.0h, 0.0h, 0.0h);
    
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const ushort2 gid_ = ushort2(gid.x, gid.y);
    if (gid.x < kernel_ - 1 - pad || gid.y < kernel_ - 1 - pad) {
        out.write(zero, gid_, gid.z);
        return;
    }
    
    if (((gid.x - (kernel_ - 1 - pad)) % stride == 0) &&
        ((gid.y - (kernel_ - 1 - pad)) % stride == 0)) {
        ushort2 in_pos((gid.x - (kernel_ - 1 - pad)) / stride, (gid.y - (kernel_ - 1 - pad)) / stride);
        
        if (in_pos.x < in.get_width() && in_pos.y < in.get_height()) {
            half4 input = in.read(in_pos, gid.z);
            out.write(input, gid_, gid.z);
        } else {
            out.write(zero, gid_, gid.z);
        }
    } else {
        out.write(zero, gid_, gid.z);
    }
}

constant bool has_in_arr = (ushort_arg_4 > 1 || ushort_arg_0 * ushort_arg_0 * ushort_arg_3 > 4);
constant bool has_out_arr = (ushort_arg_4 > 1 || ushort_arg_3 > 4);
constant bool has_in_tex = (!has_in_arr);
constant bool has_out_tex = (!has_out_arr);

kernel void col2im(
                   texture2d_array<half, access::read> ina[[ texture(0), function_constant(has_in_arr) ]],
                   texture2d<half, access::read> in[[ texture(0), function_constant(has_in_tex) ]],
                   texture2d_array<half, access::write> outa[[ texture(1), function_constant(has_out_arr) ]],
                   texture2d<half, access::write> out[[ texture(1), function_constant(has_out_tex) ]],
                   ushort3 gid[[thread_position_in_grid]]) {
    const ushort kernel_ = ushort_arg_0;
    const ushort stride = ushort_arg_1;
    const ushort pad = ushort_arg_2;
    const ushort C = ushort_arg_3;
    //  const int N = ushort_arg_4;
    const ushort height_col = ushort_arg_5; //(outa.get_height() + pad + pad - kernel_) / stride + 1;
    const ushort width_col = ushort_arg_6; // (outa.get_width() + pad + pad - kernel_) / stride + 1;
    
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    
    const ushort w = gid.x + pad;
    const ushort h = gid.y + pad;
    
    // compute the start and end of the output
    const ushort w_col_start = (w < kernel_) ? 0 : (w - kernel_) / stride + 1;
    const ushort w_col_end = min(ushort(w / stride + 1), ushort(width_col));
    const ushort h_col_start = (h < kernel_) ? 0 : (h - kernel_) / stride + 1;
    const ushort h_col_end = min(ushort(h / stride + 1), ushort(height_col));
    
    half4 val(0.0, 0.0, 0.0, 0.0);
    for (ushort h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (ushort w_col = w_col_start; w_col < w_col_end; ++w_col) {
            const ushort w_k = w - w_col * stride;
            const ushort h_k = h - h_col * stride;
            
            // layout is essentially: [N][K][K][C][H][W]
            // - where the divRoundUp(K * K * C, 4) channels are interleaved as usual.
            // Thus, it's actually [N][divRoundUp(K * K * C, 4)][H][W].
            
            // If C % 4 is not zero, then we have to play some games via partial indexing.
            // TODO: is it worth optimizing this loop via padding in C?
            if (C % 4 == 0) {
                ushort c_col = n * kernel_ * kernel_ * divRoundUp(C, 4) + h_k * kernel_ * divRoundUp(C, 4) + w_k * divRoundUp(C, 4) + c;
                if (has_in_arr) {
                    val += ina.read(ushort2(w_col, h_col), c_col);
                }
                if (has_in_tex) {
                    val += in.read(ushort2(w_col, h_col), c_col);
                }
            } else {
                half4 components(0, 0, 0, 0);
                for (auto i = 0; i < 4; ++i) {
                    ushort c_col_i = n * divRoundUp(kernel_ * kernel_ * C, 4) * 4 + h_k * kernel_ * C + w_k * C + c * 4 + i;
                    ushort c_col_i_z = c_col_i / 4;
                    ushort c_col_i_off = c_col_i - c_col_i_z * 4;
                    if (has_in_arr) {
                        components[i] = ina.read(ushort2(w_col, h_col), c_col_i_z)[c_col_i_off];
                    }
                    if (has_in_tex) {
                        components[i] = in.read(ushort2(w_col, h_col))[c_col_i_off];
                    }
                }
                val += components;
            }
        }
    }
    if (has_out_arr) {
        outa.write(val, ushort2(gid.x, gid.y), gid.z);
    }
    if (has_out_tex) {
        out.write(val, ushort2(gid.x, gid.y));
    }
}

kernel void preprocess_stylizer(device uchar4* in[[buffer(0)]],
                                constant half* mean[[buffer(1)]],
                                constant half4* noise[[buffer(2)]],
                                texture2d<half, access::write> out[[texture(0)]],
                                ushort2 gid[[thread_position_in_grid]]) {
    
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const ushort noise_size = ushort_arg_0;
    
    half4 mean_half(mean[0], mean[1], mean[2], 0.0h);
    uint input_noise_idx = ((uint)out.get_width() * (uint)gid.y + (uint)gid.x) % (noise_size / 4);
    const half4 input_noise = noise[input_noise_idx];
    const uint W = out.get_width();
#define in_at(h, w) in[(uint)(h)*W + (uint)(w)]
    uchar4 input = in_at(gid.y, gid.x);
#undef in_at
    half4 input_half = static_cast<half4>(input);
    out.write(input_half - mean_half + input_noise, gid);
}

kernel void deprocess_stylizer(texture2d<half, access::read> in[[texture(0)]],
                               device uchar4* out[[buffer(0)]],
                               constant half* mean[[buffer(1)]],
                               ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= in.get_width() || gid.y >= in.get_height()) {
        return;
    }
    
    half4 value = in.read(gid);
    
    half4 mean_h(mean[0], mean[1], mean[2], 0.0h);
    half4 min_h(0.0h, 0.0h, 0.0h, 255.0h);
    half4 max_h(255.0h, 255.0h, 255.0h, 255.0h);
    half4 clamped = clamp(value + mean_h, min_h, max_h);
    const uint W = in.get_width();
#define out_at(h, w, v) out[(uint)(h)*W + (uint)(w)] = (v)
    out_at(gid.y, gid.x, static_cast<uchar4>(clamped));
#undef out_at
}

kernel void reflection_padding_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                        texture2d<half, access::write> out[[texture(1)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort H = in.get_height();
    ushort PH = out.get_height();
    
    // Note: we assume symmetric padding on H/W here, which is verified
    // in the calling code.
    ushort pad_h = (PH - H) / 2;
    ushort W = in.get_width();
    ushort PW = out.get_width();
    ushort pad_w = (PW - W) / 2;
    
    short h = short(gid.y) - short(pad_h);
    h = max(h, short(-h));
    h = min(h, short(2 * H - h - 2));
    
    short w = short(gid.x) - short(pad_w);
    w = max(w, short(-w));
    w = min(w, short(2 * W - w - 2));
    
    ushort2 inid(w, h);
    out.write(in.read(inid), gid);
}

kernel void reflection_padding(texture2d_array<half, access::read> in[[texture(0)]],
                               texture2d_array<half, access::write> out[[texture(1)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort H = in.get_height();
    ushort PH = out.get_height();
    
    // Note: we assume symmetric padding on H/W here, which is verified
    // in the calling code.
    ushort pad_h = (PH - H) / 2;
    ushort W = in.get_width();
    ushort PW = out.get_width();
    ushort pad_w = (PW - W) / 2;
    
    short h = short(gid.y) - short(pad_h);
    h = max(h, short(-h));
    h = min(h, short(2 * H - h - 2));
    
    short w = short(gid.x) - short(pad_w);
    w = max(w, short(-w));
    w = min(w, short(2 * W - w - 2));
    
    ushort2 inid(w, h);
    
    out.write(in.read(inid, gid.z), ushort2(gid.x, gid.y), gid.z);
}

kernel void bilinear_upsample(texture2d<half, access::sample> in[[texture(0)]],
                              texture2d<half, access::write> out[[texture(1)]],
                              ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 src = gid / 2;
    constexpr sampler sampler(address::clamp_to_edge, filter::linear, coord::pixel);
    half4 value = in.sample(sampler, static_cast<float2>(src));
    out.write(value, gid);
}

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
    ushort2 gid_ = ushort2(gid.x, gid.y);
    out.write(in0.read(gid_, gid.z) + in1.read(gid_, gid.z), gid_, gid.z);
}

constant bool has_in0_arg = (ushort_arg_0 > 0);
constant bool has_in1_arg = (ushort_arg_1 > 0);
constant bool has_in2_arg = (ushort_arg_2 > 0);
constant bool has_in3_arg = (ushort_arg_3 > 0);

constant bool has_in0_tex = (has_in0_arg && ushort_arg_0 <= 4 && ushort_arg_4 <= 1);
constant bool has_in1_tex = (has_in1_arg && ushort_arg_1 <= 4 && ushort_arg_4 <= 1);
constant bool has_in2_tex = (has_in2_arg && ushort_arg_2 <= 4 && ushort_arg_4 <= 1);
constant bool has_in3_tex = (has_in3_arg && ushort_arg_3 <= 4 && ushort_arg_4 <= 1);

constant bool has_in0_array = (has_in0_arg && !has_in0_tex);
constant bool has_in1_array = (has_in1_arg && !has_in1_tex);
constant bool has_in2_array = (has_in2_arg && !has_in2_tex);
constant bool has_in3_array = (has_in3_arg && !has_in3_tex);

inline ushort idx_3(ushort z, ushort C0, ushort C1, ushort C2, ushort C3) {
    if (z < C0 / 4) {
        return 0;
    }
    if (z < (C0 + C1) / 4) {
        return 1;
    }
    if (z < (C0 + C1 + C2) / 4) {
        return 2;
    }
    return 3;
}

inline ushort idx_2(ushort z, ushort C0, ushort C1, ushort C2) {
    if (z < C0 / 4) {
        return 0;
    }
    if (z < (C0 + C1) / 4) {
        return 1;
    }
    return 2;
}

inline ushort idx_1(ushort z, ushort C0, ushort C1) {
    if (z < C0 / 4) {
        return 0;
    } else {
        return 1;
    }
}

inline ushort idx_0(ushort z, ushort C0) { return 0; }

// in a texture_array with size C, find the offset for image N at plane c.
inline constexpr ushort z_off(ushort n, ushort c, ushort C) { return n * divRoundUp(C, 4) + c; }

kernel void concat(
                   texture2d<half, access::read> in0[[ texture(0), function_constant(has_in0_tex) ]],
                   texture2d<half, access::read> in1[[ texture(1), function_constant(has_in1_tex) ]],
                   texture2d<half, access::read> in2[[ texture(2), function_constant(has_in2_tex) ]],
                   texture2d<half, access::read> in3[[ texture(3), function_constant(has_in3_tex) ]],
                   texture2d_array<half, access::read> ina0[[ texture(0), function_constant(has_in0_array) ]],
                   texture2d_array<half, access::read> ina1[[ texture(1), function_constant(has_in1_array) ]],
                   texture2d_array<half, access::read> ina2[[ texture(2), function_constant(has_in2_array) ]],
                   texture2d_array<half, access::read> ina3[[ texture(3), function_constant(has_in3_array) ]],
                   texture2d_array<half, access::write> out[[texture(5)]],
                   ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    
    const ushort C0 = ushort_arg_0;
    const ushort C1 = ushort_arg_1;
    const ushort C2 = ushort_arg_2;
    const ushort C3 = ushort_arg_3;
    const ushort C = C0 + C1 + C2 + C3;
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    
    ushort idx = 0;
    if (has_in3_arg) {
        idx = idx_3(c, C0, C1, C2, C3);
    } else if (has_in2_arg) {
        idx = idx_2(c, C0, C1, C2);
    } else if (has_in1_arg) {
        idx = idx_1(c, C0, C1);
    } else if (has_in0_arg) {
        idx = idx_0(c, C0);
    } else {
        // never reached.
        idx = 0;
    }
    
    ushort2 gid_ = ushort2(gid.x, gid.y);
    half4 value;
    switch (idx) {
        case 0: {
            if (has_in0_tex) {
                value = in0.read(gid_);
            }
            if (has_in0_array) {
                value = ina0.read(gid_, z_off(n, c, C0));
            }
            break;
        }
        case 1: {
            if (has_in1_tex) {
                value = in1.read(gid_);
            }
            if (has_in1_array) {
                value = ina1.read(gid_, z_off(n, c - (C0) / 4, C1));
            }
            break;
        }
        case 2: {
            if (has_in2_tex) {
                value = in2.read(gid_);
            }
            if (has_in2_array) {
                value = ina2.read(gid_, z_off(n, c - (C0 + C1) / 4, C2));
            }
            break;
        }
        case 3: {
            if (has_in3_tex) {
                value = in3.read(gid_);
            }
            if (has_in3_array) {
                value = ina3.read(gid_, z_off(n, c - (C0 + C1 + C2) / 4, C3));
            }
            break;
        }
    }
    out.write(value, gid_, gid.z);
}

using RoIT = half;
using RoIT4 = half4;
kernel void roi_warp(texture2d_array<half, access::sample> in[[texture(0)]],
                     texture2d_array<half, access::write> out[[texture(1)]],
                     constant half4* rois[[buffer(0)]],
                     ushort3 gid[[thread_position_in_grid]]) {
    constexpr sampler s2(coord::pixel, address::clamp_to_edge, filter::linear);
    
    const half spatial_scale = half(ushort_arg_0) / 10000;
    const ushort sampling_ratio = ushort_arg_1;
    const ushort C = ushort_arg_2;
    const ushort pw = gid.x;
    const ushort ph = gid.y;
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z % divRoundUp(C, 4);
    
    const RoIT4 roi_scaled = rois[n] * spatial_scale;
    const RoIT roi_start_w = roi_scaled[0];
    const RoIT roi_start_h = roi_scaled[1];
    const RoIT roi_end_w = roi_scaled[2];
    const RoIT roi_end_h = roi_scaled[3];
    
    // Force malformed ROIs to be 1x1
    const RoIT roi_width = max(roi_end_w - roi_start_w, (RoIT)1.);
    const RoIT roi_height = max(roi_end_h - roi_start_h, (RoIT)1.);
    
    const RoIT bin_size_h = static_cast<RoIT>(roi_height) / static_cast<RoIT>(out.get_height());
    const RoIT bin_size_w = static_cast<RoIT>(roi_width) / static_cast<RoIT>(out.get_width());
    const ushort roi_bin_grid_h = sampling_ratio;
    const ushort roi_bin_grid_w = sampling_ratio;
    const ushort iy_upper = sampling_ratio;
    const ushort ix_upper = sampling_ratio;
    
    const RoIT count = iy_upper * ix_upper;
    
    RoIT4 output_val = 0.0;
    for (int iy = 0; iy < iy_upper; iy++) {
        for (int ix = 0; ix < ix_upper; ix++) {
            const RoIT y =
            roi_start_h + ph * bin_size_h + iy * bin_size_h / static_cast<RoIT>(roi_bin_grid_h);
            const RoIT x =
            roi_start_w + pw * bin_size_w + ix * bin_size_w / static_cast<RoIT>(roi_bin_grid_w);
            output_val += in.sample(s2, float2(x + 0.5, y + 0.5), c);
        }
    }
    output_val /= count;
    out.write(static_cast<half4>(output_val), ushort2(gid.x, gid.y), gid.z);
}


)V0G0N";
