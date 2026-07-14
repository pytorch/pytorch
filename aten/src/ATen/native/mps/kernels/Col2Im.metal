#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void col2im_kernel(
    device const T* data_col [[buffer(0)]],
    device T* data_im [[buffer(1)]],
    constant uint& col_batch_stride [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint2& im_hw [[buffer(4)]],
    constant uint2& kernel_hw [[buffer(5)]],
    constant uint2& pad_hw [[buffer(6)]],
    constant uint2& stride_hw [[buffer(7)]],
    constant uint2& dilation_hw [[buffer(8)]],
    constant uint2& col_hw [[buffer(9)]],
    constant uint& im_batch_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint output_height = im_hw.x;
  const uint output_width = im_hw.y;

  uint x = gid.x;
  uint y = gid.y;

  uint bc = gid.z;
  uint batch_idx = bc / channels;
  uint c_im = bc % channels;

  uint w_im = x + pad_hw.y;
  uint h_im = y + pad_hw.x;

  uint kernel_h = kernel_hw.x;
  uint kernel_w = kernel_hw.y;
  uint stride_h = stride_hw.x;
  uint stride_w = stride_hw.y;
  uint dilation_h = dilation_hw.x;
  uint dilation_w = dilation_hw.y;

  uint height_col = col_hw.x;
  uint width_col = col_hw.y;

  uint kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
  uint kernel_extent_h = (kernel_h - 1) * dilation_h + 1;

  uint w_col_start =
      (w_im < kernel_extent_w) ? 0 : ((w_im - kernel_extent_w) / stride_w + 1);
  uint w_col_end = min((w_im / stride_w + 1), width_col);
  uint h_col_start =
      (h_im < kernel_extent_h) ? 0 : ((h_im - kernel_extent_h) / stride_h + 1);
  uint h_col_end = min((h_im / stride_h + 1), height_col);

  float accumulator = 0.0;
  uint col_batch_offset = batch_idx * col_batch_stride;

  for (uint h_col = h_col_start; h_col < h_col_end; h_col++) {
    for (uint w_col = w_col_start; w_col < w_col_end; w_col++) {
      uint h_k = h_im - (h_col * stride_h);
      uint w_k = w_im - (w_col * stride_w);

      if ((h_k % dilation_h == 0) && (w_k % dilation_w == 0)) {
        h_k /= dilation_h;
        w_k /= dilation_w;
        if (h_k < kernel_h && w_k < kernel_w) {
          uint col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col +
               h_col) *
                  width_col +
              w_col;
          accumulator +=
              static_cast<float>(data_col[col_batch_offset + col_index]);
        }
      }
    }
  }

  uint im_batch_offset = batch_idx * im_batch_stride;
  uint im_index = (c_im * output_height + y) * output_width + x;
  data_im[im_batch_offset + im_index] = static_cast<T>(accumulator);
}

#define INSTANTIATE_COL2IM(DTYPE)                             \
  template [[host_name("col2im_kernel_" #DTYPE)]] kernel void \
  col2im_kernel<DTYPE>(                                       \
      device const DTYPE* data_col [[buffer(0)]],             \
      device DTYPE* data_im [[buffer(1)]],                    \
      constant uint& col_batch_stride [[buffer(2)]],          \
      constant uint& channels [[buffer(3)]],                  \
      constant uint2& im_hw [[buffer(4)]],                    \
      constant uint2& kernel_hw [[buffer(5)]],                \
      constant uint2& pad_hw [[buffer(6)]],                   \
      constant uint2& stride_hw [[buffer(7)]],                \
      constant uint2& dilation_hw [[buffer(8)]],              \
      constant uint2& col_hw [[buffer(9)]],                   \
      constant uint& im_batch_stride [[buffer(10)]],          \
      uint3 gid [[thread_position_in_grid]]);

INSTANTIATE_COL2IM(bool);
INSTANTIATE_COL2IM(float);
INSTANTIATE_COL2IM(half);
INSTANTIATE_COL2IM(bfloat);
