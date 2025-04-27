#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void col2im_kernel(
    device const T* data_col [[buffer(0)]],
    device T* data_im [[buffer(1)]],
    constant uint& col_batch_stride [[buffer(2)]],
    constant uint& nbatch [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& height [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& kernel_h [[buffer(7)]],
    constant uint& kernel_w [[buffer(8)]],
    constant uint& pad_h [[buffer(9)]],
    constant uint& pad_w [[buffer(10)]],
    constant uint& stride_h [[buffer(11)]],
    constant uint& stride_w [[buffer(12)]],
    constant uint& dilation_h [[buffer(13)]],
    constant uint& dilation_w [[buffer(14)]],
    constant uint& height_col [[buffer(15)]],
    constant uint& width_col [[buffer(16)]],
    constant uint& im_batch_stride [[buffer(17)]],
    uint gid [[thread_position_in_grid]]) {
  uint n = channels * height * width;
  if (gid >= nbatch * n)
    return;
  uint ibatch = gid / n;
  uint index = gid % n;

  uint w_im = (index % width) + pad_w;
  uint h_im = ((index / width) % height) + pad_h;
  uint c_im = index / (height * width);

  uint kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
  uint kernel_extent_h = (kernel_h - 1) * dilation_h + 1;

  uint w_col_start =
      (w_im < kernel_extent_w) ? 0 : ((w_im - kernel_extent_w) / stride_w + 1);
  uint w_col_end = min((w_im / stride_w + 1), width_col);
  uint h_col_start =
      (h_im < kernel_extent_h) ? 0 : ((h_im - kernel_extent_h) / stride_h + 1);
  uint h_col_end = min((h_im / stride_h + 1), height_col);

  float accumulator = 0.0;
  uint col_batch_offset = ibatch * col_batch_stride;

  for (uint h_col = h_col_start; h_col < h_col_end; h_col++) {
    for (uint w_col = w_col_start; w_col < w_col_end; w_col++) {
      int h_k = h_im - (h_col * stride_h);
      int w_k = w_im - (w_col * stride_w);

      if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
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

  uint im_batch_offset = ibatch * im_batch_stride;
  data_im[im_batch_offset + index] = static_cast<T>(accumulator);
}

#define INSTANTIATE_COL2IM(DTYPE)                             \
  template [[host_name("col2im_kernel_" #DTYPE)]] kernel void \
  col2im_kernel<DTYPE>(                                       \
      device const DTYPE* data_col [[buffer(0)]],             \
      device DTYPE* data_im [[buffer(1)]],                    \
      constant uint& col_batch_stride [[buffer(2)]],          \
      constant uint& batch [[buffer(3)]],                     \
      constant uint& channels [[buffer(4)]],                  \
      constant uint& height [[buffer(5)]],                    \
      constant uint& width [[buffer(6)]],                     \
      constant uint& kernel_h [[buffer(7)]],                  \
      constant uint& kernel_w [[buffer(8)]],                  \
      constant uint& pad_h [[buffer(9)]],                     \
      constant uint& pad_w [[buffer(10)]],                    \
      constant uint& stride_h [[buffer(11)]],                 \
      constant uint& stride_w [[buffer(12)]],                 \
      constant uint& dilation_h [[buffer(13)]],               \
      constant uint& dilation_w [[buffer(14)]],               \
      constant uint& height_col [[buffer(15)]],               \
      constant uint& width_col [[buffer(16)]],                \
      constant uint& im_batch_stride [[buffer(17)]],          \
      uint gid [[thread_position_in_grid]]);

INSTANTIATE_COL2IM(bool);
INSTANTIATE_COL2IM(float);
INSTANTIATE_COL2IM(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_COL2IM(bfloat);
#endif
