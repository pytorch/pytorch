#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/im2col.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/vec.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void col2im(
    Tensor& im,
    const Tensor& col,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  scalar_t* col_data = col.data_ptr<scalar_t>();
  scalar_t* im_data = im.data_ptr<scalar_t>();

  // col shape: [output_channels * kH * kW, input_height * input_width]
  // im shape: [output_channels, output_height, output_width]
  int64_t stride_oc = kH * kW * input_height * input_width;

  at::parallel_for(0, output_channels, 0, [&](int64_t begin, int64_t end) {
    for (int64_t oc = begin; oc < end; oc++) {
      scalar_t* col_ptr = col_data + oc * stride_oc;
      scalar_t* im_ptr = im_data + oc * output_height * output_width;

      // incremental indexing for col
      int64_t i = 0;
      for (int64_t kh = 0; kh < kH; kh++) {
        for (int64_t kw = 0; kw < kW; kw++) {
          for (int64_t ih = 0; ih < input_height; ih++) {
            int64_t oh = ih * sH - pH + kh * dH;
            bool valid_h = oh >= 0 && oh < output_height;

            for (int64_t iw = 0; iw < input_width; iw++) {
              int64_t ow = iw * sW - pW + kw * dW;
              bool valid_w = ow >= 0 && ow < output_width;

              if (valid_h && valid_w) {
                im_ptr[oh * output_width + ow] += col_ptr[i];
              }
              i++;
            }
          }
        }
      }
    }
  });
}

template <typename scalar_t>
void im2col(
    Tensor& col,
    const Tensor& im,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  scalar_t* col_data = col.data_ptr<scalar_t>();
  scalar_t* im_data = im.data_ptr<scalar_t>();

  // col shape: [output_channels * kH * kW, input_height * input_width]
  // im shape: [output_channels, output_height, output_width]
  at::parallel_for(0, col.numel(), 0, [&](int64_t begin, int64_t end) {
    int64_t oc{0}, kh{0}, kw{0}, ih{0}, iw{0};
    data_index_init(begin, oc, output_channels, kh, kH, kw, kW, ih, input_height, iw, input_width);

    for (int64_t i = begin; i < end; i++) {
      int64_t oh = ih * sH - pH + kh * dH;
      int64_t ow = iw * sW - pW + kw * dW;

      bool valid = oh >= 0 && ow >= 0 && oh < output_height && ow < output_width;
      col_data[i] = valid ? im_data[oc * output_height * output_width + oh * output_width + ow]
                          : scalar_t(0);

      data_index_step(oc, output_channels, kh, kH, kw, kW, ih, input_height, iw, input_width);
    }
  });
}

template <typename scalar_t>
void col2im_channels_last(
    Tensor& im,
    const Tensor& col,
    int64_t nbatch,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  scalar_t* col_data = col.data_ptr<scalar_t>();
  scalar_t* im_data = im.data_ptr<scalar_t>();

  // col shape: [nbatch * input_height * input_width, kH * kW * output_channels]
  // im shape: [nbatch * output_height * output_width, output_channels]
  //
  // parallel on dim nbatch and do vectorization on dim output_channels
  int64_t size = output_channels;
  using Vec = vec::Vectorized<scalar_t>;
  at::parallel_for(0, nbatch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; n++) {
      scalar_t* col_ptr = col_data + n * input_height * input_width * kH * kW * output_channels;
      scalar_t* im_ptr = im_data + n * output_height * output_width * output_channels;

      // incremental indexing for col
      int64_t i = 0;
      for (int64_t ih = 0; ih < input_height; ih++) {
        for (int64_t iw = 0; iw < input_width; iw++) {
          for (int64_t kh = 0; kh < kH; kh++) {
          int64_t oh = ih * sH - pH + kh * dH;
          bool valid_h = oh >= 0 && oh < output_height;

            for (int64_t kw = 0; kw <kW; kw++) {
              int64_t ow = iw * sW - pW + kw * dW;
              bool valid_w = ow >= 0 && ow < output_width;

              if (valid_h && valid_w) {
                scalar_t* col_ = col_ptr + i * output_channels;
                scalar_t* im_ = im_ptr + oh * output_width * output_channels + ow * output_channels;

                int64_t d = 0;
                for (; d < size - (size % Vec::size()); d += Vec::size()) {
                  Vec im_vec = Vec::loadu(im_ + d) + Vec::loadu(col_ + d);
                  im_vec.store(im_ + d);
                }
                for (; d < size; d++) {
                  im_[d] += col_[d];
                }
              }
              i++;
            }
          }
        }
      }
    }
  });
}

template <typename scalar_t>
void im2col_channels_last(
    Tensor& col,
    const Tensor& im,
    int64_t nbatch,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  scalar_t* col_data = col.data_ptr<scalar_t>();
  scalar_t* im_data = im.data_ptr<scalar_t>();

  // col shape: [nbatch * input_height * input_width, kH * kW * output_channels]
  // im shape: [nbatch * output_height * output_width, output_channels]
  int64_t stride_n = output_height * output_width * output_channels;
  int64_t stride_oh = output_width * output_channels;
  int64_t stride_ow = output_channels;

  // copy from im to col has no thread conflict, crash all dims excpet OC.
  int64_t m = col.numel() / output_channels;

  int64_t size = output_channels;
  using Vec = vec::Vectorized<scalar_t>;
  at::parallel_for(0, m, 0, [&](int64_t begin, int64_t end) {
    int64_t n{0}, ih{0}, iw{0}, kh{0}, kw{0};
    data_index_init(begin, n, nbatch, ih, input_height, iw, input_width, kh, kH, kw, kW);

    for (int64_t i = begin; i < end; i++) {
      int64_t oh = ih * sH - pH + kh * dH;
      int64_t ow = iw * sW - pW + kw * dW;

      scalar_t* im_ptr = im_data + n * stride_n + oh * stride_oh + ow * stride_ow;
      scalar_t* col_ptr = col_data + i * output_channels;

      bool need_update = oh >= 0 && ow >= 0 && oh < output_height && ow < output_width;
      int64_t d = 0;
      for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec im_vec = need_update ? Vec::loadu(im_ptr + d) : Vec(scalar_t(0));
        im_vec.store(col_ptr + d);
      }
      for (; d < size; d++) {
        col_ptr[d] = need_update ? im_ptr[d] : scalar_t(0);
      }

      // move on to next output index
      data_index_step(n, nbatch, ih, input_height, iw, input_width, kh, kH, kw, kW);
    }
  });
}

void col2im_kernel(
    Tensor& im,
    const Tensor& col,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, col.scalar_type(), "col2im", [&] {
    col2im<scalar_t>(
        im,
        col,
        output_channels,
        output_height, output_width,
        input_height, input_width,
        kH, kW,
        pH, pW,
        sH, sW,
        dH, dW);
  });
}

void im2col_kernel(
    Tensor& col,
    const Tensor& im,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, col.scalar_type(), "im2col", [&] {
    im2col<scalar_t>(
      col,
      im,
      output_channels,
      output_height, output_width,
      input_height, input_width,
      kH, kW,
      pH, pW,
      sH, sW,
      dH, dW);
  });
}

void col2im_channels_last_kernel(
    Tensor& im,
    const Tensor& col,
    int64_t nbatch,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, col.scalar_type(), "col2im_channels_last", [&] {
    col2im_channels_last<scalar_t>(
        im,
        col,
        nbatch,
        output_channels,
        output_height, output_width,
        input_height, input_width,
        kH, kW,
        pH, pW,
        sH, sW,
        dH, dW);
  });
}

void im2col_channels_last_kernel(
    Tensor& col,
    const Tensor& im,
    int64_t nbatch,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, col.scalar_type(), "im2col_channels_last", [&] {
    im2col_channels_last<scalar_t>(
      col,
      im,
      nbatch,
      output_channels,
      output_height, output_width,
      input_height, input_width,
      kH, kW,
      pH, pW,
      sH, sW,
      dH, dW);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(col2im_stub, &col2im_kernel);
REGISTER_DISPATCH(im2col_stub, &im2col_kernel);
REGISTER_DISPATCH(col2im_channels_last_stub, &col2im_channels_last_kernel);
REGISTER_DISPATCH(im2col_channels_last_stub, &im2col_channels_last_kernel);

}} // at::native
