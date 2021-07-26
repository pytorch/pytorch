#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using padding_fn = void(*)(const Tensor& output, const Tensor& input, IntArrayRef padding_size);
DECLARE_DISPATCH(padding_fn, replication_pad1d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad2d_kernel);
DECLARE_DISPATCH(padding_fn, replication_pad3d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad1d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad2d_kernel);
DECLARE_DISPATCH(padding_fn, reflection_pad3d_kernel);

using padding_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding_size);
DECLARE_DISPATCH(padding_backward_fn, replication_pad1d_backward_kernel);
DECLARE_DISPATCH(padding_backward_fn, replication_pad2d_backward_kernel);
DECLARE_DISPATCH(padding_backward_fn, replication_pad3d_backward_kernel);
DECLARE_DISPATCH(padding_backward_fn, reflection_pad1d_backward_kernel);
DECLARE_DISPATCH(padding_backward_fn, reflection_pad2d_backward_kernel);
DECLARE_DISPATCH(padding_backward_fn, reflection_pad3d_backward_kernel);

struct PaddingIndexr {
  int pad_start;
  int input_start;
  int output_start;
  int64_t input_size;

  PaddingIndexr(int p_start, int64_t i_size)
    : pad_start(p_start)
    , input_start(std::max(0, -p_start))
    , output_start(std::max(0, p_start))
    , input_size(i_size) {}
};

struct ReplicationPadIndexr : PaddingIndexr {
  using PaddingIndexr::PaddingIndexr;

  int64_t get(int64_t output_index) {
    int64_t input_index;
    if (output_index < pad_start) {
      input_index = pad_start;
    } else if (output_index >= pad_start && output_index < input_size + pad_start) {
      input_index = output_index;
    } else {
      input_index = input_size + pad_start - 1;
    }
    return input_index - output_start + input_start;
  }
};

struct ReflectionPadIndexr : PaddingIndexr {
  using PaddingIndexr::PaddingIndexr;

  int64_t get(int64_t output_index) {
    int64_t input_index;
    if (output_index < pad_start) {
      input_index = pad_start * 2 - output_index;
    } else if (output_index >= pad_start && output_index < input_size + pad_start) {
      input_index = output_index;
    } else {
      input_index = (input_size + pad_start - 1) * 2 - output_index;
    }
    return input_index - output_start + input_start;
  }
};

struct PaddingParams {
  int64_t ndim;
  int64_t nbatch;
  int64_t channels;
  int64_t input_depth;
  int64_t input_height;
  int64_t input_width;
  int64_t output_depth;
  int64_t output_height;
  int64_t output_width;

  PaddingParams(const Tensor& input, const Tensor& output, bool is_batch_mode) {
    ndim = input.ndimension();
    if (is_batch_mode) {
      // 1D: NCW
      // 2D: NCHW
      // 3D: NCDHW
      nbatch = input.size(0);
      channels = input.size(1);
      input_depth = (ndim == 5) ? input.size(-3) : 1;
      input_height = (ndim >= 4) ? input.size(-2) : 1;
      input_width = input.size(-1);
      output_depth = (ndim == 5) ? output.size(-3) : 1;
      output_height = (ndim >= 4) ? output.size(-2) : 1;
      output_width = output.size(-1);
    } else {
      // 1D: CW
      // 2D: CHW
      // 3D: CDHW
      nbatch = 1;
      channels = input.size(0);
      input_depth = (ndim == 4) ? input.size(-3) : 1;
      input_height = (ndim >= 3) ? input.size(-2) : 1;
      input_width = input.size(-1);
      output_depth = (ndim == 4) ? output.size(-3) : 1;
      output_height = (ndim >= 3) ? output.size(-2) : 1;
      output_width = output.size(-1);
    }
  }
};

}} // namespace at::native
