#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using adaptive_avg_pooling_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
using adaptive_avg_pooling_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
DECLARE_DISPATCH(adaptive_avg_pooling_fn, adaptive_avg_pool2d_kernel);
DECLARE_DISPATCH(adaptive_avg_pooling_backward_fn, adaptive_avg_pool2d_backward_kernel);
DECLARE_DISPATCH(adaptive_avg_pooling_fn, adaptive_avg_pool3d_kernel);
DECLARE_DISPATCH(adaptive_avg_pooling_backward_fn, adaptive_avg_pool3d_backward_kernel);

using adaptive_max_pooling_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, IntArrayRef output_size);
using adaptive_max_pooling_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
DECLARE_DISPATCH(adaptive_max_pooling_fn, adaptive_max_pool2d_kernel);
DECLARE_DISPATCH(adaptive_max_pooling_backward_fn, adaptive_max_pool2d_backward_kernel);
DECLARE_DISPATCH(adaptive_max_pooling_fn, adaptive_max_pool3d_kernel);
DECLARE_DISPATCH(adaptive_max_pooling_backward_fn, adaptive_max_pool3d_backward_kernel);

static inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::floor((float)(a * c) / b);
}

static inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::ceil((float)((a + 1) * c) / b);
}

namespace {

struct PreCalc {
  int64_t id0;
  int64_t id1;
  int64_t ih0;
  int64_t ih1;
  int64_t iw0;
  int64_t iw1;
};

// Pre calculate input volume indices for each output index.
// The indices are shared per each output channel.
static inline void pre_calc_for_adaptive_pool(
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    std::vector<PreCalc>& pre_calc) {
  int64_t pre_calc_index = 0;
  for (int64_t od = 0; od < output_depth; od++) {
    int64_t id0 = start_index(od, output_depth, input_depth);
    int64_t id1 = end_index(od, output_depth, input_depth);

    for (int64_t oh = 0; oh < output_height; oh++) {
      int64_t ih0 = start_index(oh, output_height, input_height);
      int64_t ih1 = end_index(oh, output_height, input_height);

      for (int64_t ow = 0; ow < output_width; ow++) {
        int64_t iw0 = start_index(ow, output_width, input_width);
        int64_t iw1 = end_index(ow, output_width, input_width);

        PreCalc pc;
        pc.id0 = id0;
        pc.id1 = id1;
        pc.ih0 = ih0;
        pc.ih1 = ih1;
        pc.iw0 = iw0;
        pc.iw1 = iw1;
        pre_calc[pre_calc_index++] = pc;
      }
    }
  }
}

} // namespace

}} // namespace at::native
