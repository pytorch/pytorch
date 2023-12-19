#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>
#endif

namespace at::meta {
TORCH_META_FUNC(adaptive_max_pool2d) (const Tensor& input, IntArrayRef output_size) {
  int ndim = input.ndimension();
  TORCH_CHECK(ndim == 3 || ndim == 4,
              "adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: ",
              input.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(input.size(i) > 0,
        "adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i,
        " being empty");
  }

  TORCH_CHECK(output_size.size() == 2,
      "adaptive_max_pool2d(): internal error: output_size.size() must be 2");

  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimH++;
  }

  sizeD = input.size(dimH - 1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  /* resize output */
  if (input.ndimension() == 3) {
    set_output_raw_strided(0, {sizeD, osizeH, osizeW}, {}, input.options());
    /* indices will contain i,j locations for each output point */
    set_output_raw_strided(1, {sizeD, osizeH, osizeW}, {}, input.options().dtype(kLong));
  } else {
    set_output_raw_strided(0, {sizeB, sizeD, osizeH, osizeW}, {}, input.options().memory_format(input.suggest_memory_format()));
    /* indices will contain i,j locations for each output point */
    set_output_raw_strided(1, {sizeB, sizeD, osizeH, osizeW}, {}, input.options().memory_format(input.suggest_memory_format()).dtype(kLong));
  }
}

TORCH_META_FUNC(adaptive_max_pool2d_backward)
(const Tensor& grad_output, const Tensor& input, const Tensor& indices) {
  int64_t ndim = grad_output.ndimension();
  TORCH_CHECK(ndim == 3 || ndim == 4,
    "adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: ", grad_output.sizes());

  at::native::adaptive_pool_empty_output_check(grad_output, "adaptive_max_pool2d_backward");

  TORCH_CHECK(input.dtype() == grad_output.dtype(),
    "expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());

  set_output_raw_strided(0, input.sizes(), {}, input.options().memory_format(input.suggest_memory_format()));
}
} // namespace at::meta

namespace at::native {

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_cpu)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
  adaptive_max_pool2d_kernel(kCPU, output, indices, input, output_size);
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_cpu)
(const Tensor& grad_output, const Tensor& input, const Tensor& indices, const Tensor& grad_input) {
  grad_input.zero_();
  adaptive_max_pool2d_backward_kernel(kCPU, grad_input, grad_output, indices);
 }

DEFINE_DISPATCH(adaptive_max_pool2d_kernel);
DEFINE_DISPATCH(adaptive_max_pool2d_backward_kernel);

} // namespace at::native
