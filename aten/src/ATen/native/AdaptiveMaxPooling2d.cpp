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

/**
 * Adaptive Max Pooling 2D
 *
 * This function performs adaptive max pooling on a 2D input tensor. Adaptive max pooling allows you to specify the output size
 * instead of a fixed kernel size. It divides the input into non-overlapping regions and computes the maximum value in each
 * region to create the output.
 *
 * Input:
 *   - `input`: The input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W
 *     are the input height and width.
 *
 * Output:
 *   - The output tensor of shape (B, C, oH, oW), where oH and oW are the output height and width specified by `output_size`.
 *
 * Parameters:
 *   - `output_size`: A tuple (oH, oW) specifying the output size. oH and oW must be integers greater than 0 and less than or
 *     equal to H and W, respectively.
 *
 * Behavior:
 *   - If the input shape is divisible by the output size (H % oH == 0 and W % oW == 0), the function divides the input into
 *     non-overlapping regions of size (H/oH) x (W/oW) and computes the maximum value in each region to produce the output.
 *   - If the input shape is not divisible by the output size, the function adjusts the region sizes to ensure that each output
 *     element corresponds to the maximum value over one or more input elements. Specifically, it divides the input into regions
 *     of size ceil(H/oH) x ceil(W/oW) and computes the maximum value in each region.
 *
 * Example:
 *   - Input tensor: [B=1, C=1, H=4, W=4]
 *   - Output size: (oH=2, oW=2)
 *   - Output tensor: [B=1, C=1, oH=2, oW=2]
 *   - Calculation:
 *     - Region 1 (2x2): Max value is taken from [1, 2, 5, 6] => Output element [0, 0] = 6
 *     - Region 2 (2x2): Max value is taken from [2, 3, 6, 7] => Output element [0, 1] = 7
 *     - Region 3 (2x2): Max value is taken from [9, 10, 13, 14] => Output element [1, 0] = 14
 *     - Region 4 (2x2): Max value is taken from [10, 11, 14, 15] => Output element [1, 1] = 15
 *
 * Notes:
 *   - The function adapts to input and output sizes, ensuring that each output element represents the maximum value over one
 *     or more input elements.
 *   - This function works for both CPU and GPU (CUDA) backends.
 *   - It supports batched inputs (multiple samples in the batch).
 *
 * @param input The input tensor.
 * @param output_size The desired output size.
 * @return The output tensor.
 */


namespace at {
namespace meta {
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
} // namespace meta

namespace native {

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

} // at::native
} // at
