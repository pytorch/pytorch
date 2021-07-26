#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/AdaptivePooling.h>


namespace at {
namespace meta {
TORCH_META_FUNC(adaptive_max_pool3d) (const Tensor& input, IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(input.size(i) > 0,
        "adaptive_max_pool3d: expected input to have non-empty spatial dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i,
        " being empty");
  }

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(output_size.size() == 3,
      "adaptive_max_pool3d: internal error: output_size.size() must be 3");

  int dimD = 0;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (input.ndimension() == 5) {
    sizeB = input.size(0);
    dimD++;
  }

  /* sizes */
  sizeD = input.size(dimD);

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  /* resize output */
  if (input.ndimension() == 4) {
    set_output(0, {sizeD, osizeT, osizeH, osizeW}, input.options());
    /* indices will contain max input locations for each output point */
    set_output(1, {sizeD, osizeT, osizeH, osizeW}, input.options().dtype(kLong));
  } else {
    set_output(0, {sizeB, sizeD, osizeT, osizeH, osizeW}, input.options().memory_format(input.suggest_memory_format()));
    /* indices will contain max input locations for each output point */
    set_output(1, {sizeB, sizeD, osizeT, osizeH, osizeW}, input.options().memory_format(input.suggest_memory_format()).dtype(kLong));
  }
}

TORCH_META_FUNC(adaptive_max_pool3d_backward)
(const Tensor& gradOutput, const Tensor& input, const Tensor& indices) {
  set_output(0, input.sizes(), input.options().memory_format(input.suggest_memory_format()));
}
} // namespace meta

namespace native {

TORCH_IMPL_FUNC(adaptive_max_pool3d_out_cpu)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
  adaptive_max_pool3d_kernel(kCPU, output, indices, input, output_size);
}

TORCH_IMPL_FUNC(adaptive_max_pool3d_backward_out_cpu)
(const Tensor& grad_output,
 const Tensor& input,
 const Tensor& indices,
 const Tensor& grad_input) {
  grad_input.zero_();
  adaptive_max_pool3d_backward_kernel(kCPU, grad_input, grad_output, indices);
}

DEFINE_DISPATCH(adaptive_max_pool3d_kernel);
DEFINE_DISPATCH(adaptive_max_pool3d_backward_kernel);

} // at::native
} // at
