#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/ReduceNormKernel.cuh>

namespace at::native {

void norm_launch_kernel_part2(TensorIterator& iter, double ord);

void norm_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return norm_kernel_cuda_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::Half, float, float>(iter, ord);
  }
  else if(iter.dtype(0) == kBFloat16) {
    return norm_kernel_cuda_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::BFloat16, float, float>(iter, ord);
  }
  norm_launch_kernel_part2(iter, ord);
}

} // namespace at::native
