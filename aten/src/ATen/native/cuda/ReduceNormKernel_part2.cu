#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/ReduceNormKernel.cuh>

namespace at::native {

void norm_launch_kernel_part2(TensorIterator& iter, double ord) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_cuda", [&] {
    norm_kernel_cuda_impl<scalar_t>(iter, ord);
  });
}

} // namespace at::native
