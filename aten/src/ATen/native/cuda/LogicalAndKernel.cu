// The content of BinaryLogicalOpsKernel.cuh and Logical*Kernel.cu should have inhabited in BinaryOpsKernel.cu, like its
// CPU counterpart. But doing so will make the compilation of BinaryOpsKernel.cu so long and cause the CI to break.
// These files merely serve as a workaround to reduce the compilation time of BinaryOpsKernel.cu by breaking down
// BinaryOpsKernel.cu.

#include <ATen/native/cuda/BinaryLogicalOpsKernel.cuh>

namespace at { namespace native {

void logical_and_kernel_cuda(TensorIterator& iter) {
  logical_binary_kernel_cuda_impl(iter, "logical_and_cuda", []GPU_LAMBDA(bool a, bool b) -> bool { return a && b; });
}

REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel_cuda);

}} // namespace at::native
