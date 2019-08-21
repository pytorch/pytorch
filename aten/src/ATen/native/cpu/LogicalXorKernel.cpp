// The content of BinaryLogicalOpsKernel.h and Logical*Kernel.cpp should have inhabited in BinaryOpsKernel.cpp. But
// doing so will make the compilation of BinaryOpsKernel.cpp so long and cause the CI to break. These files merely serve
// as a workaround to reduce the compilation time of BinaryOpsKernel.cpp by breaking down BinaryOpsKernel.cpp to
// multiple files.

#include <ATen/native/cpu/BinaryLogicalOpsKernel.h>

namespace at { namespace native {

static void logical_xor_kernel(TensorIterator& iter) {
  logical_binary_kernel_impl(iter, "logical_xor_cpu", [](bool a, bool b) -> bool { return a != b; });
}

REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel);

}} // namespace at::native
