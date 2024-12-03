#pragma once

namespace at {
struct TensorIteratorBase;

namespace native {

void direct_copy_kernel_cuda(TensorIteratorBase& iter);

}
} // namespace at
