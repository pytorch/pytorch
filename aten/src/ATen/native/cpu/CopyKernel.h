#pragma once

namespace at {
struct TensorIteratorBase;

namespace native {
inline namespace CPU_CAPABILITY {

void direct_copy_kernel(TensorIteratorBase &iter);
void copy_kernel(TensorIterator& iter, bool /*non_blocking*/);

} // namespace CPU_CAPABILITY
} // namespace native
} // namespace at
