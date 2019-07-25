#include <ATen/ATen.h>

namespace at {
namespace native {

namespace {
inline bool all_dims_are_same(const Tensor& t1, const Tensor& t2) {
  if (t1.ndimension() != t2.ndimension()) {
    return false;
  }

  IntArrayRef t1_size = t1.sizes();
  IntArrayRef t2_size = t2.sizes();
  for (int64_t i = 0; i < t1.ndimension(); ++i) {
    if (t1_size[i] != t2_size[i]) {
      return false;
    }
  }
  return true;
}
} // namespace

namespace binary_op_fast_path_conditions {

bool enable_contiguous_eq_size_cpu_fastpath(const Tensor& t1, const Tensor& t2) {
  return all_dims_are_same(t1, t2) && t2.is_contiguous() &&
         t1.device().is_cpu() && t2.device().is_cpu() &&
         t1.is_contiguous() &&
         (t1.scalar_type() == t2.scalar_type());
}

} // binary_op_fast_path_conditions
} // native
} // at
