
namespace at {
struct TensorIterator;
}

namespace c10 {
class Scalar;
}

namespace at { namespace native {

void norm_launch_kernel(TensorIterator &iter, double val);
void min_launch_kernel(TensorIterator &iter);
void max_launch_kernel(TensorIterator &iter);
void aminmax_launch_kernel(TensorIterator &iter);
void min_all_launch_kernel(TensorIterator &iter);
void max_all_launch_kernel(TensorIterator &iter);
void aminmax_allreduce_launch_kernel(TensorIterator &iter);

}}  // namespace at::native
