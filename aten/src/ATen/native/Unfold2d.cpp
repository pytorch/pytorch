#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Unfold2d.h>

namespace at::native {

DEFINE_DISPATCH(unfolded2d_copy_stub);
DEFINE_DISPATCH(unfolded2d_acc_stub);

} // namespace at::native
