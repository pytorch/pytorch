#include <ATen/native/Unfold2d.h>

namespace at { namespace native {

DEFINE_DISPATCH(unfolded2d_copy_stub);
DEFINE_DISPATCH(unfolded2d_copy_channels_last_stub);
DEFINE_DISPATCH(unfolded2d_acc_stub);
DEFINE_DISPATCH(unfolded2d_acc_channels_last_stub);

}}
