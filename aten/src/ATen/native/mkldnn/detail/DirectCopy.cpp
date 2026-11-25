#include <ATen/native/mkldnn/detail/DirectCopy.h>

#if AT_MKLDNN_ENABLED()

namespace at::native::mkldnn {

void direct_copy(const ideep::tensor& src, ideep::tensor& dst) {
  // May replace dst's buffer and layout when they differ from src's.
  dst.reinit_if_possible(src.get_desc());
  src.reorder_to(dst);
  if (src.has_scale()) {
    dst.set_scale(src.get_scale());
  }
}

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
