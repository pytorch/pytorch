#ifndef MetalCopy_h
#define MetalCopy_h

#include <ATen/Tensor.h>

namespace at::native::metal {

Tensor copy_to_host(const Tensor& input);

} // namespace at::native::metal

#endif
