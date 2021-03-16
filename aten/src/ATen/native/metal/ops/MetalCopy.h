#ifndef MetalCopy_h
#define MetalCopy_h

#include <ATen/Tensor.h>

namespace at {
namespace native {
namespace metal {

Tensor copy_to_host(const Tensor& input);

}
} // namespace native
} // namespace at

#endif
