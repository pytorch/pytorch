#include <ATen/mkldnn/detail/MKLDNNHooks.h>
#include <ATen/Config.h>

namespace at { namespace native { namespace detail {

bool MKLDNNHooks::hasMKLDNN() const {
  return AT_MKLDNN_ENABLED();
}

bool MKLDNNHooks::compiledWithMKLDNN() const {
  return AT_MKLDNN_ENABLED();
}

// TODO: add mkldnn dilated convolution support
bool MKLDNNHooks::supportsDilatedConvolutionWithMKLDNN() const {
  return false;
}

bool MKLDNNHooks::supportsRNNWithMKLDNN() const {
  return true;
}

using at::MKLDNNHooksRegistry;
using at::RegistererMKLDNNHooksRegistry;

REGISTER_MKLDNN_HOOKS(MKLDNNHooks);

}}} // namespace at::native::detail
