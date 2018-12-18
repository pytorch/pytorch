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

// TODO: add mkldnn rnn support
bool MKLDNNHooks::supportsRNNWithMKLDNN() const {
  return false;
}

using at::MKLDNNHooksRegistry;
using at::RegistererMKLDNNHooksRegistry;

REGISTER_MKLDNN_HOOKS(MKLDNNHooks);

}}} // namespace at::native::detail
