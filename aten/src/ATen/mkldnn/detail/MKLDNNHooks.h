#include <ATen/detail/MKLDNNHooksInterface.h>

namespace at { namespace native { namespace detail {

// The real implementation of MKLDNNHooksInterface
struct MKLDNNHooks : public at::MKLDNNHooksInterface {
  MKLDNNHooks(at::MKLDNNHooksArgs) {}
  bool hasMKLDNN() const override;
  bool compiledWithMKLDNN() const override;
  bool supportsDilatedConvolutionWithMKLDNN() const override;
  bool supportsRNNWithMKLDNN() const override;
};

}}} // namespace at::native::detail
