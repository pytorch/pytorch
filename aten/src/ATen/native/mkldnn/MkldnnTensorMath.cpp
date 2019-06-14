#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_zero_(Tensor& self) {
  AT_ERROR("mkldnn_zero_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor& mkldnn_zero_(Tensor& self) {
  using Vec = vec256::Vec256<float>;

  ideep::tensor& x = itensor_from_mkldnn(self);

  auto n = x.get_nelems();
  auto* x_ = static_cast<float*>(x.get_data_handle());
  parallel_for(0, n, 2048, [x_](int64_t begin, int64_t end) {
    vec256::map(
        [](Vec /* unused */) { return 0.0; },
        x_ + begin,
        x_ + begin,
        end - begin);
  });

  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
