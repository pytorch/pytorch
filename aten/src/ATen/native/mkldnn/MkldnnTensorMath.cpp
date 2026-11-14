#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <cstring>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/zero_native.h>
#endif

#if !AT_MKLDNN_ENABLED()


namespace at::native {

Tensor& mkldnn_zero_(Tensor& self) {
  TORCH_CHECK(false, "mkldnn_zero_: ATen not compiled with MKLDNN support");
}

} // namespace at::native


#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>


namespace at::native {

Tensor& mkldnn_zero_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  const int64_t nbytes = nbytes_from_mkldnn(self);
  if (nbytes == 0) {
    return self;
  }

  auto* data = static_cast<char*>(x.get_data_handle());
  parallel_for(0, nbytes, 8192, [data](int64_t begin, int64_t end) {
    std::memset(data + begin, 0, end - begin);
  });

  return self;
}

} // namespace at::native


#endif // AT_MKLDNN_ENABLED
