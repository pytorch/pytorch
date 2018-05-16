#include "ATen/ATen.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/ScalarType.h"
#include "THC/THCTensorRandom.h"

#include <algorithm>
#include <sstream>

namespace at {
namespace native {

Tensor& eye_out_cuda(Tensor& result, int64_t n, int64_t m) {
  if (n <= 0) {
    std::ostringstream oss;
    oss << "n must be greater than 0, got: " << n;
    std::runtime_error(oss.str());
  }
  if(m <= 0) {
    m = n;
  }

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

namespace {
template <typename scalar_t>
void randperm_cuda(Tensor& result, int64_t n, Generator* generator) {
  result.resize_({n});

  for(int64_t i = 0; i < n; i++) {
    result[i] = i;
  }

  for(int64_t i = 0; i < n - 1; i++)
  {
    auto t = result.type().tensor(1).random_(generator);
    int64_t z = Scalar(t[0]).toLong() % (n-i);

    scalar_t sav = Scalar(result[i]).to<scalar_t>();
    result[i] = result[z+i];
    result[z+i] = sav;
  }
}
} // namespace

Tensor& randperm_out_cuda(Tensor& result, int64_t n, Generator* generator) {
  if (n < 0) {
    std::ostringstream oss;
    oss << "n must be non-negative, got " << n;
    throw std::runtime_error(oss.str());
  }

  AT_DISPATCH_ALL_TYPES(result.type(), "randperm", [&]() -> void {
    randperm_cuda<scalar_t>(result, n, generator);
  });

  return result;
}

}} // namespace at::native
