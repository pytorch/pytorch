#include "torch_ltc_ts_test.h"

#include <ATen/ATen.h>

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace cpp_test {

void LtcTsTest::SetUp() {
  at::manual_seed(42);
  LazyTensor::SetRngSeed(GetCurrentDevice(), 42);
}

void LtcTsTest::TearDown() {}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
