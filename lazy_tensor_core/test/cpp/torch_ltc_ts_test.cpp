#include "torch_ltc_ts_test.h"

#include <ATen/ATen.h>

#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/tensor.h>

namespace torch_lazy_tensors {
namespace cpp_test {

void LtcTsTest::SetUp() {
  at::manual_seed(42);
  torch::lazy::LazyGraphExecutor::Get()->SetRngSeed(torch::lazy::BackendDevice(), 42);
}

void LtcTsTest::TearDown() {}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
