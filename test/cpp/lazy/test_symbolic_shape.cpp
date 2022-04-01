
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/torch.h>
#include <iostream>

namespace torch {
namespace lazy {

// Lazy Tensor is disabled in FBCODE until addressing non-virtual methods (e.g.
// sizes) in TensorImpl
#ifndef FBCODE_CAFFE2

namespace {
// This registers the torchscript backend, without which lazy device won't work
torch::lazy::BackendRegistrar g_registrar(GetTSBackendImpl());

static inline at::DeviceType DefaultDevice() {
  return torch::lazy::getBackend()->EagerFallbackDeviceType();
}

const Shape& getShapeFromLazyTensor(at::Tensor& lazy_tensor) {
  auto ltc_tensor = GetLtcTensor(lazy_tensor);
  Value ir_val = ltc_tensor->GetIrValue();
  // Bit of a hack around the const
  return ir_val->shape();
}

class LazyTsTest : public ::testing::Test {
 protected:
  void SetUp() override;

  void TearDown() override;

  static void CommonSetup() {}

  void ExpectCounterNotChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ExpectCounterChanged(
      const std::string& counter_regex,
      const std::unordered_set<std::string>* ignore_set) {}

  void ResetCounters() {}

 private:
  void MakeEndSnapshot() {}
};

class LazyShapeTest : public LazyTsTest {
 protected:
  static void SetUpTestCase() {}
};

void LazyTsTest::SetUp() {
  at::manual_seed(42);
  torch::lazy::LazyGraphExecutor::Get()->SetRngSeed(
      torch::lazy::BackendDevice(), 42);
}

} // namespace
void LazyTsTest::TearDown() {}

TEST_F(LazyShapeTest, TestAdd) {
  // TODO: Figure out if this test is needed for Milestone 2
  // Currently missing some pieces needed for this test
  /*
  torch::Tensor a = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // set_is_dynamic(a, {true, false});
  torch::Tensor b = torch::rand(
      {2, 2}, torch::TensorOptions(torch::kFloat).device(DefaultDevice()));
  // set_is_dynamic(b, {true, false});
  torch::Tensor c = torch::add(a, b);
  std::vector<bool> expected = {true, false};
  EXPECT_EQ(getShapeFromLazyTensor(c).is_symbolic(), expected);
  */
  ASSERT_TRUE(true);
};
#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace torch
