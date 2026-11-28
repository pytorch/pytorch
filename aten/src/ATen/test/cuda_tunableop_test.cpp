#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/TunableOpExecution.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

namespace {

using at::cuda::tunable::ResultEntry;
using at::cuda::tunable::ServingTunableOpResults;

std::shared_ptr<const ServingTunableOpResults> table(const std::string& solution) {
  return std::make_shared<const ServingTunableOpResults>(
      at::cuda::tunable::ResultsMap{{"Gemm", {{"shape", ResultEntry(solution, 1.0)}}}});
}

at::TunableOpBinding enabledBinding(
    int device,
    std::shared_ptr<const ServingTunableOpResults> results = table("A")) {
  return std::make_shared<const at::TunableOpExecutionBinding>(
      at::TunableOpExecutionBinding{
          at::TunableOpPermission::Enabled, device, /*generation=*/7, std::move(results)});
}

class TunableOpCUDATest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!at::cuda::is_available()) {
      GTEST_SKIP() << "requires a CUDA or HIP device";
    }
    device_guard_.emplace(0);
    at::globalContext().lazyInitDevice(at::kCUDA);
    stream_guard_.emplace(c10::cuda::getStreamFromPool());
    at::InitializeManagedTunableOp();
    at::SetTunableOpBinding(enabledBinding(0));
  }

  std::optional<c10::cuda::CUDAGuard> device_guard_;
  std::optional<c10::cuda::CUDAStreamGuard> stream_guard_;
  at::TunableOpExecutionGuard restore_binding_{nullptr};
};

TEST_F(TunableOpCUDATest, PermissionOverridesLegacyEnablement) {
  at::cuda::tunable::TuningContext context;
  context.EnableTunableOp(true);
  {
    at::TunableOpExecutionGuard unbound(nullptr);
    EXPECT_FALSE(context.IsTunableOpEnabled());
  }
  {
    at::TunableOpExecutionGuard disabled(at::DisabledTunableOpBinding());
    EXPECT_FALSE(context.IsTunableOpEnabled());
  }
  EXPECT_TRUE(context.IsTunableOpEnabled());
  context.EnableTunableOp(false);
  EXPECT_TRUE(context.IsTunableOpEnabled());
}

TEST_F(TunableOpCUDATest, WrongDeviceCannotGrantPermission) {
  at::TunableOpExecutionGuard wrong_device(enabledBinding(-1));
  EXPECT_FALSE(at::cuda::tunable::getTuningContext()->IsTunableOpEnabled());
}

TEST_F(TunableOpCUDATest, ManagedExecutionDisablesOnlineSideEffects) {
  at::cuda::tunable::TuningContext context;
  context.EnableTuning(true);
  context.EnableRecordUntuned(true);
  context.EnableWildcardFallback(true);
  EXPECT_FALSE(context.IsTuningEnabled());
  EXPECT_FALSE(context.IsRecordUntunedEnabled());
  EXPECT_FALSE(context.IsWildcardFallbackEnabled());
}

TEST_F(TunableOpCUDATest, BindingWithoutResultsIsDisabled) {
  at::TunableOpExecutionGuard no_results(enabledBinding(0, nullptr));
  EXPECT_FALSE(at::cuda::tunable::getTuningContext()->IsTunableOpEnabled());
}

TEST_F(TunableOpCUDATest, EachBindingLooksUpOnlyItsOwnTable) {
  using at::cuda::tunable::LookupServingTunableOpResult;
  {
    at::TunableOpExecutionGuard a(enabledBinding(0, table("A")));
    EXPECT_EQ(LookupServingTunableOpResult("Gemm", "shape").GetKey(), "A");
    EXPECT_EQ(LookupServingTunableOpResult("Gemm", "other"), ResultEntry::Null());
    {
      at::TunableOpExecutionGuard b(enabledBinding(0, table("B")));
      EXPECT_EQ(LookupServingTunableOpResult("Gemm", "shape").GetKey(), "B");
    }
    EXPECT_EQ(LookupServingTunableOpResult("Gemm", "shape").GetKey(), "A");
  }
  {
    at::TunableOpExecutionGuard disabled(at::DisabledTunableOpBinding());
    EXPECT_EQ(LookupServingTunableOpResult("Gemm", "shape"), ResultEntry::Null());
  }
  {
    at::TunableOpExecutionGuard unbound(nullptr);
    EXPECT_EQ(LookupServingTunableOpResult("Gemm", "shape"), ResultEntry::Null());
  }
}

TEST_F(TunableOpCUDATest, LoadsOnlyResultsForThisEnvironment) {
  const std::string path = ::testing::TempDir() + "tunableop_results.csv";
  auto write = [&](bool matching) {
    std::ofstream file(path);
    for (const auto& [key, value] : at::cuda::tunable::getTuningContext()
             ->GetTuningResultsValidator()
             .GetAllValidators()) {
      file << "Validator," << key << "," << (matching ? value : "other") << "\n";
    }
    file << "Gemm,shape,Default,1.0\n";
  };
  write(true);
  const auto loaded = at::cuda::tunable::LoadServingTunableOpResults(path);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->GetSize(), 1);
  EXPECT_EQ(loaded->Lookup("Gemm", "shape").GetKey(), "Default");
  write(false);
  EXPECT_EQ(at::cuda::tunable::LoadServingTunableOpResults(path), nullptr);
  std::remove(path.c_str());
  EXPECT_EQ(at::cuda::tunable::LoadServingTunableOpResults(path), nullptr);
}

TEST_F(TunableOpCUDATest, StreamCaptureKeepsTheCallsPermission) {
  auto* context = at::cuda::tunable::getTuningContext();
  const auto options = at::TensorOptions().device(at::kCUDA);
  const auto a = at::randn({64, 32}, options);
  const auto b = at::randn({32, 16}, options);
  auto out = at::empty({64, 16}, options);
  at::mm_out(out, a, b);
  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  EXPECT_TRUE(context->IsTunableOpEnabled());
  at::mm_out(out, a, b);
  graph.capture_end();
  out.zero_();
  graph.replay();
  EXPECT_TRUE(at::allclose(out, at::mm(a, b)));
}

} // namespace
