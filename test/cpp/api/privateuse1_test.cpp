#include <gtest/gtest.h>

#include <ATen/AccumulateType.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <test/cpp/api/support.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <optional>

// Exercises the "registered but unequipped" path: a `PrivateUse1HooksInterface`
// subclass that leaves `toAccumulateType` to the base-class default (returns
// `std::nullopt` for every dtype). Every dtype should defer to the CPU
// accumulation type and emit `TORCH_WARN_ONCE`.
//
// `RegisterPrivateUse1HooksInterface` is process-global and one-shot, so the
// registration is guarded by `isPrivateUse1HooksRegistered()`: if another test
// in the `test_api` binary has already registered hooks, this test skips
// rather than tripping the one-shot `TORCH_CHECK`.

namespace {

c10::ScalarType cpu_acc_type(c10::ScalarType t) {
  return at::toAccumulateType(t, c10::DeviceType::CPU);
}

// Registered hooks that do NOT override `toAccumulateType`. The base-class
// default returns `std::nullopt` for every dtype, so every query should defer
// to the CPU accumulation type and emit `TORCH_WARN_ONCE`.
struct DefaultAccTypeHooks final : at::PrivateUse1HooksInterface {
  bool hasPrimaryContext(c10::DeviceIndex) const override {
    return true;
  }
};

} // namespace

TEST(PrivateUse1Test, RegisteredWithoutOverrideDefersToCpuAndWarnsOnce) {
  if (at::isPrivateUse1HooksRegistered()) {
    GTEST_SKIP()
        << "PrivateUse1 hooks already registered by another test; one-shot "
        << "registration prevents the unequipped-default path from running.";
  }

  at::RegisterPrivateUse1HooksInterface(new DefaultAccTypeHooks());

  // kFloat is the canary: CPU default is kDouble, so a buggy implementation
  // that didn't actually take the fallback would be caught here.
  for (auto t : {
           c10::ScalarType::Half,
           c10::ScalarType::BFloat16,
           c10::ScalarType::Float,
           c10::ScalarType::Double,
           c10::ScalarType::Int,
           c10::ScalarType::Long,
       }) {
    EXPECT_EQ(
        at::toAccumulateType(t, c10::DeviceType::PrivateUse1), cpu_acc_type(t));
  }

  // WarnOnce fires per-process; with WarnAlways(true) every call warns, so the
  // captured log records at least one warning after multiple calls and
  // contains the documented substr.
  torch::test::WarningCapture capture;
  c10::WarningUtils::WarnAlways warn_always(true);
  EXPECT_EQ(
      at::toAccumulateType(
          c10::ScalarType::Float, c10::DeviceType::PrivateUse1),
      cpu_acc_type(c10::ScalarType::Float));
  EXPECT_FALSE(capture.messages().empty());
  EXPECT_NE(capture.str().find("accumulate-type mapping"), std::string::npos);
}
