#include <gtest/gtest.h>

#include <torch/headeronly/util/AccumulateType.h>

#include <type_traits>

namespace torch {
namespace aot_inductor {

TEST(TestAccumulateType, TestAccType) {
  using torch::headeronly::acc_type;
  using torch::headeronly::acc_type_device;
  using torch::headeronly::AccumulateType;
  using torch::headeronly::AccumulateTypeDevice;
  using torch::headeronly::DeviceType;
  using torch::headeronly::Half;

  static_assert(
      std::is_same_v<acc_type<Half, false>, float>,
      "Half CPU acc_type should be float");
  static_assert(
      std::is_same_v<acc_type<Half, true>, float>,
      "Half CUDA acc_type should be float");
  static_assert(
      std::is_same_v<acc_type_device<Half, DeviceType::CPU>, float>,
      "Half CPU acc_type_device should be float");
  static_assert(
      std::is_same_v<typename AccumulateType<Half, false>::type, float>,
      "Half CPU AccumulateType should be float");
  static_assert(
      std::is_same_v<
          typename AccumulateTypeDevice<Half, DeviceType::CPU>::type,
          float>,
      "Half CPU AccumulateTypeDevice should be float");
}

} // namespace aot_inductor
} // namespace torch
