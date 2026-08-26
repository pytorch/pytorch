#include <gtest/gtest.h>

#include <torch/headeronly/util/AccumulateType.h>

namespace torch {
namespace aot_inductor {

TEST(TestAccumulateType, TestCPU) {
  using torch::headeronly::acc_type;

  static_assert(
      std::is_same_v<acc_type<torch::headeronly::Half, false>, float>);
  static_assert(std::is_same_v<acc_type<float, false>, double>);
  static_assert(std::is_same_v<acc_type<double, false>, double>);
  static_assert(std::is_same_v<acc_type<int64_t, false>, int64_t>);
  static_assert(std::is_same_v<acc_type<bool, false>, bool>);
}

TEST(TestAccumulateType, TestCUDA) {
  using torch::headeronly::acc_type;

  static_assert(std::is_same_v<acc_type<torch::headeronly::Half, true>, float>);
  static_assert(std::is_same_v<acc_type<float, true>, float>);
  static_assert(std::is_same_v<acc_type<double, true>, double>);
  static_assert(std::is_same_v<acc_type<int64_t, true>, int64_t>);
}

TEST(TestAccumulateType, TestAccumulateTypeDevice) {
  using torch::headeronly::acc_type_device;
  using torch::headeronly::AccumulateType;
  using torch::headeronly::AccumulateTypeDevice;

  static_assert(
      std::is_same_v<acc_type_device<float, c10::DeviceType::CPU>, double>);
  static_assert(std::is_same_v<
                AccumulateTypeDevice<float, c10::DeviceType::CUDA>::type,
                float>);
  static_assert(std::is_same_v<AccumulateType<float, false>::type, double>);
}

} // namespace aot_inductor
} // namespace torch
