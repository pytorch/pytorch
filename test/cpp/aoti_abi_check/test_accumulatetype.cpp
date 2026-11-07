#include <gtest/gtest.h>

#include <torch/headeronly/core/AccumulateType.h>

#include <type_traits>

using torch::headeronly::acc_type;
using torch::headeronly::acc_type_device;
using torch::headeronly::AccumulateType;
using torch::headeronly::AccumulateTypeDevice;
using torch::headeronly::BFloat16;
using torch::headeronly::complex;
using torch::headeronly::DeviceType;
using torch::headeronly::Float8_e4m3fn;
using torch::headeronly::Float8_e4m3fnuz;
using torch::headeronly::Float8_e5m2;
using torch::headeronly::Float8_e5m2fnuz;
using torch::headeronly::Half;

TEST(TestAccumulateType, CUDA) {
  static_assert(std::is_same_v<acc_type<Half, true>, float>);
  static_assert(std::is_same_v<acc_type<BFloat16, true>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e5m2, true>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e4m3fn, true>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e5m2fnuz, true>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e4m3fnuz, true>, float>);
  static_assert(std::is_same_v<acc_type<float, true>, float>);
  static_assert(std::is_same_v<acc_type<double, true>, double>);
  static_assert(std::is_same_v<acc_type<int8_t, true>, int64_t>);
  static_assert(std::is_same_v<acc_type<uint8_t, true>, int64_t>);
  static_assert(std::is_same_v<acc_type<char, true>, int64_t>);
  static_assert(std::is_same_v<acc_type<int16_t, true>, int64_t>);
  static_assert(std::is_same_v<acc_type<int32_t, true>, int64_t>);
  static_assert(std::is_same_v<acc_type<int64_t, true>, int64_t>);
  static_assert(std::is_same_v<acc_type<bool, true>, bool>);
  static_assert(std::is_same_v<acc_type<complex<Half>, true>, complex<float>>);
  static_assert(std::is_same_v<acc_type<complex<float>, true>, complex<float>>);
  static_assert(
      std::is_same_v<acc_type<complex<double>, true>, complex<double>>);
  EXPECT_EQ(typeid(AccumulateType<Half, true>::type), typeid(float));
}

TEST(TestAccumulateType, CPU) {
  static_assert(std::is_same_v<acc_type<Half, false>, float>);
  static_assert(std::is_same_v<acc_type<BFloat16, false>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e5m2, false>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e4m3fn, false>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e5m2fnuz, false>, float>);
  static_assert(std::is_same_v<acc_type<Float8_e4m3fnuz, false>, float>);
  static_assert(std::is_same_v<acc_type<float, false>, double>);
  static_assert(std::is_same_v<acc_type<double, false>, double>);
  static_assert(std::is_same_v<acc_type<int8_t, false>, int64_t>);
  static_assert(std::is_same_v<acc_type<uint8_t, false>, int64_t>);
  static_assert(std::is_same_v<acc_type<char, false>, int64_t>);
  static_assert(std::is_same_v<acc_type<int16_t, false>, int64_t>);
  static_assert(std::is_same_v<acc_type<int32_t, false>, int64_t>);
  static_assert(std::is_same_v<acc_type<int64_t, false>, int64_t>);
  static_assert(std::is_same_v<acc_type<bool, false>, bool>);
  static_assert(std::is_same_v<acc_type<complex<Half>, false>, complex<float>>);
  static_assert(
      std::is_same_v<acc_type<complex<float>, false>, complex<double>>);
  static_assert(
      std::is_same_v<acc_type<complex<double>, false>, complex<double>>);
  EXPECT_EQ(typeid(AccumulateType<float, false>::type), typeid(double));
}

TEST(TestAccumulateType, Device) {
  static_assert(
      std::is_same_v<acc_type_device<float, DeviceType::CPU>, double>);
  static_assert(
      std::is_same_v<acc_type_device<float, DeviceType::CUDA>, float>);
  static_assert(std::is_same_v<acc_type_device<float, DeviceType::MPS>, float>);
  static_assert(std::is_same_v<acc_type_device<float, DeviceType::XPU>, float>);
  static_assert(
      std::is_same_v<acc_type_device<double, DeviceType::MPS>, float>);
  static_assert(std::is_same_v<
                acc_type_device<complex<double>, DeviceType::MPS>,
                complex<float>>);
  static_assert(
      std::is_same_v<acc_type_device<double, DeviceType::XPU>, double>);
  static_assert(
      std::is_same_v<acc_type_device<int8_t, DeviceType::MPS>, int64_t>);
  static_assert(std::is_same_v<acc_type_device<Half, DeviceType::XPU>, float>);
  static_assert(std::is_same_v<
                acc_type<float, false>,
                acc_type_device<float, DeviceType::CPU>>);
  static_assert(std::is_same_v<
                acc_type<float, true>,
                acc_type_device<float, DeviceType::CUDA>>);
  EXPECT_EQ(
      typeid(AccumulateTypeDevice<double, DeviceType::MPS>::type),
      typeid(float));
}
