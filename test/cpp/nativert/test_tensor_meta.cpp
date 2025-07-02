#include <gtest/gtest.h>
#include <torch/nativert/graph/TensorMeta.h>

namespace torch::nativert {
TEST(TensorMetaTest, ScalarTypeConversion) {
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::FLOAT),
      c10::ScalarType::Float);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::INT),
      c10::ScalarType::Int);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::HALF),
      c10::ScalarType::Half);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::COMPLEXHALF),
      c10::ScalarType::ComplexHalf);
  EXPECT_EQ(
      convertJsonScalarType(torch::_export::ScalarType::BFLOAT16),
      c10::ScalarType::BFloat16);
  EXPECT_THROW(
      convertJsonScalarType(static_cast<torch::_export::ScalarType>(100)),
      c10::Error);
}
TEST(TensorMetaTest, MemoryFormatConversion) {
  EXPECT_EQ(
      convertJsonMemoryFormat(torch::_export::MemoryFormat::ContiguousFormat),
      c10::MemoryFormat::Contiguous);
  EXPECT_EQ(
      convertJsonMemoryFormat(torch::_export::MemoryFormat::ChannelsLast),
      c10::MemoryFormat::ChannelsLast);
  EXPECT_EQ(
      convertJsonMemoryFormat(torch::_export::MemoryFormat::PreserveFormat),
      c10::MemoryFormat::Preserve);
  EXPECT_THROW(
      convertJsonMemoryFormat(static_cast<torch::_export::MemoryFormat>(100)),
      c10::Error);
}

TEST(TensorMetaTest, LayoutConversion) {
  EXPECT_EQ(
      convertJsonLayout(torch::_export::Layout::Strided), c10::Layout::Strided);
  EXPECT_EQ(
      convertJsonLayout(torch::_export::Layout::SparseCsr),
      c10::Layout::SparseCsr);
  EXPECT_EQ(
      convertJsonLayout(torch::_export::Layout::_mkldnn), c10::Layout::Mkldnn);
  EXPECT_THROW(
      convertJsonLayout(static_cast<torch::_export::Layout>(100)), c10::Error);
}
TEST(TensorMetaTest, DeviceConversion) {
  torch::_export::Device cpu_device;
  cpu_device.set_type("cpu");
  EXPECT_EQ(convertJsonDevice(cpu_device), c10::Device(c10::DeviceType::CPU));
  torch::_export::Device cuda_device;
  cuda_device.set_type("cuda");
  cuda_device.set_index(0);
  EXPECT_EQ(
      convertJsonDevice(cuda_device), c10::Device(c10::DeviceType::CUDA, 0));
}

} // namespace torch::nativert
