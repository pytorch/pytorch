#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/ATenDispatch.h>

using namespace at;

static int test_int;

Tensor empty_override(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> optional_memory_format) {
  test_int = 1;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(nullptr, Device(DeviceType::MSNPU, 1)), nullptr, false),
      MSNPUTensorId());
  tensor_impl->set_sizes_contiguous(size);
  return Tensor(std::move(tensor_impl));
}

Tensor add_override(const Tensor & a, const Tensor & b , Scalar c) {
  test_int = 2;
  return a;
}

Tensor fake_xla_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  //AT_ERROR("Backend should either override this function or use thnn implementation");
  std::cout << "hello world" << std::endl;
  test_int = 3;
  return input;
}

struct MSNPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::MSNPU;
  MSNPUGuardImpl() {}
  MSNPUGuardImpl(DeviceType t) {
    AT_ASSERT(t == DeviceType::MSNPU);
  }
  DeviceType type() const override {
    return DeviceType::MSNPU;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MSNPU);
    AT_ASSERT(d.index() == 1);
    return d;
  }
  Device getDevice() const override {
    return Device(DeviceType::MSNPU, 1);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MSNPU);
    AT_ASSERT(d.index() == 1);
  }
  void uncheckedSetDevice(Device d) const noexcept override {
  }
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MSNPU, 1));
  }
  Stream exchangeStream(Stream s) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MSNPU, 1));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
};

constexpr DeviceType MSNPUGuardImpl::static_type;
C10_REGISTER_GUARD_IMPL(MSNPU, MSNPUGuardImpl);


TEST(BackendExtensionTest, TestRegisterOp) {
  EXPECT_ANY_THROW(empty({5, 5}, at::kMSNPU));
  globalATenDispatch().registerOp(
    Backend::MSNPU,
    "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
    &empty_override);
  Tensor a = empty({5, 5}, at::kMSNPU);
  ASSERT_EQ(a.device().type(), at::kMSNPU);
  ASSERT_EQ(a.device().index(), 1);
  ASSERT_EQ(a.dtype(), caffe2::TypeMeta::Make<float>());
  ASSERT_EQ(test_int, 1);

  Tensor b = empty_like(a, at::kMSNPU);
  ASSERT_EQ(b.device().type(), at::kMSNPU);
  ASSERT_EQ(b.device().index(), 1);
  ASSERT_EQ(b.dtype(), caffe2::TypeMeta::Make<float>());

  EXPECT_ANY_THROW(add(a, b));
  globalATenDispatch().registerOp(
    Backend::MSNPU,
    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    &add_override);
  add(a, b);
  ASSERT_EQ(test_int, 2);

  // Ensure that non-MSNPU operator still works
  Tensor d = empty({5, 5}, at::kCPU);
  ASSERT_EQ(d.device().type(), at::kCPU);

  // Attempt to register on a schema that has already has a function
  EXPECT_ANY_THROW(
    globalATenDispatch().registerOp(
      Backend::MSNPU,
      "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
      &empty_override)
  );
}

TEST(BackendExtensionTest, TestConvGeneric) {
  //EXPECT_ANY_THROW(empty({5, 5}, at::kMSNPU));
  globalATenDispatch().registerOp(
    Backend::MSNPU,
    "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
    &empty_override);

  globalATenDispatch().registerOp(
    Backend::MSNPU,
    "aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
    &fake_xla_convolution);
  Tensor input = empty({2, 4, 10, 2}, at::kMSNPU);
  Tensor weight= empty({6, 4, 3, 2}, at::kMSNPU);
  Tensor bias = empty({6}, at::kMSNPU);
  Tensor output = conv1d(input, weight, bias, 2, 0, 1, 1);

  std::cout << "DONE" << std::endl;

}
