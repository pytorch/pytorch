#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include <torch/csrc/jit/runtime/operator.h>

using namespace at;

static int test_int;

Tensor empty_override(IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> layout,
                      c10::optional<Device> device, c10::optional<bool> pin_memory, c10::optional<MemoryFormat> optional_memory_format) {
  test_int = 1;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          Storage::use_byte_size_t(),
          0,
          at::DataPtr(nullptr, Device(DeviceType::MSNPU, 1)),
          nullptr,
          false),
      DispatchKey::MSNPU,
      caffe2::TypeMeta::Make<float>());
  return Tensor(std::move(tensor_impl));
}

Tensor add_override(const Tensor & a, const Tensor & b , const Scalar& c) {
  test_int = 2;
  return a;
}

Tensor empty_strided_override(
  IntArrayRef size,
  IntArrayRef stride,
  c10::optional<c10::ScalarType> dtype,
  c10::optional<c10::Layout> layout,
  c10::optional<c10::Device> device,
  c10::optional<bool> pin_memory) {

  return empty_override(size, dtype, layout, device, pin_memory, c10::nullopt);
}

TORCH_LIBRARY_IMPL(aten, MSNPU, m) {
  m.impl("aten::empty.memory_format",  empty_override);
  m.impl("aten::empty_strided",        empty_strided_override);
  m.impl("aten::add.Tensor",           add_override);
}

TEST(BackendExtensionTest, TestRegisterOp) {
  Tensor a = empty({5, 5}, at::kMSNPU);
  ASSERT_EQ(a.device().type(), at::kMSNPU);
  ASSERT_EQ(a.device().index(), 1);
  ASSERT_EQ(a.dtype(), caffe2::TypeMeta::Make<float>());
  ASSERT_EQ(test_int, 1);

  Tensor b = empty_like(a, at::kMSNPU);
  ASSERT_EQ(b.device().type(), at::kMSNPU);
  ASSERT_EQ(b.device().index(), 1);
  ASSERT_EQ(b.dtype(), caffe2::TypeMeta::Make<float>());

  add(a, b);
  ASSERT_EQ(test_int, 2);

  // Ensure that non-MSNPU operator still works
  Tensor d = empty({5, 5}, at::kCPU);
  ASSERT_EQ(d.device().type(), at::kCPU);
}
