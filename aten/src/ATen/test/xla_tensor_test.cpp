#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

void XLAFree(void *ptr) {
  free(ptr);
}

void* XLAMalloc(ptrdiff_t size) {
  return malloc(size);
}

struct XLAAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    auto* ptr = XLAMalloc(size);
    return {ptr, ptr, &XLAFree, at::DeviceType::XLA};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &XLAFree;
  }
};

TEST(XlaTensorTest, TestNoStorage) {
  XLAAllocator allocator;
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      DispatchKey::XLA,
      caffe2::TypeMeta::Make<float>(),
      at::Device(DeviceType::XLA, 0));
  at::Tensor t(std::move(tensor_impl));
  ASSERT_TRUE(t.device() == at::Device(DeviceType::XLA, 0));
}
