#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <ATen/test/allocator_clone_test.h>

using namespace at;

void XLAFree(void *ptr) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(ptr);
}

void* XLAMalloc(ptrdiff_t size) {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  return malloc(size);
}

struct XLAAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) override {
    auto* ptr = XLAMalloc(size);
    return {ptr, ptr, &XLAFree, at::DeviceType::XLA};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &XLAFree;
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
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

TEST(XlaTensorTest, test_allocator_clone) {
  if (!at::hasXLA()) {
    return;
  }
  XLAAllocator allocator;
  test_allocator_clone(&allocator);
}
