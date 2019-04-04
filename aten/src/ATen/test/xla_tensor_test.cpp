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
  auto storage = Storage(caffe2::TypeMeta::Make<float>(), 0, &allocator, true);
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      std::move(storage),
      XLATensorId(),
      /*is_variable=*/false);
  at::Tensor t(std::move(tensor_impl));
  ASSERT_TRUE(t.device() == DeviceType::XLA);
}
