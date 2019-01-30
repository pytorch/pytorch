#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

TEST(TpuTensorTest, TestNoStorage) {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      XLATensorId(),
      caffe2::TypeMeta::Make<float>(),
      /*allocator=*/nullptr,
      /*is_variable=*/false);
  at::Tensor t(std::move(tensor_impl));
  ASSERT_FALSE(t.storage());
}
