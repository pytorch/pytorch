#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

TEST(BackendExtensionTest, TestRegisterOp) {
  Storage storage = Storage(
      caffe2::TypeMeta::Make<float>(), 0, at::DataPtr(), nullptr, false);
  Tensor tensor = Tensor(storage, FPGATensorId(), false);
  hinge_embedding_loss(tensor);
}
