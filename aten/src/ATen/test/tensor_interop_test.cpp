#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>

TEST(TestTensorInterop, Caffe2ToPytorchSimpleLegacy) {
  caffe2::Tensor c2_tensor(caffe2::CPU);
  c2_tensor.Resize(4, 4);
  auto data = c2_tensor.mutable_data<int64_t>();
  for (int64_t i = 0; i < 16; i++) {
    data[i] = i;
  }

  // TODO: find out why calling data on tensor doesn't work
  at::Tensor at_tensor(c2_tensor.getIntrusivePtr());
  at::TensorImpl* impl = at_tensor.unsafeGetTensorImpl();

  auto it = impl->data<int64_t>();
  for (int64_t i = 0; i < 16; i++) {
    ASSERT_EQ(it[i], i);
  }
}

TEST(TestTensorInterop, Caffe2ToPytorchSimple) {
  caffe2::Tensor c2_tensor = caffe2::empty({4, 4}, at::kLong);
  auto data = c2_tensor.mutable_data<int64_t>();
  for (int64_t i = 0; i < 16; i++) {
    data[i] = i;
  }
  at::Tensor at_tensor(c2_tensor.getIntrusivePtr());
  at::TensorImpl* impl = at_tensor.unsafeGetTensorImpl();

  auto it = impl->data<int64_t>();
  for (int64_t i = 0; i < 16; i++) {
    ASSERT_EQ(it[i], i);
  }
}

TEST(TestTensorInterop, Caffe2ToPytorchOp) {
  caffe2::Tensor c2_tensor(caffe2::CPU);
  c2_tensor.Resize(3, 3);
  auto data = c2_tensor.mutable_data<int64_t>();
  for (int64_t i = 0; i < 9; i++) {
    data[i] = i;
  }
  at::Tensor at_tensor(c2_tensor.getIntrusivePtr());

  ASSERT_EQ(at::sum(at_tensor).item<int64_t>(), 36);
}

TEST(TestTensorInterop, Caffe2ToPytorchUnsupportedDevice) {
  caffe2::Tensor c2_tensor(caffe2::IDEEP);
  at::Tensor at_tensor(c2_tensor.getIntrusivePtr());
  ASSERT_ANY_THROW(at::sum(at_tensor));
}

TEST(TestTensorInterop, PytorchToCaffe2Op) {
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat));
  auto at_tensor_b = at::ones({5, 5}, at::dtype(at::kFloat));
  auto at_tensor_c = at::ones({5, 5}, at::dtype(at::kFloat));

  auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), at_tensor_a.getIntrusivePtr());
  auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), at_tensor_b.getIntrusivePtr());

  // Test ShareData as well
  {
    auto c2_tensor_c = XBlobGetMutableTensor(workspace.CreateBlob("c"), {0}, at::kCPU);
    c2_tensor_c.ResizeLike(at_tensor_c.getIntrusivePtr());
    c2_tensor_c.ShareData(at_tensor_c.getIntrusivePtr());
  }

  {
    auto op = net.add_op();
    op->set_type("Sum");
    op->add_input("a");
    op->add_input("b");
    op->add_input("c");
    op->add_output("d");
  }

  workspace.RunNetOnce(net);

  auto result = XBlobGetMutableTensor(workspace.CreateBlob("d"), {5, 5}, at::kCPU);

  auto it = result.data<float>();
  for (int64_t i = 0; i < 25; i++) {
    ASSERT_EQ(it[i], 3.0);
  }
  at::Tensor at_result(result.getIntrusivePtr());
  ASSERT_EQ(at::sum(at_result).item<float>(), 75);
}

TEST(TestTensorInterop, PytorchToCaffe2SharedStorage) {
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat));
  auto at_tensor_b = at_tensor_a.view({5, 5});

  auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), at_tensor_a.getIntrusivePtr());
  auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), at_tensor_b.getIntrusivePtr());

  {
    auto op = net.add_op();
    op->set_type("Add");
    op->add_input("a");
    op->add_input("b");
    op->add_output("c");
  }

  workspace.RunNetOnce(net);

  auto result = XBlobGetMutableTensor(workspace.CreateBlob("c"), {5, 5}, at::kCPU);
  auto it = result.data<float>();
  for (int64_t i = 0; i < 25; i++) {
    ASSERT_EQ(it[i], 2.0);
  }
  at::Tensor at_result(result.getIntrusivePtr());
  ASSERT_EQ(at::sum(at_result).item<float>(), 50);
}

TEST(TestTensorInterop, PytorchToCaffe2Strided) {
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  auto at_tensor = at::ones({5, 5}, at::dtype(at::kFloat)).t();
  auto* c2_tensor = BlobSetTensor(workspace.CreateBlob("blob"), at_tensor.getIntrusivePtr());

  {
    auto op = net.add_op();
    op->set_type("Sum");
    op->add_input("blob");
    op->add_output("out");
  }

  ASSERT_ANY_THROW(workspace.RunNetOnce(net));
}
