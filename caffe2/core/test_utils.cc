#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

#include "test_utils.h"

#include <gtest/gtest.h>

namespace caffe2 {
namespace testing {

void assertTensorListEquals(
    const std::vector<std::string>& tensorNames,
    const Workspace& workspace1,
    const Workspace& workspace2) {
  for (const string& tensorName : tensorNames) {
    EXPECT_TRUE(workspace1.HasBlob(tensorName));
    EXPECT_TRUE(workspace2.HasBlob(tensorName));
    auto& tensor1 = getTensor(workspace1, tensorName);
    auto& tensor2 = getTensor(workspace2, tensorName);
    if (tensor1.IsType<float>()) {
      EXPECT_TRUE(tensor2.IsType<float>());
      assertsTensorEquals<float>(tensor1, tensor2);
    } else if (tensor1.IsType<int>()) {
      EXPECT_TRUE(tensor2.IsType<int>());
      assertsTensorEquals<int>(tensor1, tensor2);
    }
    // Add more types if needed.
  }
}

const caffe2::Tensor& getTensor(
    const caffe2::Workspace& workspace,
    const std::string& name) {
  return workspace.GetBlob(name)->Get<caffe2::Tensor>();
}

caffe2::Tensor* createTensor(
    const std::string& name,
    caffe2::Workspace* workspace) {
  return BlobGetMutableTensor(workspace->CreateBlob(name), caffe2::CPU);
}

caffe2::OperatorDef* createOperator(
    const std::string& type,
    const std::vector<string>& inputs,
    const std::vector<string>& outputs,
    caffe2::NetDef* net) {
  auto* op = net->add_op();
  op->set_type(type);
  for (const auto& in : inputs) {
    op->add_input(in);
  }
  for (const auto& out : outputs) {
    op->add_output(out);
  }
  return op;
}

NetMutator& NetMutator::newOp(
    const std::string& type,
    const std::vector<string>& inputs,
    const std::vector<string>& outputs) {
  createOperator(type, inputs, outputs, net_);
  return *this;
}

} // namespace testing
} // namespace caffe2
