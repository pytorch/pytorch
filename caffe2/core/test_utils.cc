#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

#include "test_utils.h"

namespace caffe2 {
namespace testing {

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
